"""Merging volunteer runs within a family (issue #20, milestone 2): rounds of workers trained
from a shared start, each checked and gated, then merged by a recorded rule.

A round starts from an incumbent: θ₀ for round 1, then the previous round's merged checkpoint.
Each worker trains from it on a disjoint shard of the training slices and contributes
τᵢ = θᵢ − θ_start:

    python -m tracks.llm.train --anchor v1 --manifest tracks/llm/data/slices/index.json \\
      --allow-within-session --variants real --shard 1/3 --seed 42 --output runs/r1-s1
    python -m tracks.llm.merge round --output runs/round1 runs/r1-s*/real.pt
    python -m tracks.llm.train ... --start runs/round1/merged.pt --shard 1/3 --output runs/r2-s1
    python -m tracks.llm.merge round --start runs/round1/merged.pt --output runs/round2 runs/r2-s*/real.pt

    python -m tracks.llm.merge replay runs/round1/round.json   # recompute the round from its record
    python -m tracks.llm.merge score runs/round1/merged.pt     # test and control scores, for reporting only

The rule (all optional; the default is the plain baseline, an average by training targets):

- DARE (--dare-p): each delta is compressed as a volunteer would send it, dropping a fraction p
  of its entries and storing the rest as int8 with sparse indices, within a result card's
  16 KB. The round merges and gates what the payload rebuilds.
- TIES (--merge ties): trim each delta to its largest entries, elect each parameter's sign, and
  average only the nonzero entries that agree with it.
- Greedy soup (--soup greedy): workers are tried best first by validation, each kept only if
  the merge improves with it, so a worker that barely moved can't dilute the others.
- DiLoCo (--outer-lr, --outer-momentum): the merged delta is a pseudo-gradient for an outer
  Nesterov step, whose momentum carries from round to round in the merged checkpoint.

The gate: each worker and the merge must score within a tolerance of the incumbent on
validation. The other family's held-out validation scan (the selection control) is recorded
for each, as a diagnostic, not a veto: its bits, and how much the model gains from the real
order of its tokens over shuffled orders. Its likelihood alone can't show a model found
structure there: matching token frequencies or calibration moves it too. On the published
off-source scans, neither number separates a model trained on them from a Voyager model
(MERGE_RESULTS.md). The other family's test scan is the reporting control, scored only
afterwards, like the test split. A round never reads it or the test split.
Its record names every contributor (by file hash), what became of it and why, the rule, and
the merged weights' digest, so anyone holding the same files can replay it.
"""

import argparse
import base64
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import NamedTuple

import numpy as np
import torch

from .anchor import ANCHORS, ROOT, codebook_digest, difference, flatten, load_anchor, weights_digest
from .data import control_family, dataset_digest, load_slices, split_records
from .model import SignalLM, evaluate
from .runtime import atomic_output, require_new_output, training_runtime

TOLERANCE_BITS = 0.02  # how much worse than the incumbent on validation a worker or merge may be
MAX_ABS_DELTA = 10.0  # a sanity bound on any one parameter's change: θ₀'s weights are of order 1
CARD_BUDGET_BYTES = 16 * 1024  # a result card's limit (MAX_CARD_BYTES in web/js/community.js)
ORDER_SHUFFLES = 19  # shuffled copies of the selection control its real order is compared with

# The plain baseline. A round records the whole rule it used, so it can be replayed.
RULE = {
    "merge": "average",  # or "ties"
    "ties_keep": 1.0,  # TIES: the fraction of each delta's entries kept, by magnitude
    "dare": None,  # or {"method": "random" or "magnitude", "p": the fraction dropped}
    "soup": "all",  # or "greedy"
    "outer": {"lr": 1.0, "momentum": 0.0},  # lr 1 with no momentum is θ + τ̄, plain merging
    "tolerance_bits": TOLERANCE_BITS,
    "max_abs_delta": MAX_ABS_DELTA,
}


def check_rule(rule=None):
    """The baseline rule with `rule`'s changes, refusing anything it doesn't define."""
    out = {**RULE, **(rule or {})}
    finite = lambda x, low=0.0: isinstance(x, (int, float)) and math.isfinite(x) and x >= low
    if set(out) != set(RULE):
        raise ValueError(f"Unknown rule settings: {', '.join(sorted(set(out) - set(RULE)))}")
    if out["merge"] not in ("average", "ties"):
        raise ValueError("The merge is average or ties")
    if not (finite(out["ties_keep"]) and 0 < out["ties_keep"] <= 1) or (out["merge"] == "average" and out["ties_keep"] != 1):
        raise ValueError("TIES keeps a fraction in (0, 1]; an average trims nothing")
    dare = out["dare"]
    if dare is not None and not (isinstance(dare, dict) and set(dare) == {"method", "p"} and dare["method"] in ("random", "magnitude")
                                 and finite(dare["p"]) and dare["p"] < 1):
        raise ValueError("DARE takes a method (random or magnitude) and a drop fraction p in [0, 1)")
    if out["soup"] not in ("all", "greedy"):
        raise ValueError("The soup is all or greedy")
    outer = out["outer"]
    if not (isinstance(outer, dict) and set(outer) == {"lr", "momentum"} and finite(outer["lr"]) and outer["lr"] > 0
            and finite(outer["momentum"]) and outer["momentum"] < 1):
        raise ValueError("The outer step takes a positive lr and a momentum in [0, 1)")
    if not (finite(out["tolerance_bits"]) and finite(out["max_abs_delta"]) and out["max_abs_delta"] > 0):
        raise ValueError("The tolerance must be ≥ 0 and the bound positive")
    return out


def parse_shard(text):
    """'2/3' -> (2, 3): the second of three shards."""
    try:
        index, count = (int(part) for part in str(text).split("/"))
    except ValueError:
        raise ValueError(f"--shard takes i/n, like 2/3; got {text!r}") from None
    if not 1 <= index <= count:
        raise ValueError(f"--shard {text}: need 1 ≤ i ≤ n")
    return index, count


def shard_records(records, index, count):
    """Shard `index` of `count` (1-based) of the training slices: every count-th slice in a fixed
    order (by content), so shards are disjoint, cover the split together, and don't depend on the
    manifest's order."""
    ordered = sorted(records, key=lambda r: r["sha256"])
    if len(ordered) < count:
        raise ValueError(f"{count} shards need at least {count} training slices; this split has {len(ordered)}")
    return ordered[index - 1::count]


def anchor_named(identity):
    """The pinned anchor a record or checkpoint names, refusing one this code doesn't pin."""
    for name, spec in ANCHORS.items():
        if spec["id"] == (identity or {}).get("anchor"):
            anchor = load_anchor(name)
            if anchor.identity() == identity:
                return anchor
    raise ValueError("It names an anchor this code doesn't pin")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, anchor):
    """A checkpoint trained or merged from this anchor, with its tokenizer and model settings."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eiathome-signal-lm" or checkpoint.get("version") != 1:
        raise ValueError(f"{path}: not a supported signal-LM checkpoint")
    if checkpoint.get("anchor") != anchor.identity():
        raise ValueError(f"{path}: not trained from anchor {anchor.name}")
    if codebook_digest(checkpoint["tokenizer"]) != anchor.spec["codebook_sha256"] or checkpoint.get("config") != anchor.spec["model"]:
        raise ValueError(f"{path}: its tokenizer or model settings aren't anchor {anchor.name}'s")
    if not isinstance(checkpoint.get("merge"), dict):
        raise ValueError(f"{path}: has no round record (train it with --anchor, after this change)")
    return checkpoint


def lineage(run):
    """What every round of a chain shares, from a run's report or checkpoint: its family, its
    dataset, how that was split, and the held-out slices."""
    held = [entry for name in ("validation", "test") for entry in run["splits"][name]]
    families = {entry.get("family") for entry in held}
    if len(families) != 1:
        raise ValueError("Its validation and test slices aren't of one family")
    return {"family": families.pop(), "dataset_sha256": run["dataset_sha256"], "evaluation": run["evaluation"],
            "validation": run["splits"]["validation"], "test": run["splits"]["test"]}


class Start(NamedTuple):
    """What a round starts from: θ₀ for round 1 (in no chain yet), or the last round's merge,
    with the outer rule its chain uses and that rule's momentum (None without any)."""
    state: OrderedDict
    round: int
    weights_sha256: str
    lineage: dict | None
    outer: dict | None = None
    momentum: OrderedDict | None = None


def round_start(anchor, path=None):
    """θ₀, or a merged round's checkpoint to start the next round from. Refuses one whose weights
    or momentum don't match the digests its round recorded."""
    if path is None:
        return Start(anchor.theta0, 0, anchor.spec["theta0_sha256"], None)
    checkpoint = load_checkpoint(path, anchor)
    merge = checkpoint["merge"]
    if not merge.get("merged") or not isinstance(merge.get("round"), int) or merge["round"] < 1:
        raise ValueError(f"{path}: not a merged round's checkpoint")
    state = checkpoint["model"]
    difference(state, anchor.theta0)  # the anchor's parameters, in order
    digest = weights_digest(state)
    if digest != merge.get("weights_sha256"):
        raise ValueError(f"{path}: its weights don't match the digest its round recorded")
    momentum = checkpoint.get("outer_momentum")
    if momentum is not None:
        difference(momentum, anchor.theta0)
    if (None if momentum is None else weights_digest(momentum)) != merge.get("momentum_sha256"):
        raise ValueError(f"{path}: its outer momentum doesn't match the digest its round recorded")
    return Start(state, merge["round"], digest, lineage(checkpoint), merge.get("outer"), momentum)


def check_continuation(path, start, chain):
    """Refuses to continue a chain on another family, dataset or split: its training slices
    could then be another run's held-out ones, or its control a family it trained on."""
    if start.lineage is not None and start.lineage != chain:
        differ = [key for key in chain if chain[key] != start.lineage[key]]
        raise ValueError(f"{path} was merged from {start.lineage['family']} runs, and this one's {', '.join(differ)} "
                         "differ: a round chain keeps one family, dataset and split")


def held_out(anchor, family, evaluation, manifest=None, dataset_sha256=None):
    """The family's split as train.py made it, from the manifest the runs were trained on."""
    records = load_slices(manifest or ROOT / anchor.spec["manifest"])
    if dataset_sha256 is not None and dataset_digest(records) != dataset_sha256:
        raise ValueError("This manifest isn't the one the runs were trained on")
    partitions, independent = split_records(records, family, allow_within_session=evaluation != "independent_sessions")
    if ("independent_sessions" if independent else "within_session_scan_holdout") != evaluation:
        raise ValueError("The manifest no longer splits as the workers' runs did")
    return records, partitions


def controls(records, family, evaluation):
    """The other family's held-out slices, split as the family's own are: its validation part
    selects (the gate), its test part reports. Neither is trained on, and the codebook was fitted
    on its training part only. None if the manifest has no other family that splits."""
    other = control_family(records, family)
    if other is None:
        return None
    try:
        partitions, _ = split_records(records, other, allow_within_session=evaluation != "independent_sessions")
    except ValueError:
        return None
    return {"family": other, "selection": partitions["validation"], "reporting": partitions["test"]}


def recorded_shard(path, merge):
    """A worker's shard as (i, n); (1, 1) for one trained on the whole training split."""
    shard = merge.get("shard")
    if shard is None:
        return 1, 1
    if not (isinstance(shard, list) and len(shard) == 2 and all(type(v) is int for v in shard) and 1 <= shard[0] <= shard[1]):
        raise ValueError(f"{path}: its shard isn't i of n")
    return tuple(shard)


def same_slices(a, b):
    canonical = lambda entries: sorted(json.dumps(e, sort_keys=True) for e in entries)
    return canonical(a) == canonical(b)


def worker_problem(checkpoint, start, max_abs_delta, tau, shard, slices):
    """Why a worker's delta can't join the round, or None. slices: the training slices its shard
    assigns, from the round's own manifest."""
    merge = checkpoint["merge"]
    if merge.get("merged"):
        return "it's a merged checkpoint, not a worker's"
    if checkpoint.get("variant") != "real":
        return f"it's the {checkpoint.get('variant')} surrogate, not the real stream"
    if merge.get("round") != start.round + 1 or merge.get("start_sha256") != start.weights_sha256:
        return f"it was trained for round {merge.get('round')} from another start"
    if not same_slices(checkpoint["splits"]["train"] or [], [r["entry"] for r in slices]):
        return f"its training slices aren't shard {shard[0]}/{shard[1]}'s"
    targets = sum(len(r["frames"]) - 1 for r in slices)
    if merge.get("training_targets") != targets:
        return f"it records {merge.get('training_targets')!r} training targets, but shard {shard[0]}/{shard[1]} has {targets}"
    return delta_problem(tau, max_abs_delta)


def delta_problem(tau, max_abs_delta, what="its delta"):
    """Why a delta can't reach the model, or None: it isn't finite, or an entry is out of bound."""
    if not all(torch.isfinite(v).all() for v in tau.values()):
        return f"{what} isn't finite"
    largest = max(v.abs().max().item() for v in tau.values())
    if largest > max_abs_delta:
        return f"{what} reaches {largest:.3g}, beyond the bound of {max_abs_delta:g}"
    return None


def order_gain(model, sequences, shuffles=ORDER_SHUFFLES):
    """A diagnostic of what a model gets from the order of a control's tokens: its bits on the
    real order, how many fewer than on average over `shuffles` copies whose targets (every token
    but each slice's first) are shuffled within the slice, and a permutation p-value. A model
    that ignores context scores the same on every copy, since the targets are the same tokens.
    For tokens with no temporal structure, the real order is one of shuffles + 1 exchangeable
    orders, so the share of them scoring at least as well as it is uniformly distributed."""
    real = evaluate(model, sequences)["bits_per_token"]
    rng = np.random.default_rng(0)
    copies = [evaluate(model, [np.concatenate([s[:1], rng.permutation(s[1:])]) for s in sequences])["bits_per_token"]
              for _ in range(shuffles)]
    return {"bits": real, "order_gain_bits": statistics.fmean(copies) - real,
            "order_p": (1 + sum(c <= real for c in copies)) / (shuffles + 1)}


def model_from(anchor, state):
    model = SignalLM(**anchor.spec["model"])
    model.load_state_dict(OrderedDict((k, v.float()) for k, v in state.items()))
    return model


def add(base, delta):
    """base + delta, as float32 weights."""
    return OrderedDict((k, (v.double() + delta[k]).float()) for k, v in base.items())


def unflatten(anchor, flat):
    """A flat float64 vector in parameter order, as tensors shaped like the anchor's."""
    out, offset = OrderedDict(), 0
    for name, value in anchor.theta0.items():
        out[name] = torch.from_numpy(np.array(flat[offset:offset + value.numel()], dtype=np.float64).reshape(value.shape))
        offset += value.numel()
    return out


# DARE, and the payload a volunteer would send.

def dare(flat, p, method, seed):
    """DARE: drops a fraction p of a flat delta's entries; returns the kept indices and values.
    'random' drops each entry with probability p (a PCG64 stream from `seed`) and rescales the
    survivors by 1/(1 − p), so the expected delta is unchanged. 'magnitude' drops the smallest p
    by |τ| and keeps the survivors as they are: they're already the large entries, so rescaling
    them would multiply the delta's size instead of preserving it."""
    if method == "random":
        kept = np.flatnonzero(np.random.Generator(np.random.PCG64(seed)).random(flat.size) >= p)
        return kept, flat[kept] / (1 - p)
    kept = np.sort(np.argsort(-np.abs(flat), kind="stable")[:flat.size - round(p * flat.size)])
    return kept, flat[kept]


def varints(values):
    out = bytearray()
    for value in values:
        while value >= 0x80:
            out.append(value & 0x7F | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


def read_varints(data):
    values, value, shift = [], 0, 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 28:
                raise ValueError("An index in the payload is too large")
        else:
            values.append(value)
            value, shift = 0, 0
    if shift:
        raise ValueError("The payload's indices are truncated")
    return np.array(values, dtype=np.int64)


def tensor_of(anchor, indices):
    """Which of the anchor's tensors each flat index falls in."""
    bounds = np.cumsum([0] + [v.numel() for v in anchor.theta0.values()])
    return np.searchsorted(bounds, indices, side="right") - 1, int(bounds[-1])


def encode_delta(anchor, tau, base_sha256, p, method, seed):
    """A delta as a volunteer would send it: DARE-dropped, the survivors quantized to int8 with
    one float32 scale per tensor, and their indices gap-coded as varints. Compact JSON text."""
    kept, values = dare(flatten(tau), p, method, seed)
    tensor, size = tensor_of(anchor, kept)
    peak = np.zeros(len(anchor.theta0))
    np.maximum.at(peak, tensor, np.abs(values))
    scales = (peak / 127).astype(np.float32)
    scale = scales.astype(np.float64)[tensor]
    q = np.clip(np.rint(np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)), -127, 127).astype(np.int8)
    payload = {
        "format": "eiathome-signal-lm-delta", "version": 1, "anchor": anchor.spec["id"], "base_sha256": base_sha256,
        "dare": {"method": method, "p": p, "seed": seed}, "parameters": size, "kept": len(kept),
        "scales": [float(s) for s in scales],
        "indices": base64.b64encode(varints(np.diff(kept, prepend=0).tolist())).decode(),
        "values": base64.b64encode(q.tobytes()).decode(),
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_delta(anchor, text, base_sha256):
    """The delta a payload rebuilds, refusing one for another anchor or start, or a malformed one."""
    payload = json.loads(text)
    if payload.get("format") != "eiathome-signal-lm-delta" or payload.get("version") != 1:
        raise ValueError("Not a signal-LM delta payload")
    if payload.get("anchor") != anchor.spec["id"] or payload.get("base_sha256") != base_sha256:
        raise ValueError("The payload is for another anchor or start")
    gaps = read_varints(base64.b64decode(payload["indices"], validate=True))
    q = np.frombuffer(base64.b64decode(payload["values"], validate=True), dtype=np.int8)
    kept = np.cumsum(gaps)
    tensor, size = tensor_of(anchor, kept)
    scales = np.array(payload["scales"], dtype=np.float64)
    if (payload.get("parameters") != size or len(scales) != len(anchor.theta0) or len(q) != len(kept) or len(kept) != payload.get("kept")
            or (len(kept) and (kept[-1] >= size or (gaps[1:] < 1).any())) or not np.isfinite(scales).all() or (scales < 0).any()
            or not np.array_equal(scales.astype(np.float32).astype(np.float64), scales)):
        raise ValueError("The payload's indices, values or scales are malformed")
    flat = np.zeros(size)
    flat[kept] = q * scales[tensor]
    return unflatten(anchor, flat)


# Combining deltas.

def shard_weights(entries):
    """Each contributor's weight: its shard's training targets over all the contributing shards',
    shared equally by the workers on that shard, so each slice counts once."""
    on_shard = {}
    for e in entries:
        on_shard.setdefault(tuple(e["shard"]), []).append(e)
    total = sum(group[0]["training_targets"] for group in on_shard.values())
    return [e["training_targets"] / len(on_shard[tuple(e["shard"])]) / total for e in entries]


def trim(flat, keep):
    """The largest `keep` fraction of a flat delta's entries by magnitude; the rest zero."""
    if keep >= 1:
        return flat
    kept = np.argsort(-np.abs(flat), kind="stable")[:round(keep * flat.size)]
    out = np.zeros_like(flat)
    out[kept] = flat[kept]
    return out


def ties(flats, weights, keep):
    """TIES-merging of flat deltas: trim each to its largest `keep` fraction, elect each
    parameter's sign as the sign of the weighted sum, and take, for each parameter, the weighted
    mean of only the nonzero entries that agree with it. A delta that's zero at a parameter, or
    disagrees there, doesn't pull the others toward zero."""
    trimmed = np.stack([trim(f, keep) for f in flats])
    w = np.asarray(weights, dtype=np.float64)[:, None]
    elected = np.sign((w * trimmed).sum(axis=0))
    agree = (np.sign(trimmed) == elected) & (trimmed != 0)
    mass = (w * agree).sum(axis=0)
    return np.divide((w * trimmed * agree).sum(axis=0), mass, out=np.zeros(trimmed.shape[1]), where=mass > 0)


def combine(flats, weights, rule):
    """The merged flat delta τ̄ of some contributors' flat deltas, by the rule's merge."""
    if rule["merge"] == "ties":
        return ties(flats, weights, rule["ties_keep"])
    return sum(w * f for w, f in zip(weights, flats))


def greedy_soup(ranked, merged_bits):
    """A greedy soup: start from the best candidate alone, then try the others best first and
    keep each only if the merge with it scores lower on validation than without. merged_bits
    scores the merge of a list of candidates. Returns the soup, its score, and each trial as
    (candidate, bits before it, bits with it, kept)."""
    soup, best = [ranked[0]], merged_bits([ranked[0]])
    trials = [(ranked[0], None, best, True)]
    for candidate in ranked[1:]:
        bits = merged_bits(soup + [candidate])
        trials.append((candidate, best, bits, bits < best))
        if bits < best:
            soup, best = soup + [candidate], bits
    return soup, best, trials


def outer_step(state, tau_bar, momentum, lr, mu):
    """DiLoCo's outer step, with the merged delta τ̄ as the negative pseudo-gradient and Nesterov
    momentum: m ← μ·m + τ̄, then θ ← θ + lr·(τ̄ + μ·m). With lr 1 and μ 0, it's θ + τ̄ exactly.
    Returns the new weights (float32) and momentum (float32; None when μ is 0)."""
    m = OrderedDict((k, mu * (momentum[k].double() if momentum is not None else 0.0) + tau_bar[k]) for k in tau_bar)
    weights = OrderedDict((k, (v.double() + lr * (tau_bar[k] + mu * m[k])).float()) for k, v in state.items())
    return weights, (OrderedDict((k, v.float()) for k, v in m.items()) if mu else None)


class Merged(NamedTuple):
    state: OrderedDict
    momentum: OrderedDict | None


# A round.

def merge_round(anchor, workers, start=None, rule=None, manifest=None):
    """One round: check each worker (compressing its delta if the rule says so), gate it against
    the incumbent, merge the accepted deltas by the rule, take the outer step, and gate the
    result. workers: checkpoint paths. Returns (record, Merged or None)."""
    rule = check_rule(rule)
    start_ = round_start(anchor, start)
    if start_.outer is not None and start_.outer != rule["outer"]:
        raise ValueError(f"{start} was merged with outer step {start_.outer}: a round chain keeps one outer step")
    loaded = [(str(path), load_checkpoint(path, anchor)) for path in workers]
    if not loaded:
        raise ValueError("A round needs at least one worker")
    chain = lineage(loaded[0][1])
    if any(lineage(c) != chain for _, c in loaded[1:]):
        raise ValueError("A round merges workers of one family, on one dataset and split")
    check_continuation(start, start_, chain)
    records, partitions = held_out(anchor, chain["family"], chain["evaluation"], manifest, chain["dataset_sha256"])
    if any(chain[name] != [r["entry"] for r in partitions[name]] for name in ("validation", "test")):
        raise ValueError("The workers' validation or test split isn't this manifest's")
    # One division into shards per round, so the accepted shards are disjoint; each worker's
    # training slices and target count are checked against its shard, not taken from its record.
    shards = [recorded_shard(path, c["merge"]) for path, c in loaded]
    counts = sorted({n for _, n in shards})
    if len(counts) != 1:
        raise ValueError(f"A round merges workers from one division into shards; these use {' and '.join(map(str, counts))} shards")
    assigned = {i: shard_records(partitions["train"], i, counts[0]) for i in range(1, counts[0] + 1)}

    encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
    validation = encode(partitions["validation"])
    control = controls(records, chain["family"], chain["evaluation"])
    selection = encode(control["selection"]) if control else None
    bits = lambda state, tokens: evaluate(model_from(anchor, state), tokens)["bits_per_token"]
    # The selection control is recorded for the incumbent, each worker and the merge, not gated on.
    diagnose = lambda state: order_gain(model_from(anchor, state), selection) if control else None
    incumbent = {"validation_bits": bits(start_.state, validation), "control": diagnose(start_.state)}

    def gate(state):
        """A candidate's scores, and why it fails the gate (or None)."""
        scores = {"validation_bits": bits(state, validation), "control": diagnose(state)}
        if scores["validation_bits"] > incumbent["validation_bits"] + rule["tolerance_bits"]:
            return scores, (f"it scores {scores['validation_bits']:.4f} bits on validation, worse than the incumbent's "
                            f"{incumbent['validation_bits']:.4f} by more than {rule['tolerance_bits']:g}")
        return scores, None

    contributors, accepted, seen = [], [], set()
    for (path, c), shard in zip(loaded, shards):
        merge = c["merge"]
        entry = {"file": path, "file_sha256": file_digest(path), "shard": list(shard), "seed": merge.get("seed"),
                 "training_targets": sum(len(r["frames"]) - 1 for r in assigned[shard[0]]), "validation_bits": None,
                 "control": None, "weight": 0.0}
        key = (shard, merge.get("seed"))
        tau = difference(c["model"], start_.state)
        problem = worker_problem(c, start_, rule["max_abs_delta"], tau, shard, assigned[shard[0]])
        if problem is None and key in seen:
            problem = "duplicate: the same shard and seed as an earlier contributor, so counted once"
        if problem is None and rule["dare"]:
            # The payload's random drop is seeded by the worker's weights, so a replay redraws it.
            seed = int(weights_digest(c["model"])[:13], 16)
            text = encode_delta(anchor, tau, start_.weights_sha256, rule["dare"]["p"], rule["dare"]["method"], seed).encode()
            entry["payload"] = {"bytes": len(text), "sha256": hashlib.sha256(text).hexdigest(), "seed": seed}
            if len(text) > CARD_BUDGET_BYTES:
                problem = f"its compressed delta takes {len(text):,} bytes, over the {CARD_BUDGET_BYTES:,}-byte budget"
            else:
                # What reaches the model is the rebuilt delta, rescaled by 1/(1 − p): bound that too.
                tau = decode_delta(anchor, text, start_.weights_sha256)
                problem = delta_problem(tau, rule["max_abs_delta"], "its rebuilt delta")
        if problem is None:
            seen.add(key)
            scores, problem = gate(add(start_.state, tau))
            entry.update(scores)
        entry.update(status="rejected" if problem else "accepted", reason=problem)
        contributors.append(entry)
        if not problem:
            accepted.append((entry, flatten(tau)))

    record = {
        "format": "eiathome-signal-lm-merge-round", "version": 1, "anchor": anchor.identity(),
        "round": start_.round + 1, "family": chain["family"], "evaluation": chain["evaluation"],
        "dataset_sha256": chain["dataset_sha256"], "shards": counts[0],
        "control": None if control is None else {"family": control["family"], "selection": [r["entry"] for r in control["selection"]],
                                                 "shuffles": ORDER_SHUFFLES, "role": "diagnostic"},
        "start": {"checkpoint": "theta0" if start is None else str(start), "round": start_.round,
                  "weights_sha256": start_.weights_sha256, **incumbent},
        "rule": rule,
        "contributors": contributors,
    }
    if not accepted:
        record["merged"] = {"decision": "rejected", "reason": "no worker passed the checks and the gate"}
        return record, None

    def merged_delta(members):
        return unflatten(anchor, combine([f for _, f in members], shard_weights([e for e, _ in members]), rule))

    best = min((e for e, _ in accepted), key=lambda e: e["validation_bits"])
    record["best_worker"] = {"file": best["file"], "validation_bits": best["validation_bits"]}
    if rule["soup"] == "greedy":
        # Best first by validation (ties broken by file hash), each kept only if it helps.
        ranked = sorted(accepted, key=lambda a: (a[0]["validation_bits"], a[0]["file_sha256"]))
        soup, soup_bits, trials = greedy_soup(ranked, lambda members: bits(add(start_.state, merged_delta(members)), validation))
        for rank, ((e, _), before, with_it, kept) in enumerate(trials, 1):
            e["soup"] = {"rank": rank, "validation_bits_with_it": with_it, "kept": kept}
            if not kept:
                e.update(status="set aside", reason=f"the soup scores {with_it:.4f} bits on validation with it, "
                                                    f"no better than {before:.4f} without it")
    else:
        soup = accepted
        soup_bits = bits(add(start_.state, merged_delta(soup)), validation)
    for (e, _), w in zip(soup, shard_weights([e for e, _ in soup])):
        e["weight"] = w
    state, momentum = outer_step(start_.state, merged_delta(soup), start_.momentum, rule["outer"]["lr"], rule["outer"]["momentum"])
    scores, problem = gate(state)
    record["merged"] = {
        "decision": "rejected" if problem else "accepted",
        "reason": None if problem is None else problem.replace("it scores", "the merged model scores", 1),
        "contributors": len(soup), "soup_validation_bits": soup_bits, **scores,
        "weights_sha256": weights_digest(state), "momentum_sha256": None if momentum is None else weights_digest(momentum),
    }
    return record, None if problem else Merged(state, momentum)


def merged_checkpoint(anchor, record, merged, chain):
    """The merged model as a checkpoint: loadable by sample.py, and a start for the next round,
    with its chain's lineage, outer rule and momentum."""
    return {
        "format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": anchor.spec["model"],
        "model": OrderedDict((k, v.clone()) for k, v in merged.state.items()), "tokenizer": anchor.tokenizer.state(),
        "dataset_sha256": record["dataset_sha256"], "evaluation": record["evaluation"], "anchor": anchor.identity(),
        "splits": {"train": None, "validation": chain["validation"], "test": chain["test"]},
        "outer_momentum": merged.momentum,
        "merge": {"merged": True, "round": record["round"], "start_sha256": record["start"]["weights_sha256"],
                  "weights_sha256": record["merged"]["weights_sha256"], "outer": record["rule"]["outer"],
                  "momentum_sha256": record["merged"]["momentum_sha256"]},
    }


def run_round(output, workers, start=None, rule=None, anchor_name="v1"):
    """Merges a round into a new folder: round.json, and merged.pt if the merge passed."""
    require_new_output(output)
    anchor = load_anchor(anchor_name)
    with training_runtime(1, "cpu"), atomic_output(output) as folder:
        record, merged = merge_round(anchor, workers, start, rule)
        (folder / "round.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if merged is not None:
            chain = lineage(load_checkpoint(workers[0], anchor))
            torch.save(merged_checkpoint(anchor, record, merged, chain), folder / "merged.pt")
    return record


def summary(record):
    lines = [f"round {record['round']}: incumbent {record['start']['validation_bits']:.4f} bits on validation"]
    for c in record["contributors"]:
        bits = "" if c["validation_bits"] is None else f" {c['validation_bits']:.4f} bits,"
        size = f" {c['payload']['bytes']:,} bytes," if "payload" in c else ""
        lines.append(f"  {c['file']}: shard {c['shard']}, seed {c['seed']},{size}{bits} {c['status']}"
                     + (f" ({c['reason']})" if c["reason"] else f", weight {c['weight']:.3f}"))
    m = record["merged"]
    lines.append(f"  merged: {m['decision']}" + (f", {m['validation_bits']:.4f} bits on validation" if "validation_bits" in m else "")
                 + (f" ({m['reason']})" if m.get("reason") else ""))
    return "\n".join(lines)


def replay(path, manifest=None):
    """Recomputes a round from its record and the same files, and checks it comes out the same."""
    record = json.loads(Path(path).read_text())
    if record.get("format") != "eiathome-signal-lm-merge-round":
        raise ValueError(f"{path}: not a merge round record")
    anchor = anchor_named(record.get("anchor"))
    for c in record["contributors"]:
        if file_digest(c["file"]) != c["file_sha256"]:
            raise ValueError(f"{c['file']} isn't the file the round used")
    start = None if record["start"]["checkpoint"] == "theta0" else record["start"]["checkpoint"]
    with training_runtime(1, "cpu"):
        again, _ = merge_round(anchor, [c["file"] for c in record["contributors"]], start, record["rule"], manifest)
    if again != record:
        raise ValueError(f"{path}: replaying the round gave a different result")
    return record


def score(path, manifest=None):
    """Test bits for a checkpoint from an anchor, and the reporting control (the other family's
    test scan): for reporting, never for choosing a worker or a merge."""
    anchor = anchor_named(torch.load(path, map_location="cpu", weights_only=True).get("anchor"))
    checkpoint = load_checkpoint(path, anchor)
    family = lineage(checkpoint)["family"]
    records, partitions = held_out(anchor, family, checkpoint["evaluation"], manifest, checkpoint["dataset_sha256"])
    model = model_from(anchor, checkpoint["model"])
    encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
    out = {name: evaluate(model, encode(partitions[name]))["bits_per_token"] for name in ("validation", "test")}
    control = controls(records, family, checkpoint["evaluation"])
    if control:
        out["control"] = {"family": control["family"], "bits_per_token": evaluate(model, encode(control["reporting"]))["bits_per_token"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("round", help="merge workers' checkpoints into the next round")
    r.add_argument("workers", nargs="+", type=Path, help="workers' real.pt checkpoints")
    r.add_argument("--output", type=Path, required=True, help="a new folder for round.json and merged.pt")
    r.add_argument("--start", type=Path, default=None, help="the previous round's merged.pt (default: θ₀, round 1)")
    r.add_argument("--anchor", default="v1", choices=list(ANCHORS))
    r.add_argument("--merge", choices=("average", "ties"), default="average")
    r.add_argument("--ties-keep", type=float, default=None, help="TIES: the fraction of each delta kept by magnitude (default 1)")
    r.add_argument("--dare-p", type=float, default=None, help="compress each delta, dropping this fraction of its entries")
    r.add_argument("--dare-method", choices=("random", "magnitude"), default="random")
    r.add_argument("--soup", choices=("all", "greedy"), default="all")
    r.add_argument("--outer-lr", type=float, default=1.0)
    r.add_argument("--outer-momentum", type=float, default=0.0)
    r.add_argument("--tolerance", type=float, default=TOLERANCE_BITS, help="bits/token a worker or merge may lose on validation")
    r.add_argument("--max-abs-delta", type=float, default=MAX_ABS_DELTA)
    sub.add_parser("replay", help="recompute a round from its record").add_argument("record", type=Path)
    sub.add_parser("score", help="test and control bits, for reporting only").add_argument("checkpoint", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "round":
            rule = {"merge": args.merge, "ties_keep": 1.0 if args.ties_keep is None else args.ties_keep,
                    "dare": None if args.dare_p is None else {"method": args.dare_method, "p": args.dare_p},
                    "soup": args.soup, "outer": {"lr": args.outer_lr, "momentum": args.outer_momentum},
                    "tolerance_bits": args.tolerance, "max_abs_delta": args.max_abs_delta}
            print(summary(run_round(args.output, args.workers, args.start, rule, args.anchor)))
        elif args.command == "replay":
            record = replay(args.record)
            print(f"round {record['round']} replays exactly: merged weights {record['merged'].get('weights_sha256', '(none)')}")
        else:
            with training_runtime(1, "cpu"):
                print(json.dumps(score(args.checkpoint), indent=2))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
