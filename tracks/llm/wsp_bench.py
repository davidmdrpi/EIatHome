"""Weight-space programming on volunteers' workers (signal-lm-wsp-bench/v0): can an editor that
reads the workers' weights merge them better than a search that doesn't?

The session benchmark (session_bench.py, BENCHMARK.md) publishes, for each seed, a base model and
four workers trained from it on disjoint observing sessions. There, no one-shot merge beat the best
worker: averaging diluted the one worker whose sessions helped. This benchmark gives several merge
methods the same cohorts and the same number of looks at validation, and compares what each picks.

**Edit programs.** Every method proposes programs in one small, declarative language, which an
independent validator checks against fixed bounds and applies (`apply`):

- `start`: "average" (the workers' deltas averaged by training targets, merge.py's plain merge) or
  "base" (no delta);
- `combine` a group: its tensors become base + Σ wₖ τₖ (wₖ in [-1, 2], Σ|wₖ| ≤ 4), where τₖ is
  worker k's delta from the base;
- `trim` a group: keep the largest `keep` fraction of each tensor's current delta from the base
  (0.05 to 1), as TIES trims;
- `low_rank`: add `scale` (-2 to 2) times the best rank-`rank` (1 to 4) approximation of one
  worker's delta to one weight matrix;
- `reset` a group to the base.

At most 32 operations; the result must be finite and within 10 of the base everywhere. The groups
are the model's parts: the token embedding, each block's attention and feed-forward halves, the
final norm and the output head. Programs are applied in float64 and stored as float32; each pick is
recorded as an edit artifact (its program, the digests of its inputs and of its output weights),
which `verify` replays.

**The budget.** Each method may score up to 32 programs on validation (mean bits per token over
the validation sessions, and each session's). Its pick is the best on validation among everything
it scored and the reference points every method gets free: the base, each worker alone and the
plain average. Nothing is chosen on test.

**The methods.**

- `random`: 32 random programs, each group a Dirichlet(½) mix of the workers scaled by U[0.5, 1.5].
- `coordinate`: from the plain average, each group in turn (the head first) tries each worker
  alone, the base and the average extrapolated by 1.5, keeping any that improves validation.
- `llm`: Claude (EDITOR below) reads the model's layout, the training
  provenance (each contributor's sessions: dates, targets, bands), per-tensor statistics of every
  delta (norms, cosines, sign agreement, singular values) and the reference scores, then proposes
  programs through a tool and sees each one's validation scores.
- `llm-anonymized`: the same, with every name, date and band hidden: contributors and sessions are
  numbered, nothing else.
- `specialist bank`: no merge. The four workers are kept whole and mixed per sequence by their
  posterior given the tokens so far (route_bench.py's posterior mixture), at four times the
  inference cost. A reference point, not a candidate.

Reported, per seed: each pick's validation and test bits, its worst test session, the control and
the order gain (session_bench.py's measures), and its costs: queries, programs refused, seconds,
and for the editor, tokens and dollars. The session benchmark's arms on the same cohort (best
worker, the merges, the chains, one model on all the data) and its train-fitted baselines are
reported beside them.

**The protocol is pinned** (PROTOCOL_SHA256): the edit language, the budget and the selection
rule, the controls and the editor (its model, settings, prompt, tools and what it reads). v0's
test scores have been seen, so it is for development. A fresh cohort (session_bench.py's v1) is
cut only after the protocol is final, and scored once; a run on it refuses any other protocol.

    python -m tracks.llm.wsp_bench context --seed 42 [--anonymized]          # what the editor reads first
    python -m tracks.llm.wsp_bench run --output runs/wsp --methods random coordinate specialist
    python -m tracks.llm.wsp_bench run --output runs/wsp-llm --methods llm llm-anonymized   # ANTHROPIC_API_KEY
    python -m tracks.llm.wsp_bench verify                                    # replay the published picks
"""

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import NamedTuple

import numpy as np
import torch

from . import session_bench as sessions
from .anchor import load_anchor, weights_digest
from .data import digest, session_groups
from .merge import ORDER_SHUFFLES, model_from, order_gain, shard_records
from .model import evaluate
from .route_bench import mixture, target_bits
from .runtime import require_new_output, training_runtime

BENCHMARK = "signal-lm-wsp-bench/v0"
GROUPS = ("embed", "block0.attention", "block0.feed-forward", "block1.attention", "block1.feed-forward", "final-norm", "head")
LIMITS = {"weight": (-1.0, 2.0), "weight_l1": 4.0, "keep": (0.05, 1.0), "rank": (1, 4), "scale": (-2.0, 2.0), "ops": 32,
          "max_abs_delta": 10.0}
BUDGET = 32
METHODS = ("random", "coordinate", "llm", "llm-anonymized", "specialist")
SEARCH_SEED = 20261107


def group_of(name):
    """The edit group a parameter belongs to."""
    if name == "token.weight":
        return "embed"
    if name.startswith("blocks."):
        _, index, part = name.split(".")[:3]
        return f"block{index}.attention" if part in ("norm1", "qkv", "out") else f"block{index}.feed-forward"
    if name.startswith("norm."):
        return "final-norm"
    if name == "head.weight":
        return "head"
    raise ValueError(f"{name} belongs to no edit group")


# ---------- the cohort ----------

class Cohort(NamedTuple):
    seed: int
    base: OrderedDict  # parameter name -> float32 tensor
    workers: dict  # "w2".."w5" -> state
    weights: dict  # the plain average's weight for each worker (by training targets)
    digests: dict  # "base" and each worker -> weights SHA-256
    headers: dict  # each one's published header


def load_cohort(bench, seed):
    """A published base and its four workers (session_bench.py), checked against their headers."""
    bench = sessions.benchmark(bench)
    anchor = load_anchor(sessions.ANCHOR)
    folder = bench.folder / "checkpoints" / f"seed{seed}"
    loaded = {name: sessions.load_published(anchor, folder / f"{name}.bin", bench=bench) for name in sessions.PUBLISHED}
    headers = {name: json.loads((folder / f"{name}.json").read_text()) for name in sessions.PUBLISHED}
    workers = OrderedDict((f"w{k}", loaded[f"worker{k}"]["model"]) for k in sessions.WORKER_SHARDS)
    targets = {f"w{k}": headers[f"worker{k}"]["merge"]["training_targets"] for k in sessions.WORKER_SHARDS}
    total = sum(targets.values())
    return Cohort(seed, loaded["base"]["model"], workers, {w: t / total for w, t in targets.items()},
                  {"base": weights_digest(loaded["base"]["model"]), **{w: weights_digest(s) for w, s in workers.items()}},
                  {"base": headers["base"], **{f"w{k}": headers[f"worker{k}"] for k in sessions.WORKER_SHARDS}})


# ---------- edit programs ----------

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_program(program, cohort):
    """Why a program can't be applied (a ValueError), or nothing: its shape, names and bounds."""
    if not isinstance(program, dict) or set(program) - {"start", "ops"} or program.get("start") not in ("average", "base"):
        raise ValueError('A program is {"start": "average" or "base", "ops": [...]}')
    ops = program.get("ops", [])
    if not isinstance(ops, list) or len(ops) > LIMITS["ops"]:
        raise ValueError(f"ops is a list of at most {LIMITS['ops']} operations")
    shapes = {name: tuple(v.shape) for name, v in cohort.base.items()}
    for i, op in enumerate(ops):
        where = f"operation {i + 1}"
        if not isinstance(op, dict) or op.get("op") not in ("combine", "trim", "low_rank", "reset"):
            raise ValueError(f"{where}: op is combine, trim, low_rank or reset")
        allowed = {"combine": {"op", "group", "weights"}, "trim": {"op", "group", "keep"}, "reset": {"op", "group"},
                   "low_rank": {"op", "tensor", "source", "rank", "scale"}}[op["op"]]
        if set(op) != allowed:
            raise ValueError(f"{where}: {op['op']} takes exactly {', '.join(sorted(allowed - {'op'}))}")
        if "group" in op and op["group"] not in GROUPS:
            raise ValueError(f"{where}: group is one of {', '.join(GROUPS)}")
        if op["op"] == "combine":
            weights = op["weights"]
            if not isinstance(weights, dict) or set(weights) - set(cohort.workers):
                raise ValueError(f"{where}: weights maps workers ({', '.join(cohort.workers)}) to numbers")
            low, high = LIMITS["weight"]
            if not all(finite(w) and low <= w <= high for w in weights.values()):
                raise ValueError(f"{where}: each weight is in [{low}, {high}]")
            if sum(abs(w) for w in weights.values()) > LIMITS["weight_l1"]:
                raise ValueError(f"{where}: the weights' absolute values sum to at most {LIMITS['weight_l1']}")
        elif op["op"] == "trim":
            low, high = LIMITS["keep"]
            if not (finite(op["keep"]) and low <= op["keep"] <= high):
                raise ValueError(f"{where}: keep is in [{low}, {high}]")
        elif op["op"] == "low_rank":
            if len(shapes.get(op["tensor"], ())) != 2:
                raise ValueError(f"{where}: tensor names a weight matrix: {', '.join(n for n, s in shapes.items() if len(s) == 2)}")
            if op["source"] not in cohort.workers:
                raise ValueError(f"{where}: source is one of {', '.join(cohort.workers)}")
            low, high = LIMITS["rank"]
            if type(op["rank"]) is not int or not low <= op["rank"] <= high:
                raise ValueError(f"{where}: rank is an integer from {low} to {high}")
            low, high = LIMITS["scale"]
            if not (finite(op["scale"]) and low <= op["scale"] <= high):
                raise ValueError(f"{where}: scale is in [{low}, {high}]")


def apply(cohort, program):
    """The weights a program makes, as float32: checked, applied in float64, then bounded."""
    check_program(program, cohort)
    base = OrderedDict((k, v.double()) for k, v in cohort.base.items())
    tau = {w: {k: s[k].double() - base[k] for k in base} for w, s in cohort.workers.items()}
    names = {g: [k for k in base if group_of(k) == g] for g in GROUPS}
    start = cohort.weights if program["start"] == "average" else {}

    def mix(name, weights):
        out = base[name].clone()
        for w in sorted(weights):
            out += weights[w] * tau[w][name]
        return out

    theta = OrderedDict((k, mix(k, start)) for k in base)
    for op in program.get("ops", []):
        if op["op"] == "combine":
            for k in names[op["group"]]:
                theta[k] = mix(k, op["weights"])
        elif op["op"] == "reset":
            for k in names[op["group"]]:
                theta[k] = base[k].clone()
        elif op["op"] == "trim":
            for k in names[op["group"]]:
                delta = (theta[k] - base[k]).flatten()
                keep = math.ceil(op["keep"] * delta.numel())
                order = torch.sort(delta.abs(), descending=True, stable=True).indices
                mask = torch.zeros_like(delta)
                mask[order[:keep]] = 1
                theta[k] = base[k] + (delta * mask).reshape(base[k].shape)
        else:
            u, s, vh = torch.linalg.svd(tau[op["source"]][op["tensor"]], full_matrices=False)
            r = op["rank"]
            theta[op["tensor"]] = theta[op["tensor"]] + op["scale"] * ((u[:, :r] * s[:r]) @ vh[:r])
    out = OrderedDict((k, v.float()) for k, v in theta.items())
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise ValueError("The program makes non-finite weights")
    largest = max((out[k] - cohort.base[k]).abs().max().item() for k in out)
    if largest > LIMITS["max_abs_delta"]:
        raise ValueError(f"The program moves a weight {largest:.3g} from the base, beyond the bound of {LIMITS['max_abs_delta']}")
    return out


def references(cohort):
    """The programs every method gets free: the base, the plain average and each worker alone."""
    alone = lambda w: {"start": "base", "ops": [{"op": "combine", "group": g, "weights": {w: 1.0}} for g in GROUPS]}
    return {"base": {"start": "base", "ops": []}, "average": {"start": "average", "ops": []},
            **{f"{w} alone": alone(w) for w in cohort.workers}}


# ---------- scoring ----------

class BudgetSpent(Exception):
    pass


def score_state(ctx, state, test=False):
    """Validation bits (the mean over its targets, and each session's) and, with test, the
    session benchmark's reporting measures."""
    model = model_from(ctx.anchor, state)
    with training_runtime(1, "cpu"):
        sessions_bits = [evaluate(model, s)["bits_per_token"] for s in ctx.validation_sessions]
        out = {"validation": evaluate(model, ctx.validation)["bits_per_token"], "validation_sessions": sessions_bits,
               "validation_worst_session": max(sessions_bits)}
        if test:
            tests = [evaluate(model, s)["bits_per_token"] for s in ctx.test_sessions]
            order = order_gain(model, ctx.test)
            out.update(test=evaluate(model, ctx.test)["bits_per_token"], test_sessions=tests, test_worst_session=max(tests),
                       control=evaluate(model, ctx.control)["bits_per_token"], test_order_gain_bits=order["order_gain_bits"])
    return out


class Oracle:
    """The validation interface every method gets: `budget` programs scored, refused programs not
    counted (up to `budget` of them), and the reference points free."""

    def __init__(self, ctx, cohort, budget=BUDGET, refs=None):
        self.ctx, self.cohort, self.budget = ctx, cohort, budget
        self.queries, self.refused, self.seconds = [], [], 0.0
        self.references = refs if refs is not None else {
            name: {"program": p, **score_state(ctx, apply(cohort, p))} for name, p in references(cohort).items()}

    @property
    def left(self):
        return self.budget - len(self.queries)

    def evaluate(self, program):
        if self.left <= 0:
            raise BudgetSpent(f"All {self.budget} validation queries are spent")
        started = time.perf_counter()
        try:
            state = apply(self.cohort, program)
        except (ValueError, KeyError, TypeError) as err:
            self.refused.append({"program": program, "reason": str(err)})
            if len(self.refused) > self.budget:
                raise BudgetSpent(f"More than {self.budget} programs refused") from err
            raise ValueError(str(err)) from err
        scores = score_state(self.ctx, state)
        self.seconds += time.perf_counter() - started
        entry = {"query": len(self.queries) + 1, "program": program, **scores, "weights_sha256": weights_digest(state)}
        self.queries.append(entry)
        return entry

    def pick(self):
        """The best on validation among the references and the queries (the earliest on a tie)."""
        candidates = [(r["validation"], 0, name, r["program"]) for name, r in self.references.items()]
        candidates += [(q["validation"], q["query"], f"query {q['query']}", q["program"]) for q in self.queries]
        validation, _, name, program = min(candidates, key=lambda c: (c[0], c[1]))
        return name, program


# ---------- the search controls ----------

def random_search(oracle, seed=SEARCH_SEED):
    rng = np.random.default_rng(seed)
    workers = list(oracle.cohort.workers)
    while oracle.left > 0:
        ops = []
        for group in GROUPS:
            mix = rng.dirichlet([0.5] * len(workers)) * rng.uniform(0.5, 1.5)
            ops.append({"op": "combine", "group": group, "weights": {w: round(float(x), 4) for w, x in zip(workers, mix)}})
        oracle.evaluate({"start": "base", "ops": ops})


def coordinate_search(oracle):
    workers = list(oracle.cohort.workers)
    average = {w: round(x, 6) for w, x in oracle.cohort.weights.items()}
    current = {g: dict(average) for g in GROUPS}
    best = oracle.references["average"]["validation"]
    options = [{w: 1.0} for w in workers] + [{}, {w: round(1.5 * x, 6) for w, x in average.items()}, dict(average)]
    program = lambda choice: {"start": "base", "ops": [{"op": "combine", "group": g, "weights": choice[g]} for g in GROUPS]}
    while oracle.left > 0:
        improved = False
        for group in reversed(GROUPS):  # the head first, the embedding last
            for option in options:
                if oracle.left <= 0:
                    return
                if option == current[group]:
                    continue
                trial = {**current, group: option}
                scores = oracle.evaluate(program(trial))
                if scores["validation"] < best:
                    best, current, improved = scores["validation"], trial, True
        if not improved:
            return


# ---------- the specialist bank ----------

def bank_bits(models, sequences):
    """Bits per token of the workers' posterior mixture (route_bench.py's, with a uniform prior):
    each target's probability mixes the models' by how well each predicted the sequence's tokens
    before it."""
    with torch.no_grad():
        per_model = [target_bits(m, sequences) for m in models]
    total, count = 0.0, 0
    for i in range(len(sequences)):
        bits = np.stack([p[i] for p in per_model])  # models x targets
        before = np.concatenate([np.zeros((len(models), 1)), -np.cumsum(bits, axis=1)[:, :-1]], axis=1)
        total += mixture(bits, before)[0].sum()
        count += bits.shape[1]
    return float(total / count)


def shuffled_gain(score, sequences, shuffles=ORDER_SHUFFLES):
    """merge.py's order gain for any scorer: bits on copies with each slice's targets shuffled,
    less bits in the real order."""
    real = score(sequences)
    rng = np.random.default_rng(0)
    copies = [score([np.concatenate([s[:1], rng.permutation(s[1:])]) for s in sequences]) for _ in range(shuffles)]
    return float(statistics.fmean(copies) - real)


def specialist_bank(ctx, cohort):
    """No merge: the four workers kept whole and mixed per sequence, on the same sets."""
    models = [model_from(ctx.anchor, s) for s in cohort.workers.values()]
    score = lambda sequences: bank_bits(models, sequences)
    with training_runtime(1, "cpu"):
        validation_sessions = [score(s) for s in ctx.validation_sessions]
        test_sessions = [score(s) for s in ctx.test_sessions]
        return {"validation": score(ctx.validation), "validation_sessions": validation_sessions,
                "validation_worst_session": max(validation_sessions), "test": score(ctx.test), "test_sessions": test_sessions,
                "test_worst_session": max(test_sessions), "control": score(ctx.control),
                "test_order_gain_bits": shuffled_gain(score, ctx.test)}


# ---------- what the editor reads ----------

CONTEXT_VERSION = 1
BANDS = (("L", 1000, 1730), ("S", 1730, 2600), ("C", 3950, 8000), ("X", 8000, 11600))  # GHz bands, in MHz


def describe_session(records, anonymized, label):
    """One observing session as the editor sees it: with names, its date, targets and band."""
    out = {"session": label, "hits": len(records), "targets_tokens": sum(len(r["frames"]) - 1 for r in records)}
    if anonymized:
        return out
    sources = [r["entry"].get("source", {}) for r in records]
    mhz = statistics.median(s.get("centre_freq_mhz", 0.0) for s in sources)
    return {**out, "date": min(s.get("sky", {}).get("epoch", "?") for s in sources)[:10],
            "targets": sorted({s.get("sky", {}).get("target") or "?" for s in sources}),
            "band": next((name for name, low, high in BANDS if low <= mhz < high), "other"), "centre_mhz": round(mhz)}


def provenance(ctx, cohort, bench, anonymized):
    """Who trained on what: each contributor's shard of training sessions, its training targets and
    best epoch (from the session benchmark's report)."""
    trained = json.loads((bench.folder / "report.json").read_text())["seeds"][str(cohort.seed)]
    best_epochs = {"base": trained["base"]["training"]["best_epoch"],
                   **{f"w{w['shard']}": w["training"]["best_epoch"] for w in trained["workers"]}}
    out, number = {}, 0
    for name, shard in [("base", sessions.BASE_SHARD)] + [(f"w{k}", k) for k in sessions.WORKER_SHARDS]:
        groups = sorted(session_groups(shard_records(ctx.partitions["train"], shard, sessions.SHARDS, sessions.UNIT)),
                        key=lambda g: min(r["start"] for r in g))
        described = []
        for g in groups:
            number += 1
            described.append(describe_session(g, anonymized, f"t{number}"))
        out[name] = {"starts_from": "the untrained initialization" if name == "base" else "base",
                     "training_targets": sum(s["targets_tokens"] for s in described), "best_epoch": best_epochs[name],
                     "sessions": described}
    return out


def layer_stats(cohort):
    """Per tensor: its group and shape, the base's norm, each worker's delta norm, the cosines between
    workers' deltas, the share of entries where all four deltas share a sign, and each delta's top
    singular values (matrices only)."""
    names = list(cohort.workers)
    rows = []
    for k, base in cohort.base.items():
        taus = [cohort.workers[w][k].double().flatten() - base.double().flatten() for w in names]
        norms = [float(t.norm()) for t in taus]
        cosine = lambda i, j: float(taus[i] @ taus[j]) / (norms[i] * norms[j]) if norms[i] and norms[j] else 0.0
        signs = torch.stack([t.sign() for t in taus])
        row = {"tensor": k, "group": group_of(k), "shape": list(base.shape), "base_norm": round(float(base.double().norm()), 4),
               "delta_norm": {w: round(n, 4) for w, n in zip(names, norms)},
               "delta_cosine": {f"{names[i]}·{names[j]}": round(cosine(i, j), 3)
                                for i in range(len(names)) for j in range(i + 1, len(names))},
               "sign_agreement": round(float(((signs == signs[0]).all(0) & (signs[0] != 0)).double().mean()), 3)}
        if base.dim() == 2:
            row["delta_singular_values"] = {w: [round(float(v), 4) for v in torch.linalg.svdvals(t.reshape(base.shape))[:4]]
                                            for w, t in zip(names, taus)}
        rows.append(row)
    return rows


DOMAIN = {
    False: ("The model predicts tokens from radio telescope data. Each sequence is one 16-frame window (2.79 s per "
            "frame, 64 fine channels) around a narrowband hit found in a Breakthrough Listen observation of a nearby "
            "star at the Green Bank Telescope; nearly all hits are human-made interference. A fixed tokenizer turns "
            "each frame into one of 33 symbols by the position and power of its strongest channel. The hits are the "
            "'stars' family; a 'blind-window' family (windows away from any hit) is a control, scored but not used "
            "for selection. Every validation session was observed after every training session."),
    True: ("The model predicts sequences of 33 discrete tokens. The data come in sessions; each sequence belongs to "
           "one session. Another family of sequences is a control, scored but not used for selection."),
}

SYSTEM = (
    "You are the editor in a weight-space programming benchmark. You read the weights of a small model's "
    "contributors and write edit programs that combine them into one model of the same shape. Your goal is the "
    "lowest mean bits per token on held-out sessions nobody has scored yet. You only ever see validation scores; "
    "the pick is made by a fixed rule: the lowest validation mean among everything you evaluated and the free "
    "reference points. So spend evaluations on programs you expect to generalize, not on overfitting the 7 "
    "validation sessions.\n\n"
    "A program is JSON: {\"start\": \"average\" | \"base\", \"ops\": [operation, ...]}. \"average\" starts from "
    "the base plus the contributors' deltas averaged by their training targets; \"base\" from the base alone. "
    "The operations, applied in order:\n"
    "- {\"op\": \"combine\", \"group\": G, \"weights\": {\"w2\": a, ...}}: every tensor in group G becomes "
    "base + Σ weight × that worker's delta (missing workers get 0). Each weight in [-1, 2]; their absolute "
    "values sum to at most 4.\n"
    "- {\"op\": \"trim\", \"group\": G, \"keep\": f}: keep the largest fraction f (0.05 to 1) of each tensor's "
    "current delta from the base, by magnitude; zero the rest.\n"
    "- {\"op\": \"low_rank\", \"tensor\": T, \"source\": \"w3\", \"rank\": r, \"scale\": s}: add s (-2 to 2) "
    "times the best rank-r (1 to 4) approximation of that worker's delta to matrix T.\n"
    "- {\"op\": \"reset\", \"group\": G}: return group G to the base.\n"
    "At most 32 operations. Groups: embed, block0.attention, block0.feed-forward, block1.attention, "
    "block1.feed-forward, final-norm, head. Every weight must stay finite and within 10 of the base. A program "
    "that breaks a rule is refused with the reason and not counted.\n\n"
    "Call evaluate_program to score a program on validation; it returns the mean and each validation session's "
    "bits. read_tensor shows any tensor's values (free). When you've spent what you think is useful (at most the budget), call finish with a short account of "
    "what you tried and why."
)

TOOLS = [
    {"name": "evaluate_program", "eager_input_streaming": True,
     "description": "Apply an edit program to the contributors' weights and score the result on validation. Returns "
                    "the validation mean bits per token, each validation session's bits, and the queries left. A refused "
                    "program returns the reason and costs nothing.",
     "input_schema": {"type": "object", "properties": {
         "program": {"type": "object", "description": "The edit program: {\"start\": ..., \"ops\": [...]}."},
         "rationale": {"type": "string", "description": "One or two sentences: what this program tests."}},
         "required": ["program"]}},
    {"name": "read_tensor", "eager_input_streaming": True,
     "description": "Read one tensor's values: the base's, a worker's, or a worker's delta from the base, optionally a "
                    "range of rows. Free: it doesn't touch validation.",
     "input_schema": {"type": "object", "properties": {
         "tensor": {"type": "string", "description": "A tensor name from the layout, e.g. head.weight."},
         "source": {"type": "string", "enum": ["base", "w2", "w3", "w4", "w5"]},
         "delta": {"type": "boolean", "description": "With a worker: its delta from the base instead of its weights."},
         "rows": {"type": "array", "items": {"type": "integer"}, "description": "[start, stop) rows of a matrix; all by default."}},
         "required": ["tensor", "source"]}},
    {"name": "finish", "eager_input_streaming": True,
     "description": "End the session. The pick is then made by the fixed rule (lowest validation mean).",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

EDITOR = {"model": "claude-opus-5-5", "max_tokens": 16000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
          "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default", "max_turns": 48, "context_version": CONTEXT_VERSION}
PRICE_PER_MTOK = {"claude-opus-5-5": {"input": 4.0, "cache_write": 5.0, "cache_read": 0.20, "output": 20.0}}


def model_code():
    import inspect

    from . import model
    return "".join(inspect.getsource(f) + "\n\n" for f in (model.rotary, model.RotaryBlock, model.SignalLM)).rstrip() + "\n"


def lines(items, shape=lambda x: x):
    return "\n".join(json.dumps(shape(x), ensure_ascii=False) for x in items)


def editor_context(ctx, cohort, bench, oracle, anonymized):
    """The editor's first message: the model, the contributors' provenance, per-tensor statistics
    of their deltas, and the free reference points' validation scores."""
    count = sum(v.numel() for v in cohort.base.values())
    validation = [describe_session(g, anonymized, f"v{i + 1}") for i, g in enumerate(session_groups(ctx.partitions["validation"]))]
    references = {name: {"validation": round(r["validation"], 4), "validation_sessions": [round(b, 4) for b in r["validation_sessions"]]}
                  for name, r in oracle.references.items()}
    sections = [
        ("The model", f"{DOMAIN[anonymized]}\n\nA 2-layer transformer with {count:,} parameters (context 15, width 32, "
                      f"4 heads, rotary positions), in {len(GROUPS)} edit groups. Tensors and groups:\n"
                      + json.dumps([{"tensor": k, "shape": list(v.shape), "group": group_of(k)} for k, v in cohort.base.items()])),
        ("The model's code (tracks/llm/model.py)", "```python\n" + model_code() + "```"),
        ("The contributors", "The base trained from the untrained initialization on its shard; workers w2 to w5 each "
                             "continued training the base on their own shard, with early stopping on validation. No two "
                             "contributors share a session.\n"
                             + lines(provenance(ctx, cohort, bench, anonymized).items(), lambda kv: {"contributor": kv[0], **kv[1]})),
        ("The validation sessions", "Scores are bits per next-token target (lower is better); the mean is over all "
                                    f"{sum(len(s) - 1 for s in ctx.validation)} validation targets.\n" + lines(validation)),
        ("Each tensor's deltas (worker − base)", lines(layer_stats(cohort))),
        ("Free reference points (validation)", "\"wK alone\" is the base plus worker K's whole delta, i.e. worker K's "
                                              "own weights.\n" + lines(references.items(), lambda kv: {"reference": kv[0], **kv[1]})),
        ("Your budget", f"{oracle.budget} evaluations. The pick is the lowest validation mean among the references above "
                        "and your evaluations."),
    ]
    return "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)


def plain(block):
    """A content block as JSON, for the transcript."""
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    if hasattr(block, "__dict__"):
        return {k: plain(v) for k, v in vars(block).items()}
    if isinstance(block, list):
        return [plain(b) for b in block]
    return block


def usage_of(response):
    u = response.usage
    get = lambda name: getattr(u, name, None) or 0
    return {"input": get("input_tokens"), "cache_write": get("cache_creation_input_tokens"),
            "cache_read": get("cache_read_input_tokens"), "output": get("output_tokens")}


def cost_of(turns):
    """Tokens summed over the turns, and dollars at the served model's list price (None for a turn
    served by a model without one here)."""
    tokens = {k: sum(t["usage"][k] for t in turns) for k in ("input", "cache_write", "cache_read", "output")}
    dollars = 0.0
    for t in turns:
        price = PRICE_PER_MTOK.get(t["model"])
        if price is None:
            return {"tokens": tokens, "dollars": None, "unpriced_model": t["model"]}
        dollars += sum(t["usage"][k] * price[k] for k in price) / 1e6
    return {"tokens": tokens, "dollars": round(dollars, 6)}


def read_tensor(cohort, args):
    """One tensor's values for the editor, rounded to 5 decimals (the largest has 4,096)."""
    if not isinstance(args, dict) or args.get("tensor") not in cohort.base:
        raise ValueError("tensor names one of the layout's tensors")
    name, source = args["tensor"], args.get("source")
    if source != "base" and source not in cohort.workers:
        raise ValueError(f"source is base or one of {', '.join(cohort.workers)}")
    values = cohort.base[name].double() if source == "base" else cohort.workers[source][name].double()
    if args.get("delta") and source != "base":
        values = values - cohort.base[name].double()
    rows = args.get("rows")
    if rows is not None:
        if not (isinstance(rows, list) and len(rows) == 2 and all(type(r) is int for r in rows) and 0 <= rows[0] < rows[1] <= len(values)):
            raise ValueError(f"rows is [start, stop) within 0 to {len(values)}")
        values = values[rows[0]:rows[1]]
    return json.dumps({"tensor": name, "source": source, "delta": bool(args.get("delta")) and source != "base",
                       "rows": rows, "shape": list(values.shape), "values": np.round(values.numpy(), 5).tolist()})


def feedback(entry, oracle):
    best = min([r["validation"] for r in oracle.references.values()] + [q["validation"] for q in oracle.queries])
    return json.dumps({"query": entry["query"], "validation": round(entry["validation"], 4),
                       "validation_sessions": [round(b, 4) for b in entry["validation_sessions"]],
                       "best_validation_so_far": round(best, 4), "queries_left": oracle.left})


def run_editor(oracle, context, send, editor=EDITOR):
    """The editor's session: a manual tool loop over `send` (one Messages API request in, the final
    message out). Every program goes through the oracle; the loop ends on finish, a spent budget,
    a turn without a tool call, a refusal or truncation, or the turn limit."""
    messages = [{"role": "user", "content": context}]
    turns, finished, stopped = [], None, None
    started = time.perf_counter()
    request = {k: editor[k] for k in ("model", "max_tokens", "thinking", "output_config", "betas", "fallbacks")}
    for _ in range(editor["max_turns"]):
        response = send({**request, "system": SYSTEM, "tools": TOOLS, "messages": messages, "cache_control": {"type": "ephemeral"}})
        turns.append({"model": response.model, "stop_reason": response.stop_reason, "usage": usage_of(response)})
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason in ("refusal", "max_tokens"):
            stopped = response.stop_reason
            break
        calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not calls:
            stopped = "no tool call"
            break
        results = []
        for call in calls:
            result = {"type": "tool_result", "tool_use_id": call.id}
            if call.name == "finish":
                finished = str(call.input.get("summary", "")) if isinstance(call.input, dict) else ""
                results.append({**result, "content": "Session ended."})
            elif call.name == "read_tensor":
                try:
                    results.append({**result, "content": read_tensor(oracle.cohort, call.input)})
                except ValueError as err:
                    results.append({**result, "content": str(err), "is_error": True})
            elif call.name != "evaluate_program":
                results.append({**result, "content": f"No tool named {call.name}", "is_error": True})
            else:
                try:
                    program = call.input.get("program") if isinstance(call.input, dict) else None
                    results.append({**result, "content": feedback(oracle.evaluate(program), oracle)})
                except BudgetSpent as err:
                    stopped = str(err)
                    results.append({**result, "content": str(err), "is_error": True})
                except ValueError as err:
                    results.append({**result, "content": f"Refused, not counted: {err}", "is_error": True})
        messages.append({"role": "user", "content": results})
        if finished is not None or stopped or oracle.left <= 0:
            stopped = stopped or ("finished" if finished is not None else "budget spent")
            break
    else:
        stopped = "turn limit"
    return {"stopped": stopped, "summary": finished, "turns": len(turns), "editor_seconds": time.perf_counter() - started,
            "served_by": sorted({t["model"] for t in turns}), "cost": cost_of(turns), "per_turn": turns,
            "transcript": [{"role": m["role"], "content": plain(m["content"])} for m in messages]}


def anthropic_send():
    """Messages API requests with the experimenter's own key (ANTHROPIC_API_KEY), to the public API
    whatever the environment's ANTHROPIC_BASE_URL says, streamed to the final message."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("The llm methods need ANTHROPIC_API_KEY: the experimenter's own key for api.anthropic.com")
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise SystemExit("Unset ANTHROPIC_AUTH_TOKEN: the editor authenticates with ANTHROPIC_API_KEY only")
    import anthropic

    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")

    def send(request):
        with client.beta.messages.stream(**request) as stream:
            return stream.get_final_message()
    return send


# ---------- the frozen protocol ----------

PROTOCOL_PARTS = (group_of, finite, check_program, apply, references, score_state, Oracle, random_search, coordinate_search,
                  bank_bits, specialist_bank, describe_session, provenance, layer_stats, model_code, lines, editor_context,
                  read_tensor, feedback, run_editor)
# The protocol as pinned: the edit language, the budget and selection rule, the controls, and the
# editor (its model, settings, prompt, tools and what it reads). A run on any benchmark but v0
# refuses another, so a fresh cohort is scored with the protocol pinned before it was cut. While
# the editor is developed on v0, the pin moves with it, each move a visible change (WSP_RESULTS.md).
PROTOCOL_SHA256 = "a23c68012486b87c840089c326c4a3bd63ebcbfa6daff7e3f770eea745026690"


def protocol_sha256():
    import inspect

    parts = {"limits": LIMITS, "budget": BUDGET, "groups": GROUPS, "search_seed": SEARCH_SEED, "editor": EDITOR,
             "system": SYSTEM, "domain": {str(k): v for k, v in DOMAIN.items()}, "tools": TOOLS, "model_code": model_code(),
             "code": {f.__name__: inspect.getsource(f) for f in PROTOCOL_PARTS}}
    return digest(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode())


# ---------- runs ----------

REPORTED = ("validation", "validation_worst_session", "test", "test_worst_session", "control", "test_order_gain_bits")


def edit_artifact(bench, split_sha256, seed, method, cohort, oracle, name, program, state):
    """What a pick is, replayably: its program, the digests of the weights it reads and makes, and
    every query the method spent."""
    return {"format": "eiathome-wsp-edit", "version": 1, "benchmark": BENCHMARK, "sessions_benchmark": bench.id,
            "split_sha256": split_sha256, "protocol_sha256": protocol_sha256(), "seed": seed, "method": method,
            "inputs": cohort.digests, "pick": name, "program": program, "output_sha256": weights_digest(state),
            "queries": [{k: q[k] for k in ("query", "program", "validation", "validation_sessions", "weights_sha256")}
                        for q in oracle.queries],
            "refused": oracle.refused}


def latency(ctx, cohort, repeats=3):
    """Seconds per test target: one model, and the bank of four."""
    model = model_from(ctx.anchor, cohort.base)
    bank = [model_from(ctx.anchor, s) for s in cohort.workers.values()]
    targets = sum(len(s) - 1 for s in ctx.test)

    def timed(score):
        fastest = math.inf
        for _ in range(repeats):
            started = time.perf_counter()
            score()
            fastest = min(fastest, time.perf_counter() - started)
        return fastest / targets

    with training_runtime(1, "cpu"), torch.no_grad():
        return {"one_model": timed(lambda: target_bits(model, ctx.test)), "bank_of_4": timed(lambda: bank_bits(bank, ctx.test))}


def run_seed(ctx, bench, split_sha256, seed, methods, send, output, log=print):
    cohort = load_cohort(bench, seed)
    refs = Oracle(ctx, cohort).references
    best = min((n for n in refs if n.endswith(" alone")), key=lambda n: refs[n]["validation"])
    out = {"inputs": cohort.digests, "best_worker": best.removesuffix(" alone"), "references": {}, "methods": {},
           "inference_seconds_per_target": latency(ctx, cohort)}
    for name, r in refs.items():  # the free reference points, scored on test for reporting
        out["references"][name] = {**score_state(ctx, apply(cohort, r["program"]), test=True), "program": r["program"]}
    for method in methods:
        started = time.perf_counter()
        if method == "specialist":
            result = specialist_bank(ctx, cohort)
        else:
            oracle = Oracle(ctx, cohort, refs=refs)
            extra = {}
            if method == "random":
                random_search(oracle)
            elif method == "coordinate":
                coordinate_search(oracle)
            else:
                anonymized = method == "llm-anonymized"
                session = run_editor(oracle, editor_context(ctx, cohort, bench, oracle, anonymized), send)
                (output / "transcripts" / f"seed{seed}").mkdir(parents=True, exist_ok=True)
                (output / "transcripts" / f"seed{seed}" / f"{method}.json").write_text(
                    json.dumps({"protocol_sha256": protocol_sha256(), "seed": seed, "method": method, **session}, indent=1) + "\n")
                extra = {k: session[k] for k in ("stopped", "summary", "turns", "editor_seconds", "served_by", "cost")}
            pick, program = oracle.pick()
            state = apply(cohort, program)
            artifact = edit_artifact(bench, split_sha256, seed, method, cohort, oracle, pick, program, state)
            (output / "edits" / f"seed{seed}").mkdir(parents=True, exist_ok=True)
            (output / "edits" / f"seed{seed}" / f"{method}.json").write_text(json.dumps(artifact, indent=1) + "\n")
            result = {"pick": pick, "program": program, **score_state(ctx, state, test=True), "weights_sha256": weights_digest(state),
                      "queries": len(oracle.queries), "refused": len(oracle.refused), "query_seconds": oracle.seconds,
                      "best_validation_by_query": [min(q["validation"] for q in oracle.queries[:i + 1]) for i in range(len(oracle.queries))],
                      **extra}
        result["seconds"] = time.perf_counter() - started
        out["methods"][method] = result
        log(f"seed {seed} {method}: validation {result['validation']:.4f}, test {result['test']:.4f}"
            + (f" (pick: {result['pick']})" if "pick" in result else ""))
    return out


def stats(values):
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def arms_of(seed_report):
    """Each arm of one seed: the references every method gets, then the methods."""
    refs = seed_report["references"]
    arms = {"base": refs["base"], "plain average": refs["average"],
            "best worker (by validation)": refs[f"{seed_report['best_worker']} alone"]}
    arms.update({f"{m}": r for m, r in seed_report["methods"].items()})
    return arms


def summarize(seeds):
    """Mean, min and max over seeds of each arm's scores, and each arm against the best worker on
    test: the mean difference and how many test sessions it does better or worse on."""
    out = {}
    names = list(arms_of(next(iter(seeds.values()))))
    for name in names:
        per_seed = [arms_of(s)[name] for s in seeds.values()]
        best = [arms_of(s)["best worker (by validation)"] for s in seeds.values()]
        entry = {k: stats([r[k] for r in per_seed]) for k in REPORTED if all(k in r for r in per_seed)}
        entry["test_minus_best_worker"] = stats([r["test"] - b["test"] for r, b in zip(per_seed, best)])
        pairs = [(a, b) for r, w in zip(per_seed, best) for a, b in zip(r["test_sessions"], w["test_sessions"])]
        entry["test_sessions_vs_best_worker"] = {"better": int(sum(a < b - 1e-9 for a, b in pairs)),
                                                 "worse": int(sum(a > b + 1e-9 for a, b in pairs)), "of": len(pairs)}
        if all("pick" in r for r in per_seed):
            entry["picks"] = [r["pick"] for r in per_seed]
        if all("cost" in r for r in per_seed):
            dollars = [r["cost"]["dollars"] for r in per_seed]
            entry["dollars"] = None if None in dollars else sum(dollars)
        entry["runs"] = len(per_seed)
        out[name] = entry
    return out


LABELS = {"random": "random search", "coordinate": "coordinate search", "llm": "WSP editor (Claude)",
          "llm-anonymized": "WSP editor (Claude), anonymized", "specialist": "specialist bank (4 models, not a merge)"}


def table(report):
    rows = ["| Arm | Validation | Test | Worst test session | Test − best worker | Test sessions better / worse |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, s in report["summary"].items():
        vs = s["test_sessions_vs_best_worker"]
        rows.append(f"| {LABELS.get(name, name)} | {s['validation']['mean']:.3f} | {s['test']['mean']:.3f} | {s['test_worst_session']['mean']:.2f} "
                    f"| {s['test_minus_best_worker']['mean']:+.3f} | {vs['better']} / {vs['worse']} of {vs['of']} |")
    return "\n".join(rows)


def sessions_arms(bench):
    """The session benchmark's own arms on the same cohort (its report's summary over seeds) and its
    train-fitted baselines, for beside ours."""
    report = json.loads((bench.folder / "report.json").read_text())
    arms = {name: {k: s[k]["mean"] for k in ("validation", "test", "test_worst_session") if k in s} for name, s in report["summary"].items()}
    return {**arms, **{f"baseline: {name}": {k: s[k] for k in ("validation", "test", "test_worst_session")}
                       for name, s in report["baselines"].items()}}


def training_compute(bench):
    """Seconds the session benchmark spent training each seed's base and workers (the cohort every
    method here shares)."""
    report = json.loads((bench.folder / "report.json").read_text())
    return {seed: {"base": s["base"]["training"]["seconds"], "workers": sum(w["training"]["seconds"] for w in s["workers"])}
            for seed, s in report["seeds"].items()}


def folder_of(bench):
    return Path(__file__).resolve().parent / "benchmarks" / f"wsp-{bench.name}"


def run(args):
    bench = sessions.benchmark(args.benchmark)
    if bench.name != "v0" and protocol_sha256() != PROTOCOL_SHA256:
        raise SystemExit(f"{bench.id} is scored once, with the protocol pinned before it was cut; "
                         f"this code's protocol is {protocol_sha256()}, not {PROTOCOL_SHA256}")
    methods = list(dict.fromkeys(args.methods))
    send = anthropic_send() if any(m.startswith("llm") for m in methods) else None
    anchor = load_anchor(sessions.ANCHOR)
    split = sessions.check_split(bench=bench)
    split_sha256 = digest(sessions.split_text(split).encode())
    ctx = sessions.Context(anchor, bench.manifest)
    output = args.output
    require_new_output(output)
    output.mkdir(parents=True)
    anchor_bytes = sum(v.numel() for v in anchor.theta0.values()) * 4
    report = {"format": "eiathome-wsp-bench-report", "version": 1, "benchmark": BENCHMARK, "sessions_benchmark": bench.id,
              "split_sha256": split_sha256, "protocol_sha256": protocol_sha256(), "pinned_protocol_sha256": PROTOCOL_SHA256,
              "code_sha256": sessions.code_sha256(), "wsp_code_sha256": digest(Path(__file__).read_bytes()),
              "environment": sessions.environment(),
              "settings": {"methods": methods, "seeds": list(args.seeds), "budget": BUDGET, "limits": LIMITS, "groups": list(GROUPS),
                           "search_seed": SEARCH_SEED, "editor": EDITOR, "editor_price_per_mtok": PRICE_PER_MTOK},
              "command": " ".join(["python -m tracks.llm.wsp_bench run", "--benchmark", bench.name, "--methods", *methods,
                                   "--seeds", *map(str, args.seeds)]),
              "communication_bytes": {"per_worker_delta": anchor_bytes, "per_merge": anchor_bytes * len(sessions.WORKER_SHARDS)},
              "training_seconds": training_compute(bench), "sessions_arms": sessions_arms(bench), "seeds": {}}
    for seed in args.seeds:
        report["seeds"][str(seed)] = run_seed(ctx, bench, split_sha256, seed, methods, send, output)
    report["summary"] = summarize(report["seeds"])
    (output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(table(report))
    if args.publish:
        publish(output, folder_of(bench))
    return report


def publish(output, folder):
    """The run's report, edit artifacts and transcripts, under benchmarks/wsp-<version>/."""
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(output.rglob("*.json")):
        target = folder / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def verify(bench=None, folder=None, tolerance=1e-9):
    """Replays every published pick: its inputs are the published cohort's, its program makes the
    weights its artifact records, and they score as the report says. The specialist bank is
    rescored too."""
    bench = sessions.benchmark(bench)
    folder = folder_of(bench) if folder is None else folder
    report = json.loads((folder / "report.json").read_text())
    if report["sessions_benchmark"] != bench.id:
        raise ValueError(f"{folder}: a report on {report['sessions_benchmark']}, not {bench.id}")
    anchor = load_anchor(sessions.ANCHOR)
    split = sessions.check_split(bench=bench)
    if digest(sessions.split_text(split).encode()) != report["split_sha256"]:
        raise ValueError(f"{folder}: the report is on another split")
    ctx = sessions.Context(anchor, bench.manifest)
    checked = 0
    for seed, seed_report in report["seeds"].items():
        cohort = load_cohort(bench, int(seed))
        if seed_report["inputs"] != cohort.digests:
            raise ValueError(f"seed {seed}: the report's inputs aren't the published cohort")
        for method, result in seed_report["methods"].items():
            if method == "specialist":
                again = specialist_bank(ctx, cohort)
            else:
                artifact = json.loads((folder / "edits" / f"seed{seed}" / f"{method}.json").read_text())
                if artifact["inputs"] != cohort.digests or artifact["program"] != result["program"]:
                    raise ValueError(f"seed {seed} {method}: the edit artifact doesn't match the report or the cohort")
                state = apply(cohort, artifact["program"])
                if weights_digest(state) != artifact["output_sha256"] or artifact["output_sha256"] != result["weights_sha256"]:
                    raise ValueError(f"seed {seed} {method}: the program doesn't make the weights its artifact records")
                for query in artifact["queries"]:
                    if weights_digest(apply(cohort, query["program"])) != query["weights_sha256"]:
                        raise ValueError(f"seed {seed} {method}: query {query['query']} doesn't replay")
                again = score_state(ctx, state, test=True)
            for key in REPORTED:
                if abs(again[key] - result[key]) > tolerance:
                    raise ValueError(f"seed {seed} {method}: {key} rescored {again[key]!r}, the report says {result[key]!r}")
            checked += 1
    print(f"{folder}: {checked} picks replayed on {len(report['seeds'])} seeds")
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    context = commands.add_parser("context", help="print what the editor reads first")
    context.add_argument("--benchmark", choices=("v0", "v1"), default="v0")
    context.add_argument("--seed", type=int, default=sessions.SEEDS[0])
    context.add_argument("--anonymized", action="store_true")
    run_ = commands.add_parser("run", help="score the methods on a published cohort")
    run_.add_argument("--benchmark", choices=("v0", "v1"), default="v0")
    run_.add_argument("--output", type=Path, required=True)
    run_.add_argument("--methods", nargs="+", choices=METHODS, default=["random", "coordinate", "specialist"])
    run_.add_argument("--seeds", nargs="+", type=int, default=list(sessions.SEEDS))
    run_.add_argument("--publish", action="store_true")
    check = commands.add_parser("verify", help="replay the published picks")
    check.add_argument("--benchmark", choices=("v0", "v1"), default="v0")
    commands.add_parser("protocol", help="print the protocol's SHA-256")
    args = parser.parse_args(argv)
    if args.command == "context":
        bench = sessions.benchmark(args.benchmark)
        ctx = sessions.Context(load_anchor(sessions.ANCHOR), bench.manifest)
        cohort = load_cohort(bench, args.seed)
        print(editor_context(ctx, cohort, bench, Oracle(ctx, cohort), args.anonymized))
    elif args.command == "run":
        run(args)
    elif args.command == "verify":
        verify(args.benchmark)
    else:
        print(protocol_sha256())


if __name__ == "__main__":
    main()
