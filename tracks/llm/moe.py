"""Upcycling family specialists into a mixture of experts, Branch-Train-MiX (issue #20,
milestone 3).

Specialists branch from one anchor's θ₀ and each train on one family (train.py --anchor v2
--family F). Upcycling keeps apart what they learned separately and merges what they share:

- the backbone (token embedding, layer norms, attention, final norm, head) is merged: θ₀ plus
  the average, or the TIES merge, of the specialists' backbone deltas;
- each block's feed-forward layer becomes N experts, one per specialist, side by side;
- a linear router in each block picks two experts per token, with gates softmax(top2(x·W_g)),
  and the block adds Σ gᵢ FFNᵢ(x). A load-balancing loss, N Σᵢ fᵢ Pᵢ as in the Switch
  Transformer (fᵢ the share of routing slots expert i gets, Pᵢ its mean router probability),
  is 1 when tokens spread evenly and grows as they crowd onto fewer experts;
- the router is then finetuned on a mix of every family's training data, the experts frozen
  (and the backbone too, unless asked otherwise), and the epoch is chosen on the mix's
  validation split.

    python -m tracks.llm.moe upcycle --output runs/moe runs/voyager/real.pt runs/offsource/real.pt runs/rosetta/real.pt
    python -m tracks.llm.moe upcycle --finetune router --output runs/moe-tuned ...

The report scores each family's test split once, after the router is chosen.

Milestone 3b adds a warm start, as Branch-Train-MiX branches from a shared trained checkpoint
rather than from random weights: `warmup` trains θ₀ for a few epochs on every family's training
scan together, and specialists branch from that θ_warm (train.py --warm), so their attention and
embeddings start in one coordinate system. A warm start records the slices it trained on, and is
refused if they aren't the families' training splits; specialists and the mixture are checked to
share it.

    python -m tracks.llm.moe warmup --anchor v3 --output runs/warm
    python -m tracks.llm.train --anchor v3 --manifest ... --warm runs/warm/warm.pt --family rosetta ...
    python -m tracks.llm.moe upcycle --anchor v3 --warm runs/warm/warm.pt --output runs/moe ...
"""

import argparse
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import re
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .anchor import ANCHORS, ROOT, build_theta0, difference, load_anchor, weights_digest
from .data import SPLITS, dataset_digest, load_slices, split_records
from .merge import file_digest, lineage, load_checkpoint, recorded_shard, same_slices, ties
from .model import RotaryBlock, SignalLM, batches, evaluate, fit, windows
from .runtime import atomic_output, require_new_output, training_runtime

TOP_K = 2
AUX_WEIGHT = 0.01  # the Switch Transformer's load-balancing coefficient
FEED = re.compile(r"^blocks\.(\d+)\.feed\.(.+)$")
EXPERT = re.compile(r"^blocks\.(\d+)\.experts\.(\d+)\.(.+)$")
FINETUNE = {"none": (), "router": ("router",), "router+backbone": ("router", "backbone")}
WARM_EPOCHS = 10  # milestone 3b's joint warm-up, fixed before its first run


def feed_forward(width, dropout):
    return nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))


def balance_loss(logits, chosen, keep=None):
    """N Σᵢ fᵢ Pᵢ over the tokens kept: fᵢ the share of routing slots expert i gets, Pᵢ its mean
    router probability. 1 when the probabilities are even; up to N/k as the router collapses."""
    keep = torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device) if keep is None else keep
    slots = torch.zeros_like(logits).scatter(-1, chosen, 1.0)[keep]
    return logits.shape[-1] * (slots.mean(0) / chosen.shape[-1] * logits.softmax(dim=-1)[keep].mean(0)).sum()


class MoEBlock(RotaryBlock):
    """RotaryBlock with its feed-forward layer replaced by `experts` of them and a top-k router."""

    def __init__(self, width, heads, dropout, experts, top_k):
        super().__init__(width, heads, dropout)
        del self.feed
        self.top_k = min(top_k, experts)
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([feed_forward(width, dropout) for _ in range(experts)])

    def forward(self, x, mask=None, force=None):
        x = self.attend(x)
        h = self.norm2(x)
        logits = self.router(h)
        if force is None:
            top, chosen = logits.topk(self.top_k, dim=-1)
            gates = top.softmax(dim=-1)
        else:  # every token to one expert, gate 1: the oracle routing used as a diagnostic
            chosen = torch.full(logits.shape[:-1] + (1,), force, device=x.device)
            gates = torch.ones(chosen.shape, dtype=x.dtype, device=x.device)
        # Each expert runs only on the tokens routed to it; its gated outputs are added back in place.
        flat, slots, weights = h.reshape(-1, h.shape[-1]), chosen.reshape(-1, chosen.shape[-1]), gates.reshape(-1, gates.shape[-1])
        y = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            token, slot = (slots == index).nonzero(as_tuple=True)
            if len(token):
                y = y.index_add(0, token, weights[token, slot, None] * expert(flat[token]))
        self.balance, self.chosen = balance_loss(logits, chosen, mask), chosen
        return x + self.drop2(y.view_as(h))


class MoESignalLM(nn.Module):
    """SignalLM (rotary positions) with mixture-of-experts feed-forward layers."""

    def __init__(self, vocab_size, context, width, layers, heads, dropout, position="rope", experts=2, top_k=TOP_K):
        super().__init__()
        if position != "rope" or experts < 1 or top_k < 1:
            raise ValueError("A mixture of experts upcycles rotary SignalLMs, with at least one expert and top-k ≥ 1")
        self.config = dict(vocab_size=vocab_size, context=context, width=width, layers=layers, heads=heads,
                           dropout=dropout, position=position, experts=experts, top_k=top_k)
        self.token = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([MoEBlock(width, heads, dropout, experts, top_k) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.force = None  # an expert index to send every token to, instead of routing

    def forward(self, tokens, mask=None):
        if not 1 <= tokens.shape[1] <= self.config["context"]:
            raise ValueError("Input length exceeds the checkpoint's context")
        x = self.token(tokens)
        for block in self.blocks:
            x = block(x, mask, self.force)
        self.balance_loss = torch.stack([block.balance for block in self.blocks]).mean()
        return self.head(self.norm(x))


def active_parameters(model):
    """Parameters a token passes through: all but the experts its router doesn't pick, which
    don't run on it."""
    per_expert = sum(p.numel() for p in model.blocks[0].experts[0].parameters())
    unused = model.config["experts"] - min(model.config["top_k"], model.config["experts"])
    return sum(p.numel() for p in model.parameters()) - model.config["layers"] * unused * per_expert


def is_backbone(name):
    return not (FEED.match(name) or EXPERT.match(name) or ".router." in name)


def dense_parameters(config):
    return sum(p.numel() for p in SignalLM(**config).parameters())


def matched_config(config, target):
    """A dense SignalLM with about `target` parameters: the width (a multiple of 8) whose standard
    model, with feed-forward layers 4 × width, comes nearest, then the feed-forward size that
    brings it closest. Used to give a mixture a dense rival of its size."""
    width = min(range(8, 257, 8), key=lambda w: abs(dense_parameters({**config, "width": w}) - target))
    base = {**config, "width": width}
    one, slope = dense_parameters({**base, "ffn": 1}), dense_parameters({**base, "ffn": 2}) - dense_parameters({**base, "ffn": 1})
    return {**base, "ffn": max(1, 1 + round((target - one) / slope))}


def dense_start(config, seed=0):
    """θ₀ for a dense model of another size, drawn as an anchor's is (PCG64, uniform initializers)."""
    return OrderedDict((k, torch.from_numpy(v)) for k, v in build_theta0({"model": config, "theta_seed": seed}).items())


class Warm(NamedTuple):
    """A verified warm start: its weights, their digest, and the families and splits it covers."""
    state: OrderedDict
    weights_sha256: str
    families: list
    dataset_sha256: str
    evaluation: str


def family_splits(records, families, evaluation):
    """Each family's split as train.py makes it, refusing one that splits another way."""
    out = {}
    for family in families:
        parts, independent = split_records(records, family, allow_within_session=evaluation != "independent_sessions")
        if ("independent_sessions" if independent else "within_session_scan_holdout") != evaluation:
            raise ValueError(f"The {family} family doesn't split as {evaluation}")
        out[family] = parts
    return out


def warmup(anchor, records, families=None, epochs=WARM_EPOCHS, seed=42, batch_size=32, lr=3e-4,
           evaluation="within_session_scan_holdout"):
    """θ_warm as a checkpoint: θ₀ trained `epochs` epochs on the families' training scans
    together, keeping the best epoch on their validation scans. It records those slices."""
    families = list(families or anchor.spec["families"])
    partitions = family_splits(records, families, evaluation)
    train, validation = mixed(anchor, partitions, "train"), mixed(anchor, partitions, "validation")
    model, history, best = fit(train, validation, anchor.spec["model"], epochs, batch_size, lr, epochs, seed, "cpu", 1,
                               init_state=anchor.theta0)
    state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    return {
        "format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": anchor.spec["model"], "model": state,
        "tokenizer": anchor.tokenizer.state(), "anchor": anchor.identity(), "dataset_sha256": dataset_digest(records),
        "evaluation": evaluation,
        "splits": {name: [r["entry"] for f in families for r in partitions[f][name]] for name in SPLITS},
        "merge": {"warm": True, "round": 0, "start_sha256": anchor.spec["theta0_sha256"], "weights_sha256": weights_digest(state)},
        "warm": {"families": families, "epochs": epochs, "best_epoch": best, "seed": seed, "lr": lr, "batch_size": batch_size,
                 "training_targets": sum(len(t) - 1 for t in train), "history": history},
    }


def load_warm(anchor, path, manifest=None):
    """A warm start from this anchor's θ₀, verified: its weights match their recorded digest, the
    manifest is the one it trained on, and it trained on its families' training scans only."""
    checkpoint = load_checkpoint(path, anchor)
    merge, warm = checkpoint["merge"], checkpoint.get("warm") or {}
    if not merge.get("warm") or merge.get("start_sha256") != anchor.spec["theta0_sha256"] or not warm.get("families"):
        raise ValueError(f"{path}: not a warm start from anchor {anchor.name}'s θ₀")
    difference(checkpoint["model"], anchor.theta0)
    digest = weights_digest(checkpoint["model"])
    if digest != merge.get("weights_sha256"):
        raise ValueError(f"{path}: its weights don't match the digest it recorded")
    records = load_slices(manifest or ROOT / anchor.spec["manifest"])
    if dataset_digest(records) != checkpoint["dataset_sha256"]:
        raise ValueError(f"{path}: this manifest isn't the one the warm start was trained on")
    partitions = family_splits(records, warm["families"], checkpoint["evaluation"])
    for name in SPLITS:
        if not same_slices(checkpoint["splits"][name] or [], [r["entry"] for parts in partitions.values() for r in parts[name]]):
            raise ValueError(f"{path}: its {name} slices aren't its families' {name} splits")
    return Warm(checkpoint["model"], digest, list(warm["families"]), checkpoint["dataset_sha256"], checkpoint["evaluation"])


def upcycle(anchor, specialists, merge="average", ties_keep=1.0, router_seed=0, top_k=TOP_K, base=None):
    """A mixture of the specialists (state dicts from this anchor, in expert order): the backbone
    merged across them, each one's feed-forward layers as an expert, and a router initialized
    like a linear layer, uniform(±1/√width) from a PCG64 stream. base: the weights they branched
    from (θ₀ by default, or a warm start); the backbone is base plus their merged deltas."""
    base = anchor.theta0 if base is None else base
    taus = [difference(state, base) for state in specialists]
    backbone = [name for name in anchor.theta0 if is_backbone(name)]
    flats = [np.concatenate([tau[name].reshape(-1).numpy() for name in backbone]) for tau in taus]
    weights = [1.0 / len(specialists)] * len(specialists)
    merged = ties(flats, weights, ties_keep) if merge == "ties" else sum(w * f for w, f in zip(weights, flats))
    offset, delta = 0, {}
    for name in backbone:
        size = anchor.theta0[name].numel()
        delta[name] = torch.from_numpy(merged[offset:offset + size].reshape(anchor.theta0[name].shape))
        offset += size
    model = MoESignalLM(**anchor.spec["model"], experts=len(specialists), top_k=top_k)
    rng = np.random.Generator(np.random.PCG64(router_seed))
    state = OrderedDict()
    for name, value in model.state_dict().items():
        expert = EXPERT.match(name)
        if expert:
            block, index, rest = expert.groups()
            state[name] = specialists[int(index)][f"blocks.{block}.feed.{rest}"].float().clone()
        elif ".router." in name:
            bound = 1.0 / math.sqrt(value.shape[1])
            state[name] = torch.from_numpy(((rng.random(tuple(value.shape)) * 2.0 - 1.0) * bound).astype(np.float32))
        else:
            state[name] = (base[name].double() + delta[name]).float()
    model.load_state_dict(state)
    return model


def load_specialists(anchor, paths, manifest=None, warm=None):
    """Specialists to upcycle: workers trained (round 1) from θ₀, or all from the warm start
    `warm` (a path), on distinct families of one dataset, split as the manifest splits it, each
    on its family's whole training split. Returns (states, families, records, partitions)."""
    loaded = [(str(path), load_checkpoint(path, anchor)) for path in paths]
    if len(loaded) < 2:
        raise ValueError("A mixture needs at least two specialists")
    start = load_warm(anchor, warm, manifest) if warm is not None else None
    start_sha = anchor.spec["theta0_sha256"] if start is None else start.weights_sha256
    for path, c in loaded:
        merge = c["merge"]
        if c.get("variant") != "real" or merge.get("merged") or merge.get("warm") or merge.get("round") != 1 or merge.get("start_sha256") != start_sha:
            raise ValueError(f"{path}: not a worker trained on the real stream from {'θ₀' if start is None else 'this warm start'}")
    chains = [lineage(c) for _, c in loaded]
    families = [chain["family"] for chain in chains]
    if len(set(families)) != len(families):
        raise ValueError(f"Each specialist must have a family of its own; these are {', '.join(families)}")
    if start is not None and not set(families) <= set(start.families):
        raise ValueError(f"The warm start covers {', '.join(start.families)}, not every specialist's family")
    if len({(chain["dataset_sha256"], chain["evaluation"]) for chain in chains}) != 1:
        raise ValueError("The specialists must be trained on one dataset, split the same way")
    records = load_slices(manifest or ROOT / anchor.spec["manifest"])
    if dataset_digest(records) != chains[0]["dataset_sha256"]:
        raise ValueError("This manifest isn't the one the specialists were trained on")
    partitions = {}
    for (path, c), chain in zip(loaded, chains):
        family = chain["family"]
        parts, _ = split_records(records, family, allow_within_session=chain["evaluation"] != "independent_sessions")
        if any(chain[name] != [r["entry"] for r in parts[name]] for name in ("validation", "test")):
            raise ValueError(f"The {family} specialist's validation or test split isn't this manifest's")
        # Its record must not contradict the split: the family's whole training split, and its targets.
        if recorded_shard(path, c["merge"]) != (1, 1):
            raise ValueError(f"{path}: a specialist trains on its family's whole training split, not a shard of it")
        if not same_slices(c["splits"]["train"] or [], [r["entry"] for r in parts["train"]]):
            raise ValueError(f"{path}: its training slices aren't the {family} training split")
        targets = sum(len(r["frames"]) - 1 for r in parts["train"])
        if c["merge"].get("training_targets") != targets:
            raise ValueError(f"{path}: it records {c['merge'].get('training_targets')!r} training targets, but the {family} training split has {targets}")
        partitions[family] = parts
    return [c["model"] for _, c in loaded], families, records, partitions


def finetune(model, train, validation, epochs, batch_size, lr, patience, seed, parts=("router",), aux_weight=AUX_WEIGHT, stride=1):
    """Trains the router (and the backbone, if `parts` names it) on mixed sequences, the experts
    always frozen; keeps the epoch with the best validation bits, as fit() does."""
    torch.random.default_generator.manual_seed(seed)
    trainable = lambda name: (".router." in name and "router" in parts) or (is_backbone(name) and "backbone" in parts)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable(name))
    params = [p for p in model.parameters() if p.requires_grad]
    best, best_epoch = evaluate(model, validation, batch_size)["bits_per_token"], 0
    best_state, history = copy.deepcopy(model.state_dict()), [{"epoch": 0, "validation_bits": best}]
    if not params:
        return model, history, 0
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    examples = windows(train, model.config["context"], stride)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        total = balance = count = 0.0
        for x, y in batches(examples, batch_size, "cpu", rng.permutation(len(examples))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask=y != -100)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            (loss + aux_weight * model.balance_loss).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            n = (y != -100).sum().item()
            total, balance, count = total + loss.item() * n, balance + model.balance_loss.item() * n, count + n
        val = evaluate(model, validation, batch_size)["bits_per_token"]
        history.append({"epoch": epoch, "train_bits": total / count / math.log(2), "balance": balance / count, "validation_bits": val})
        if val < best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, history, best_epoch


@torch.no_grad()
def routing(model, sequences):
    """Per block, the share of tokens whose first choice is each expert, and the balance loss."""
    model.eval()
    counts = [torch.zeros(model.config["experts"]) for _ in model.blocks]
    balance, n = 0.0, 0
    for x, y in batches(windows(sequences, model.config["context"]), 32, "cpu"):
        mask = y != -100
        model(x, mask=mask)
        for i, block in enumerate(model.blocks):
            counts[i] += torch.bincount(block.chosen[..., 0][mask], minlength=model.config["experts"]).float()
        balance += model.balance_loss.item() * mask.sum().item()
        n += mask.sum().item()
    return {"first_choice": [(c / c.sum()).tolist() for c in counts], "balance": balance / n}


@torch.no_grad()
def scores(model, anchor, partitions, split="test", force=None):
    """Bits per token on each family's split; force sends every token to one expert (an index,
    or the family's own: 'own')."""
    out = {}
    for i, (family, parts) in enumerate(partitions.items()):
        tokens = [anchor.tokenizer.encode(r["frames"]) for r in parts[split]]
        if isinstance(model, MoESignalLM):
            model.force = i if force == "own" else force
        out[family] = evaluate(model, tokens)["bits_per_token"]
    if isinstance(model, MoESignalLM):
        model.force = None
    out["mean"] = float(np.mean([out[f] for f in partitions]))
    return out


def mixed(anchor, partitions, split):
    return [anchor.tokenizer.encode(r["frames"]) for parts in partitions.values() for r in parts[split]]


def run(args):
    require_new_output(args.output)
    anchor = load_anchor(args.anchor)
    with training_runtime(1, "cpu"), atomic_output(args.output) as output:
        states, families, records, partitions = load_specialists(anchor, args.specialists, args.manifest, args.warm)
        warm = load_warm(anchor, args.warm, args.manifest) if args.warm else None
        model = upcycle(anchor, states, args.merge, args.ties_keep, args.router_seed, base=None if warm is None else warm.state)
        history, best_epoch = [], 0
        if FINETUNE[args.finetune]:
            model, history, best_epoch = finetune(model, mixed(anchor, partitions, "train"), mixed(anchor, partitions, "validation"),
                                                  args.epochs, args.batch_size, args.lr, args.patience, args.seed, FINETUNE[args.finetune])
        report = {
            "format": "eiathome-signal-lm-moe", "version": 1, "anchor": anchor.identity(), "experts": families,
            "start": "theta0" if warm is None else {"warm": str(args.warm), "weights_sha256": warm.weights_sha256},
            "specialists": [{"file": str(p), "file_sha256": file_digest(p), "family": f, "weights_sha256": weights_digest(s)}
                            for p, f, s in zip(args.specialists, families, states)],
            "rule": {"backbone": args.merge, "ties_keep": args.ties_keep, "top_k": TOP_K, "router_seed": args.router_seed,
                     "finetune": args.finetune, "aux_weight": AUX_WEIGHT, "epochs": args.epochs, "lr": args.lr,
                     "batch_size": args.batch_size, "patience": args.patience, "seed": args.seed},
            "parameters": sum(p.numel() for p in model.parameters()), "active_parameters": active_parameters(model),
            "best_epoch": best_epoch, "history": history,
            "validation": scores(model, anchor, partitions, "validation"),
            # Test is scored once, after the router is chosen.
            "test": scores(model, anchor, partitions, "test"),
            "routing": {f: routing(model, [anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["test"]]) for f in families},
        }
        torch.save({"format": report["format"], "version": 1, "config": model.config, "anchor": anchor.identity(),
                    "experts": families, "model": model.state_dict(), "tokenizer": anchor.tokenizer.state(), "rule": report["rule"]},
                   output / "moe.pt")
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def run_warmup(args):
    require_new_output(args.output)
    anchor = load_anchor(args.anchor)
    with training_runtime(1, "cpu"), atomic_output(args.output) as output:
        checkpoint = warmup(anchor, load_slices(args.manifest or ROOT / anchor.spec["manifest"]), args.families,
                            args.epochs, args.seed)
        torch.save(checkpoint, output / "warm.pt")
        (output / "report.json").write_text(json.dumps({"anchor": anchor.identity(), "merge": checkpoint["merge"],
                                                         "warm": checkpoint["warm"]}, indent=2) + "\n")
    return checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("warmup", help="train θ₀ briefly on every family together, for specialists to branch from")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--anchor", default="v3", choices=list(ANCHORS))
    w.add_argument("--manifest", type=Path, default=None, help="default: the anchor's manifest")
    w.add_argument("--families", nargs="+", default=None, help="default: the anchor's families")
    w.add_argument("--epochs", type=int, default=WARM_EPOCHS)
    w.add_argument("--seed", type=int, default=42)
    u = sub.add_parser("upcycle", help="build a mixture of experts from family specialists")
    u.add_argument("specialists", nargs="+", type=Path, help="each family's real.pt, trained with --anchor from θ₀ or --warm")
    u.add_argument("--warm", type=Path, default=None, help="the warm start the specialists branched from (default: θ₀)")
    u.add_argument("--output", type=Path, required=True)
    u.add_argument("--anchor", default="v2", choices=list(ANCHORS))
    u.add_argument("--manifest", type=Path, default=None, help="default: the anchor's manifest")
    u.add_argument("--merge", choices=("average", "ties"), default="average", help="how the backbone is merged")
    u.add_argument("--ties-keep", type=float, default=1.0)
    u.add_argument("--router-seed", type=int, default=0)
    u.add_argument("--finetune", choices=list(FINETUNE), default="router")
    u.add_argument("--epochs", type=int, default=50)
    u.add_argument("--lr", type=float, default=3e-4)
    u.add_argument("--batch-size", type=int, default=32)
    u.add_argument("--patience", type=int, default=8)
    u.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    try:
        if args.command == "warmup":
            if args.epochs < 1:
                raise ValueError("Need a positive number of epochs")
            checkpoint = run_warmup(args)
            print(json.dumps({"weights_sha256": checkpoint["merge"]["weights_sha256"], "families": checkpoint["warm"]["families"],
                              "best_epoch": checkpoint["warm"]["best_epoch"]}, indent=2))
            return
        if not (0 < args.ties_keep <= 1) or min(args.epochs, args.batch_size, args.patience) < 1 or not args.lr > 0:
            raise ValueError("Need 0 < --ties-keep ≤ 1, and positive epochs, batch size, patience and learning rate")
        report = run(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({"test": report["test"], "parameters": report["parameters"], "active_parameters": report["active_parameters"]}, indent=2))


if __name__ == "__main__":
    main()
