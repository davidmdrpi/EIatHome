"""The milestone 5 benchmark (issue #20): route whole sequences to milestone 3b's warm specialist
bank, without the family label.

Kept whole, with each family sent to its own specialist by its label, milestone 3b's warm
specialists beat every dense model (3.37 bits per token against 3.56 for the largest), while
routing single tokens to experts lost (3.76). Here a router picks the specialist for each
sequence (a slice: 49 frames, 48 targets) from the tokens already seen, so no target ever helps
choose the model that predicts it.

Fixed before any run, all as in milestone 3b: anchor v3 on index-moe-v2.json (Voyager, Rosetta
and the RFI sweep, three scans each within a session), seeds 42, 43 and 44. Per seed, θ_warm
(10 epochs on the three training scans together, the best on their validation scans) and one
specialist per family from it (50 epochs, patience 8), trained exactly as moe_ablation trains
them, so they're milestone 3b's specialists, unchanged. Every model is scored as evaluate()
scores it, in windows of the context.

The routers. Before each target, every specialist gets a score from the tokens before it, with
an equal prior. The first token, which no specialist predicts, counts by its add-one frequency in
that specialist's family's training scan. Then:

- first token: that first token alone, for the whole sequence;
- token frequencies: every token so far, each by that frequency (naive Bayes over the 3 x 33
  frequencies, the only fitted numbers);
- specialists' likelihood: each specialist's own predictions of the tokens so far, followed
  throughout, or committed after the first 4 or 8 tokens (from then on, one specialist runs);
- posterior mixture: every specialist's prediction, weighted by that likelihood (Bayes's rule).

The hard routers send each target to the best-scoring specialist. One router is chosen per seed
on validation, by the mean over the three families: each family counts once, though Rosetta's
validation scan has twice the targets of the others. What the concatenated validation scans
(every target once, milestone 3b's objective) would choose is recorded alongside. Test is scored
afterwards, for every router.

Also recorded, for each family's sequences: what each specialist scores on them (so what sending
them to the wrong family costs), the share of targets each router sends to the family's own
specialist, and the parameters stored and run per target.

The selection check from milestone 3b's review: its three dense rivals (the anchor's size, the
mixture's active size and its total size), retrained as there, from the same starts with the same
seeds, with no early stopping. They train 104 epochs, more than any seed's budget there
(101–103). Their best epoch is chosen twice: on the concatenated validation scans, as milestone
3b did, and on the equal-family mean, with the Voyager and sweep validation scans counted twice.

    python -m tracks.llm.route_bench --output runs/route-bench
"""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from .anchor import load_anchor
from .data import load_slices, markov_probabilities
from .merge import model_from
from .model import evaluate, fit
from .moe import active_parameters, dense_start, load_specialists, load_warm, matched_config, mixed, scores, upcycle, warmup
from .moe_ablation import ANCHOR, FAMILIES, MANIFEST, TRAINING, WARM_EPOCHS, specialist, targets_per_epoch
from .runtime import require_new_output, training_runtime

COMMIT_AFTER = (4, 8)
DENSE_EPOCHS = 104  # more than milestone 3b's budget in any seed (101–103 epochs)
LOG2 = math.log(2)


# Per-target scores.

@torch.no_grad()
def target_bits(model, sequences):
    """Each next-token target's bits, per sequence, as evaluate() scores them: windows of the
    context from the sequence's start, never across sequences."""
    model.eval()
    context = model.config["context"]
    out = []
    for s in sequences:
        bits = np.empty(len(s) - 1)
        for start in range(0, len(s) - 1, context):
            y = np.asarray(s[start + 1:start + context + 1])
            x = torch.from_numpy(np.asarray(s[start:start + len(y)], dtype=np.int64))[None]
            logp = F.log_softmax(model(x)[0].double(), dim=-1)
            bits[start:start + len(y)] = -logp[np.arange(len(y)), y].numpy() / LOG2
        out.append(bits)
    return out


def frequencies(anchor, partitions):
    """log2 of each token's add-one frequency in each family's training scan: (families, vocab)."""
    k = anchor.spec["model"]["vocab_size"]
    return np.stack([np.log2(markov_probabilities([anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["train"]], k)[0])
                     for f in FAMILIES])


# Routers. Each maps one sequence (its tokens, every specialist's bits for its targets, and the
# families' log2 frequencies) to bits per target, plus the specialist each target went to (or
# the weight on each, for the mixture), using for target t only the tokens before it.

def evidence(tokens, bits, logfreq):
    """Each specialist's score before each target (columns: targets 1..n−1), two ways: by the
    tokens' frequencies, and by its own likelihood after the first token's frequency."""
    first = logfreq[:, tokens[0]][:, None]
    by_frequency = np.cumsum(logfreq[:, tokens[:-1]], axis=1)
    by_likelihood = first + np.concatenate([np.zeros((len(bits), 1)), np.cumsum(-bits, axis=1)[:, :-1]], axis=1)
    return by_frequency, by_likelihood


def hard(bits, scores_):
    choice = scores_.argmax(axis=0)
    return bits[choice, np.arange(bits.shape[1])], np.eye(len(bits))[choice].T


def mixture(bits, by_likelihood):
    """Bits of the posterior-weighted mixture, and the weights."""
    log_w = by_likelihood * LOG2
    log_w = log_w - log_w.max(axis=0)
    w = np.exp(log_w) / np.exp(log_w).sum(axis=0)
    p = (w * np.exp2(-bits)).sum(axis=0)
    return -np.log2(p), w


ROUTERS = {
    "first token": lambda tokens, bits, f: hard(bits, np.repeat(f[:, tokens[0]][:, None], bits.shape[1], axis=1)),
    "token frequencies": lambda tokens, bits, f: hard(bits, evidence(tokens, bits, f)[0]),
    "specialists' likelihood": lambda tokens, bits, f: hard(bits, evidence(tokens, bits, f)[1]),
    **{f"specialists' likelihood, committed after {n} tokens":
       (lambda tokens, bits, f, n=n: hard(bits, evidence(tokens, bits, f)[1][:, np.minimum(np.arange(bits.shape[1]), n - 1)]))
       for n in COMMIT_AFTER},
    "posterior mixture": lambda tokens, bits, f: mixture(bits, evidence(tokens, bits, f)[1]),
}


def active(router, targets, single, router_parameters):
    """Parameters run per target, on average over a sequence's `targets` targets."""
    if router in ("first token", "token frequencies"):
        return single + router_parameters
    if router.startswith("specialists' likelihood, committed after"):
        n = int(router.split()[-2])
        everyone = min(n - 1, targets)  # every specialist predicts the targets that decide
        return (everyone * len(FAMILIES) * single + (targets - everyone) * single) / targets + router_parameters
    return len(FAMILIES) * single + router_parameters


def route(per_family, logfreq):
    """Every router on one split: bits per family, the share of each family's targets sent to its
    own specialist, and both means. per_family: {family: (token sequences, [bits (K, n−1)])}."""
    out = {}
    for name, router in ROUTERS.items():
        bits, own, targets = {}, {}, {}
        for i, family in enumerate(FAMILIES):
            sequences, matrices = per_family[family]
            routed = [router(np.asarray(s), b, logfreq) for s, b in zip(sequences, matrices)]
            n = sum(len(r[0]) for r in routed)
            bits[family] = float(sum(r[0].sum() for r in routed) / n)
            own[family] = float(sum(r[1][i].sum() for r in routed) / n)
            targets[family] = n
        out[name] = {**bits, "mean": statistics.fmean(bits.values()),
                     "concatenated": sum(bits[f] * targets[f] for f in FAMILIES) / sum(targets.values()),
                     "own_specialist": own}
    return out


def equal_family(anchor, partitions, split):
    """A split's sequences with each family repeated so every family has as many targets: the
    concatenated mean over them is the mean over families."""
    sequences = {f: [anchor.tokenizer.encode(r["frames"]) for r in partitions[f][split]] for f in FAMILIES}
    counts = {f: sum(len(s) - 1 for s in seqs) for f, seqs in sequences.items()}
    most = math.lcm(*counts.values())
    return [s for f in FAMILIES for _ in range(most // counts[f]) for s in sequences[f]]


def one_seed(folder, seed, epochs, dense_epochs):
    anchor = load_anchor(ANCHOR)
    context = anchor.spec["model"]["context"]
    folder.mkdir(parents=True)
    with training_runtime(1, "cpu"):
        records = load_slices(MANIFEST)
        torch.save(warmup(anchor, records, FAMILIES, WARM_EPOCHS, seed), folder / "warm.pt")
    warm_path = folder / "warm.pt"
    trained = {f: specialist(folder / f"warm-{f}", f, seed, epochs, warm_path) for f in FAMILIES}

    out = {}
    with training_runtime(1, "cpu"):
        states, _, _, partitions = load_specialists(anchor, [trained[f][0] for f in FAMILIES], MANIFEST, warm_path)
        warm = load_warm(anchor, warm_path, MANIFEST)
        models = [model_from(anchor, s) for s in states]
        single = sum(p.numel() for p in models[0].parameters())
        logfreq = frequencies(anchor, partitions)
        router_parameters = logfreq.size
        train = mixed(anchor, partitions, "train")
        per_epoch = {f: targets_per_epoch([anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["train"]], context) for f in FAMILIES}
        mixed_epoch = targets_per_epoch(train, context)
        cost = sum(trained[f][1] * per_epoch[f] for f in FAMILIES) + WARM_EPOCHS * mixed_epoch

        splits = {}
        for split in ("validation", "test"):
            per_family = {}
            for f in FAMILIES:
                sequences = [anchor.tokenizer.encode(r["frames"]) for r in partitions[f][split]]
                per_model = [target_bits(m, sequences) for m in models]
                per_family[f] = (sequences, [np.stack([b[j] for b in per_model]) for j in range(len(sequences))])
            splits[split] = per_family
        # Every specialist on every family's sequences: the diagonal is the bank with the label.
        out["cross"] = {split: {f: {g: float(np.concatenate([m[k] for m in splits[split][f][1]]).mean()) for k, g in enumerate(FAMILIES)}
                                for f in FAMILIES} for split in splits}
        check = scores(models[0], anchor, {FAMILIES[0]: partitions[FAMILIES[0]]})[FAMILIES[0]]
        out["scoring_check"] = abs(check - out["cross"]["test"][FAMILIES[0]][FAMILIES[0]])
        out["routers"] = {split: route(splits[split], logfreq) for split in splits}
        lengths = [len(s) - 1 for f in FAMILIES for s in splits["test"][f][0]]
        out["active_parameters"] = {name: sum(active(name, n, single, router_parameters) * n for n in lengths) / sum(lengths) for name in ROUTERS}
        valid = out["routers"]["validation"]
        out["chosen"] = {"equal_family": min(ROUTERS, key=lambda r: valid[r]["mean"]),
                         "concatenated": min(ROUTERS, key=lambda r: valid[r]["concatenated"])}
        out["parameters"] = {"specialist": single, "stored": len(FAMILIES) * single + router_parameters, "router": router_parameters}
        out["cost"] = {"bank": cost, "warmup": WARM_EPOCHS * mixed_epoch, "targets_per_mixed_epoch": mixed_epoch,
                       "specialist_epochs": {f: trained[f][1] for f in FAMILIES}}

        # Milestone 3b's dense rivals, their epoch chosen both ways.
        moe = upcycle(anchor, states, base=warm.state)
        sizes = {"total": sum(p.numel() for p in moe.parameters()), "active": active_parameters(moe)}
        validation = {"concatenated": mixed(anchor, partitions, "validation"), "equal_family": equal_family(anchor, partitions, "validation")}
        out["dense"] = {}
        for name, config in (("anchor size", anchor.spec["model"]), ("the mixture's active parameters", matched_config(anchor.spec["model"], sizes["active"])),
                             ("the mixture's total parameters", matched_config(anchor.spec["model"], sizes["total"]))):
            init = anchor.theta0 if config == anchor.spec["model"] else dense_start(config)
            rows = {}
            for objective, sequences in validation.items():
                model, history, best = fit(train, sequences, config, dense_epochs, TRAINING["batch_size"], TRAINING["lr"], dense_epochs, seed,
                                           "cpu", 1, init_state=init)
                rows[objective] = {"test": scores(model, anchor, partitions), "validation": scores(model, anchor, partitions, "validation"),
                                   "best_epoch": best, "parameters": sum(p.numel() for p in model.parameters())}
            out["dense"][name] = {**rows, "cost": dense_epochs * mixed_epoch, "config": config}
    return out


def summarize(results):
    stats = lambda values: {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    seeds = list(results["seeds"].values())
    summary = {"routers": {}, "chosen": {}, "dense": {}}
    for name in ROUTERS:
        summary["routers"][name] = {
            **{key: stats([s["routers"]["test"][name][key] for s in seeds]) for key in (*FAMILIES, "mean")},
            "validation_mean": stats([s["routers"]["validation"][name]["mean"] for s in seeds]),
            "validation_concatenated": stats([s["routers"]["validation"][name]["concatenated"] for s in seeds]),
            "own_specialist": {f: stats([s["routers"]["test"][name]["own_specialist"][f] for s in seeds]) for f in FAMILIES},
            "active_parameters": statistics.fmean(s["active_parameters"][name] for s in seeds)}
    for objective in ("equal_family", "concatenated"):
        picks = [s["chosen"][objective] for s in seeds]
        summary["chosen"][objective] = {"routers": picks, **{key: stats([s["routers"]["test"][p][key] for s, p in zip(seeds, picks)])
                                                              for key in (*FAMILIES, "mean")}}
    summary["label"] = {f: stats([s["cross"]["test"][f][f] for s in seeds]) for f in FAMILIES}
    summary["label"]["mean"] = stats([statistics.fmean(s["cross"]["test"][f][f] for f in FAMILIES) for s in seeds])
    summary["label"]["validation_mean"] = stats([statistics.fmean(s["cross"]["validation"][f][f] for f in FAMILIES) for s in seeds])
    summary["cross"] = {f: {g: statistics.fmean(s["cross"]["test"][f][g] for s in seeds) for g in FAMILIES} for f in FAMILIES}
    for name in seeds[0]["dense"]:
        summary["dense"][name] = {objective: {**{key: stats([s["dense"][name][objective]["test"][key] for s in seeds]) for key in (*FAMILIES, "mean")},
                                              "validation_mean": stats([s["dense"][name][objective]["validation"]["mean"] for s in seeds]),
                                              "best_epochs": [s["dense"][name][objective]["best_epoch"] for s in seeds],
                                              "parameters": seeds[0]["dense"][name][objective]["parameters"]}
                                  for objective in ("concatenated", "equal_family")}
    summary["parameters"] = seeds[0]["parameters"]
    summary["cost"] = stats([s["cost"]["bank"] for s in seeds])
    summary["scoring_check"] = max(s["scoring_check"] for s in seeds)
    return summary


def table(summary):
    fmt = lambda s: f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})" if s["max"] - s["min"] >= 0.005 else f"{s['mean']:.2f}"
    lines = ["| | Voyager | Rosetta | RFI sweep | Mean | Validation mean | Own specialist (V, R, S) | Active per target |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| bank, by family label | " + " | ".join(fmt(summary["label"][k]) for k in (*FAMILIES, "mean"))
             + f" | {fmt(summary['label']['validation_mean'])} | 1, 1, 1 | {summary['parameters']['specialist']:,} |"]
    for name, row in summary["routers"].items():
        own = ", ".join(f"{row['own_specialist'][f]['mean']:.2f}" for f in FAMILIES)
        lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in (*FAMILIES, "mean"))
                     + f" | {fmt(row['validation_mean'])} | {own} | {row['active_parameters']:,.0f} |")
    for name, rows in summary["dense"].items():
        for objective, row in rows.items():
            lines.append(f"| dense, {name}, chosen on the {objective.replace('_', '-')} validation | "
                         + " | ".join(fmt(row[k]) for k in (*FAMILIES, "mean")) + f" | {fmt(row['validation_mean'])} | | {row['parameters']:,} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    p.add_argument("--dense-epochs", type=int, default=DENSE_EPOCHS)
    args = p.parse_args(argv)
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    results = {"protocol": __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0], "epochs": args.epochs, "dense_epochs": args.dense_epochs,
               "seeds": {}}
    for seed in args.seeds:
        results["seeds"][str(seed)] = one_seed(args.output / f"seed{seed}", seed, args.epochs, args.dense_epochs)
        print(f"seed {seed} done", flush=True)
    results["summary"] = summarize(results)
    (args.output / "report.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(table(results["summary"]))
    print(json.dumps({"chosen": results["summary"]["chosen"], "cross": results["summary"]["cross"],
                      "parameters": results["summary"]["parameters"], "cost": results["summary"]["cost"]}, indent=1))


if __name__ == "__main__":
    main()
