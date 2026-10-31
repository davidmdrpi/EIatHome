"""The milestone 3b ablation (issue #20): with experts that each know something, and a warm start
in common, does a Branch-Train-MiX mixture beat dense models trained on the same data, given
the same training budget?

Fixed before any run: anchor v3 (the peak tokenizer, under which each Rosetta symbol is exactly
one token) on index-moe-v2.json: Voyager, Rosetta (six slices per scan) and an RFI sweep, each
split into three scans within a session. The real stream, training windows at every start
(stride 1), the defaults otherwise (learning rate 0.0003, batch 32, dropout 0.1). Seeds 42, 43
and 44. Per seed:

- θ_warm: θ₀ trained 10 epochs on the three training scans together, the best epoch on the
  three validation scans;
- specialists, one per family, trained from θ₀ (cold) and from θ_warm (warm), 50 epochs,
  patience 8. Kept whole, they're also a bank of models dispatched by the family label;
- for each start, the mixture (the specialists' feed-forward layers as experts, a top-2 router,
  the rest averaged), scored
  - with every token sent to its family's own expert (the oracle; it uses the family label);
  - after finetuning the router, experts and backbone frozen;
  - after finetuning the router and the backbone, experts frozen;
  finetuning: 50 epochs, patience 8, load-balancing weight 0.01, chosen on the validation scans;
- dense models trained from their own θ₀ on the three training scans together, chosen on the
  validation scans:
  - the anchor's size (27,584 parameters), as milestone 3 trained it (50 epochs, patience 8);
  - the anchor's size, the mixture's active parameters and its total parameters (the width,
    then the feed-forward size, that come closest), each trained for as many epochs as it
    takes to see at least as many training targets as the costliest mixture pipeline of that
    seed, with no early stopping (the best epoch is still kept).

A pipeline's cost is every next-token target its training saw: the warm-up, the specialists
(the epochs each ran) and the finetuning. Test is scored once per model, afterwards.

Added after review, as a control for Rosetta: byte statistics that know the UTF-8 framing. A
high-nibble distribution, and each low nibble's distribution given its high nibble, fitted on the
training scan with add-one smoothing; on test, each frame's phase (high or low nibble) comes from
its slice's symbol offset. Scored over the 33 tokens and over the 16 symbols. It uses that
framing, which no model is given, and no context beyond the byte.

    python -m tracks.llm.moe_ablation --output runs/moe-ablation
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from . import synthetic
from .anchor import ROOT, load_anchor
from .data import load_slices, markov_probabilities
from .merge import model_from
from .model import fit, windows
from .moe import (AUX_WEIGHT, active_parameters, dense_start, finetune, load_specialists, load_warm, matched_config, mixed,
                  routing, scores, upcycle, warmup)
from .runtime import require_new_output, training_runtime
from .train import parser, run

MANIFEST = ROOT / "tracks/llm/data/slices/index-moe-v2.json"
ANCHOR = "v3"
FAMILIES = ("voyager", "rosetta", "sweep")
TRAINING = dict(epochs=50, batch_size=32, lr=3e-4, patience=8)
WARM_EPOCHS = 10


def specialist(output, family, seed, epochs, warm=None):
    argv = ["--anchor", ANCHOR, "--manifest", str(MANIFEST), "--family", family, "--allow-within-session", "--variants", "real",
            "--train-stride", "1", "--epochs", str(epochs), "--seed", str(seed), "--output", str(output)]
    argv += ["--warm", str(warm)] if warm else []
    with contextlib.redirect_stdout(io.StringIO()):
        report = run(parser().parse_args(argv))
    return output / "real.pt", len(report["variants"]["real"]["history"])


def targets_per_epoch(sequences, context):
    """Next-token targets one epoch at stride 1 trains on (each window predicts up to `context`)."""
    return sum(len(y) for _, y in windows(sequences, context, 1))


def row(model, anchor, partitions, **extra):
    out = {"test": scores(model, anchor, partitions), "validation": scores(model, anchor, partitions, "validation"),
           "parameters": sum(p.numel() for p in model.parameters()), **extra}
    if hasattr(model, "force"):
        out["active_parameters"] = active_parameters(model)
        out["routing"] = {f: routing(model, [anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["test"]])["first_choice"]
                          for f in partitions}
    return out


def baselines(anchor, partitions):
    """Each family's unigram and Markov-1 bits on test, fitted on its training scan."""
    out = {}
    k = anchor.spec["model"]["vocab_size"]
    for family, parts in partitions.items():
        train = [anchor.tokenizer.encode(r["frames"]) for r in parts["train"]]
        test = [anchor.tokenizer.encode(r["frames"]) for r in parts["test"]]
        unigram, transitions = markov_probabilities(train, k)
        n = sum(len(t) - 1 for t in test)
        out[family] = {"unigram": float(sum(-np.log2(unigram[t[1:]]).sum() for t in test) / n),
                       "markov1": float(sum(-np.log2(transitions[t[:-1], t[1:]]).sum() for t in test) / n)}
        if family == "rosetta":
            tokens = lambda r: anchor.tokenizer.encode(r["frames"])
            symbols = lambda r: synthetic.symbols_of(r["entry"])
            out[family]["framing_tokens"] = framing_baseline(parts["train"], parts["test"], tokens, k)
            out[family]["framing_symbols"] = framing_baseline(parts["train"], parts["test"], symbols, 16)
    return out


def framing_baseline(parts_train, parts_test, encode, k):
    """Bits per Rosetta target from byte statistics with known framing: P(high nibble), and
    P(low nibble | its high nibble), add-one smoothed over k values and fitted on the training
    slices; each frame's phase comes from its slice's symbol offset."""
    high, low = np.ones(k), np.ones((k, k))
    for r in parts_train:
        t, start = encode(r), r["entry"]["source"]["symbols"][0]
        for i in range(len(t)):
            if (start + i) % 2 == 0:
                high[t[i]] += 1
            elif i > 0:
                low[t[i - 1], t[i]] += 1
    high, low = high / high.sum(), low / low.sum(axis=1, keepdims=True)
    bits = []
    for r in parts_test:
        t, start = encode(r), r["entry"]["source"]["symbols"][0]
        bits += [-np.log2(high[t[i]] if (start + i) % 2 == 0 else low[t[i - 1], t[i]]) for i in range(1, len(t))]
    return float(np.mean(bits))


def one_seed(folder, seed, epochs):
    anchor = load_anchor(ANCHOR)
    context = anchor.spec["model"]["context"]
    folder.mkdir(parents=True)
    with training_runtime(1, "cpu"):
        records = load_slices(MANIFEST)
        warm_checkpoint = warmup(anchor, records, FAMILIES, WARM_EPOCHS, seed)
        torch.save(warm_checkpoint, folder / "warm.pt")
    warm_path = folder / "warm.pt"
    trained = {start: {f: specialist(folder / f"{start}-{f}", f, seed, epochs, warm_path if start == "warm" else None) for f in FAMILIES}
               for start in ("cold", "warm")}

    out = {"rows": {}, "cost": {}}
    with training_runtime(1, "cpu"):
        _, _, _, partitions = load_specialists(anchor, [trained["cold"][f][0] for f in FAMILIES], MANIFEST)
        train, validation = mixed(anchor, partitions, "train"), mixed(anchor, partitions, "validation")
        per_epoch = {f: targets_per_epoch([anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["train"]], context) for f in FAMILIES}
        mixed_epoch = targets_per_epoch(train, context)
        out["baselines"] = baselines(anchor, partitions)
        out["rows"]["θ₀, untrained"] = row(anchor.model(), anchor, partitions)

        sizes = {}
        for start in ("cold", "warm"):
            warm = load_warm(anchor, warm_path, MANIFEST) if start == "warm" else None
            paths = [trained[start][f][0] for f in FAMILIES]
            states, _, _, _ = load_specialists(anchor, paths, MANIFEST, warm_path if start == "warm" else None)
            base = None if warm is None else warm.state
            spent = sum(trained[start][f][1] * per_epoch[f] for f in FAMILIES) + (WARM_EPOCHS * mixed_epoch if start == "warm" else 0)
            # The specialists kept whole, each family sent to its own by the label: all are stored, one runs.
            bank = {split: {f: scores(model_from(anchor, s), anchor, {f: partitions[f]}, split)[f] for f, s in zip(FAMILIES, states)}
                    for split in ("test", "validation")}
            single = sum(p.numel() for p in anchor.model().parameters())
            out["rows"][f"specialist bank, dispatched by family label ({start})"] = {
                **{split: {**v, "mean": statistics.fmean(v.values())} for split, v in bank.items()},
                "parameters": len(FAMILIES) * single, "active_parameters": single, "cost": spent}
            moe = upcycle(anchor, states, base=base)
            sizes = {"total": sum(p.numel() for p in moe.parameters()), "active": active_parameters(moe)}
            out["rows"][f"mixture ({start}), oracle routing"] = {
                "test": scores(moe, anchor, partitions, force="own"), "validation": scores(moe, anchor, partitions, "validation", force="own"),
                "parameters": sizes["total"], "active_parameters": sizes["active"], "cost": spent}
            for name, parts in (("router finetuned", ("router",)), ("router and backbone finetuned", ("router", "backbone"))):
                model, history, best = finetune(upcycle(anchor, states, base=base), train, validation, epochs, TRAINING["batch_size"],
                                                TRAINING["lr"], TRAINING["patience"], seed, parts, AUX_WEIGHT)
                cost = spent + (len(history) - 1) * mixed_epoch
                out["rows"][f"mixture ({start}), {name}"] = row(model, anchor, partitions, best_epoch=best, cost=cost)

        budget = max(r["cost"] for name, r in out["rows"].items() if name.startswith("mixture") and "finetuned" in name)
        out["cost"] = {"budget": budget, "targets_per_mixed_epoch": mixed_epoch}
        model, history, best = fit(train, validation, anchor.spec["model"], epochs, TRAINING["batch_size"], TRAINING["lr"],
                                   TRAINING["patience"], seed, "cpu", 1, init_state=anchor.theta0)
        out["rows"]["dense, anchor size, milestone 3's training"] = row(model, anchor, partitions, best_epoch=best,
                                                                        cost=(len(history)) * mixed_epoch)
        budget_epochs = math.ceil(budget / mixed_epoch)
        for name, config in (("dense, anchor size", anchor.spec["model"]),
                             ("dense, the mixture's active parameters", matched_config(anchor.spec["model"], sizes["active"])),
                             ("dense, the mixture's total parameters", matched_config(anchor.spec["model"], sizes["total"]))):
            init = anchor.theta0 if config == anchor.spec["model"] else dense_start(config)
            model, history, best = fit(train, validation, config, budget_epochs, TRAINING["batch_size"], TRAINING["lr"],
                                       budget_epochs, seed, "cpu", 1, init_state=init)
            out["rows"][f"{name}, the mixtures' budget"] = row(model, anchor, partitions, best_epoch=best, cost=budget_epochs * mixed_epoch,
                                                                config=config)
    return out


def summarize(results):
    stats = lambda values: {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    seeds = list(results["seeds"].values())
    out = {}
    for name in seeds[0]["rows"]:
        runs = [s["rows"][name] for s in seeds]
        out[name] = {key: stats([r["test"][key] for r in runs]) for key in (*FAMILIES, "mean")}
        if "validation" in runs[0]:
            out[name]["validation_mean"] = stats([r["validation"]["mean"] for r in runs])
        for key in ("parameters", "active_parameters"):
            if key in runs[0]:
                out[name][key] = runs[0][key]
        if "cost" in runs[0]:
            out[name]["cost"] = stats([r["cost"] for r in runs])
        if "routing" in runs[0]:
            out[name]["own_expert_share"] = {f: stats([statistics.fmean(layer[i] for layer in r["routing"][f]) for r in runs])
                                             for i, f in enumerate(FAMILIES)}
    out["baselines"] = {f: {k: statistics.fmean(s["baselines"][f][k] for s in seeds) for k in seeds[0]["baselines"][f]} for f in FAMILIES}
    return out


def table(summary):
    fmt = lambda s: f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})" if s["max"] - s["min"] >= 0.005 else f"{s['mean']:.2f}"
    lines = ["| | Voyager | Rosetta | RFI sweep | Mean | Validation mean | Parameters (active) | Training targets |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        if name == "baselines":
            continue
        params = "" if "parameters" not in row else f"{row['parameters']:,}" + (f" ({row['active_parameters']:,})" if "active_parameters" in row else "")
        cost = "" if "cost" not in row else f"{row['cost']['mean'] / 1000:,.0f}k"
        valid = fmt(row["validation_mean"]) if "validation_mean" in row else ""
        lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in (*FAMILIES, "mean")) + f" | {valid} | {params} | {cost} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    args = p.parse_args(argv)
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    results = {"protocol": __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0], "epochs": args.epochs, "seeds": {}}
    for seed in args.seeds:
        results["seeds"][str(seed)] = one_seed(args.output / f"seed{seed}", seed, args.epochs)
        print(f"seed {seed} done", flush=True)
    results["summary"] = summarize(results)
    (args.output / "report.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(table(results["summary"]))


if __name__ == "__main__":
    main()
