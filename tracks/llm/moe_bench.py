"""The milestone 3 benchmark (issue #20): does a mixture of family specialists, upcycled
Branch-Train-MiX style, beat one dense model trained on all the families?

Fixed before any run: anchor v2 on the slices with the Rosetta control (index-rosetta.json):
Voyager, off-source and Rosetta, each split into three scans (train, validation, test), within
one session. The real stream only, training windows at every start (stride 1), 50 epochs with
the defaults otherwise (learning rate 0.0003, batch 32, patience 8, dropout 0.1). Seeds 42,
43 and 44. Per seed:

- three specialists, one per family, each trained from θ₀ on its family's training scan;
- one dense model trained from θ₀ on all three training scans together, chosen on all three
  validation scans: the same data and epochs as the specialists together;
- the specialists merged into one dense model: their average, and TIES (top 20%);
- the mixture: the specialists' feed-forward layers as three experts, a top-2 router, and the
  rest averaged (the Branch-Train-MiX default), scored
  - with its untrained router;
  - with every token sent to its family's own expert: a diagnostic, since it uses the family
    label a router doesn't have;
  - after finetuning the router on the three training scans, experts and backbone frozen;
  - after finetuning the router and the backbone, experts frozen;
- the same, with the backbone merged by TIES (top 20%), router and backbone finetuned.

Finetuning uses the same settings as training (50 epochs, learning rate 0.0003, batch 32,
patience 8, stride 1) and a load-balancing weight of 0.01, chosen on the three validation scans.
Test is scored once per model, afterwards.

    python -m tracks.llm.moe_bench --output runs/moe-bench
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics

from .anchor import ROOT, difference, flatten, load_anchor
from .merge import model_from, ties, unflatten
from .model import fit
from .moe import AUX_WEIGHT, active_parameters, finetune, load_specialists, mixed, routing, scores, upcycle
from .runtime import require_new_output, training_runtime
from .train import parser, run

MANIFEST = ROOT / "tracks/llm/data/slices/index-rosetta.json"
FAMILIES = ("voyager", "offsource", "rosetta")
TRAINING = dict(epochs=50, batch_size=32, lr=3e-4, patience=8)
TIES_KEEP = 0.2


def specialist(output, family, seed, epochs):
    argv = ["--anchor", "v2", "--manifest", str(MANIFEST), "--family", family, "--allow-within-session", "--variants", "real",
            "--train-stride", "1", "--epochs", str(epochs), "--seed", str(seed), "--output", str(output)]
    with contextlib.redirect_stdout(io.StringIO()):
        run(parser().parse_args(argv))
    return output / "real.pt"


def dense_merge(anchor, states, rule):
    """All of the specialists' parameters merged into one dense model."""
    flats = [flatten(difference(s, anchor.theta0)) for s in states]
    weights = [1.0 / len(states)] * len(states)
    merged = ties(flats, weights, TIES_KEEP) if rule == "ties" else sum(w * f for w, f in zip(weights, flats))
    tau = unflatten(anchor, merged)
    return model_from(anchor, {k: (v.double() + tau[k]).float() for k, v in anchor.theta0.items()})


def mixture_row(model, anchor, partitions, extra=None):
    out = {"test": scores(model, anchor, partitions), "validation": scores(model, anchor, partitions, "validation"),
           "parameters": sum(p.numel() for p in model.parameters())}
    if hasattr(model, "force"):
        out["active_parameters"] = active_parameters(model)
        out["routing"] = {f: routing(model, [anchor.tokenizer.encode(r["frames"]) for r in partitions[f]["test"]])
                          for f in partitions}
    return {**out, **(extra or {})}


def one_seed(folder, seed, epochs):
    anchor = load_anchor("v2")
    paths = [specialist(folder / family, family, seed, epochs) for family in FAMILIES]
    states, families, _, partitions = load_specialists(anchor, paths, MANIFEST)
    train, validation = mixed(anchor, partitions, "train"), mixed(anchor, partitions, "validation")
    settings = {**TRAINING, "epochs": epochs}
    rows = {}
    with training_runtime(1, "cpu"):
        rows["θ₀, untrained"] = mixture_row(anchor.model(), anchor, partitions)
        for family, state in zip(families, states):
            rows[f"{family} specialist"] = mixture_row(model_from(anchor, state), anchor, partitions)
        dense, history, best = fit(train, validation, anchor.spec["model"], settings["epochs"], settings["batch_size"],
                                   settings["lr"], settings["patience"], seed, "cpu", 1, init_state=anchor.theta0)
        rows["one dense model on all three families"] = mixture_row(dense, anchor, partitions, {"best_epoch": best})
        rows["specialists averaged (dense)"] = mixture_row(dense_merge(anchor, states, "average"), anchor, partitions)
        rows["specialists merged by TIES (dense)"] = mixture_row(dense_merge(anchor, states, "ties"), anchor, partitions)

        moe = upcycle(anchor, states)
        rows["mixture, untrained router"] = mixture_row(moe, anchor, partitions)
        rows["mixture, each family to its own expert (oracle)"] = {
            "test": scores(moe, anchor, partitions, force="own"), "validation": scores(moe, anchor, partitions, "validation", force="own")}
        for name, parts, merge in [("mixture, router finetuned", ("router",), "average"),
                                   ("mixture, router and backbone finetuned", ("router", "backbone"), "average"),
                                   ("mixture (TIES backbone), router and backbone finetuned", ("router", "backbone"), "ties")]:
            model = upcycle(anchor, states, merge, TIES_KEEP if merge == "ties" else 1.0)
            model, history, best = finetune(model, train, validation, settings["epochs"], settings["batch_size"], settings["lr"],
                                            settings["patience"], seed, parts, AUX_WEIGHT)
            rows[name] = mixture_row(model, anchor, partitions, {"best_epoch": best})
    return rows


def summarize(results):
    stats = lambda values: {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    seeds = list(results["seeds"].values())
    out = {}
    for name in seeds[0]:
        runs = [s[name] for s in seeds]
        out[name] = {key: stats([r["test"][key] for r in runs]) for key in (*FAMILIES, "mean")}
        out[name]["validation_mean"] = stats([r["validation"]["mean"] for r in runs])
        if "routing" in runs[0]:
            # Share of each family's test tokens whose first choice is its own expert, averaged over blocks.
            out[name]["own_expert_share"] = {f: stats([statistics.fmean(layer[i] for layer in r["routing"][f]["first_choice"]) for r in runs])
                                             for i, f in enumerate(FAMILIES)}
        for key in ("parameters", "active_parameters"):
            if key in runs[0]:
                out[name][key] = runs[0][key]
    return out


def table(summary):
    fmt = lambda s: f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})" if s["max"] - s["min"] >= 0.005 else f"{s['mean']:.2f}"
    lines = ["| | Voyager | Off-source | Rosetta | Mean | Validation mean |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in (*FAMILIES, "mean", "validation_mean")) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=50)
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
