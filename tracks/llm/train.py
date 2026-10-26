"""python -m tracks.llm.train --help"""

import argparse
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from .data import SPLITS, Tokenizer, digest, load_slices, markov_probabilities, split_records, split_sizes, surrogate, unique_records
from .model import evaluate, fit, generate, validate_config
from .runtime import atomic_output, require_new_output, training_runtime

ROOT = Path(__file__).resolve().parents[2]


def window_starts(model, sequences, batch_size):
    """Bits/token of full-length windows (context + 1 frames) by where they start in the slice.
    Every window has the same length and the same contexts within it; only its start differs.
    Start 0 is the scan's first frame; mid-scan pools all later starts. A frame is scored once
    per window that contains it, so these are per-window scores, not one sequence likelihood.
    None if no slice has room for a full window."""
    context = model.config["context"]
    by_start = []
    for start in range(max(len(s) for s in sequences) - context):
        chunks = [s[start:start + context + 1] for s in sequences if len(s) >= start + context + 1]
        score = evaluate(model, chunks, batch_size)
        by_start.append({"start": start, "windows": len(chunks), "bits_per_token": score["bits_per_token"]})
    if not by_start:
        return None
    mid = by_start[1:]
    windows = sum(s["windows"] for s in mid)
    return {
        "scan_start_bits": by_start[0]["bits_per_token"],
        # Every window has `context` targets, so pooling weights by window count.
        "mid_scan_bits": sum(s["bits_per_token"] * s["windows"] for s in mid) / windows if windows else None,
        "mid_scan_windows": windows,
        "by_start": by_start,
    }


def baseline_scores(train, test, k):
    unigram, transitions = markov_probabilities(train, k)
    targets = sum(len(s) - 1 for s in test)
    return {
        "unigram_bits": float(sum(-np.log2(unigram[s[1:]]).sum() for s in test) / targets),
        "markov1_bits": float(sum(-np.log2(transitions[s[:-1], s[1:]]).sum() for s in test) / targets),
        "uniform_bits": math.log2(k),
    }


def run(args):
    require_new_output(args.output)
    config = dict(vocab_size=args.vocab_size, context=args.context, width=args.width,
                  layers=args.layers, heads=args.heads, dropout=args.dropout, position=args.position)
    validate_config(**config)
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1:
        raise ValueError("Epochs, batch size, patience and threads must be positive")
    if args.train_stride is not None and not 1 <= args.train_stride <= args.context:
        raise ValueError("Training stride must be between 1 and the context")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    with training_runtime(args.threads, args.device), atomic_output(args.output) as output:
        return _run(args, config, output)


def _run(args, config, output):
    records = load_slices(args.manifest)
    partitions, independent = split_records(records, args.family, args.allow_within_session)
    if args.context > max(len(r["frames"]) - 1 for r in partitions["train"]):
        raise ValueError("Context exceeds the longest training sequence's next-token span; reduce --context")
    tokenizer = Tokenizer.fit([r["frames"] for r in partitions["train"]], args.vocab_size, args.seed)
    streams = {name: [tokenizer.encode(r["frames"]) for r in partitions[name]] for name in SPLITS}
    probabilities = markov_probabilities(streams["train"], args.vocab_size)
    provenance = {name: [r["entry"] for r in partitions[name]] for name in SPLITS}
    identity = [{"sha256": r["sha256"], "entry": r["entry"]} for r in records]
    dataset_sha = digest(json.dumps(identity, sort_keys=True).encode())
    code_sha = digest(b"".join((Path(__file__).parent / name).read_bytes()
                               for name in ("data.py", "model.py", "train.py", "runtime.py")))
    report = {
        "format": "eiathome-signal-lm", "version": 1,
        "evaluation": "independent_sessions" if independent else "within_session_scan_holdout",
        "independent_sessions": independent, "family": args.family,
        "limitation": ("Small-sample next-frame prediction; no inference of language or intelligence."
                       if independent else "Smoke test only: scans share an observing session. "
                       "Not evidence of generalization across sessions or of language/intelligence."),
        "dataset_sha256": dataset_sha, "code_sha256": code_sha,
        "splits": provenance, "split_unit": "session" if independent else "scan",
        "split_sizes": split_sizes(records, partitions, independent),
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
        "model": config, "variants": {},
        "tokenizer": {"fit_split": "train", "training_frames": sum(len(s) for s in streams["train"]),
                      "sha256": digest(json.dumps(tokenizer.state(), sort_keys=True).encode())},
    }
    print(report["evaluation"] + ": " + report["limitation"], flush=True)
    for variant in ("real", "shuffled", "markov"):
        # Separate deterministic randomness for each split; never shuffle across partitions.
        data = {name: surrogate(streams[name], variant, args.seed + 1000 + i, probabilities)
                for i, name in enumerate(SPLITS)}
        model, history, best_epoch = fit(data["train"], data["validation"], config, args.epochs,
                                         args.batch_size, args.lr, args.patience, args.seed, args.device,
                                         args.train_stride)
        # Test is consulted only after model selection, once per final variant.
        score = evaluate(model, data["test"], args.batch_size)
        score.update(baseline_scores(data["train"], data["test"], args.vocab_size))
        result = {"test": score, "test_window_starts": window_starts(model, data["test"], args.batch_size),
                  "best_epoch": best_epoch, "history": history}
        if variant == "real":
            controls = unique_records(r for r in records if r["entry"]["family"] != args.family)
            if controls:
                control_tokens = [tokenizer.encode(r["frames"]) for r in controls]
                result["other_family_control"] = {
                    "family": controls[0]["entry"]["family"],
                    "interpretation": "Untrained family, possibly from the same observing sessions; not an independent-session test.",
                    "slices": [r["entry"] for r in controls],
                    **evaluate(model, control_tokens, args.batch_size),
                    **baseline_scores(data["train"], control_tokens, args.vocab_size),
                    "window_starts": window_starts(model, control_tokens, args.batch_size),
                }
            # Qualitative continuation from a training prompt; never represented as a forecast.
            prompt = data["train"][0][:min(8, args.context)].tolist()
            if args.position == "rope":
                # As long as the prompt's slice, sliding a window of at most `context` tokens.
                count = max(args.context + 1 - len(prompt), len(data["train"][0]) - len(prompt))
                limit = ("RoPE: continues to the slice's length; past the trained context each step "
                         "conditions on the last context tokens. Qualitative, not a forecast.")
            else:
                count = None
                limit = "Stops at the trained context without sliding positional embeddings."
            tokens = generate(model, prompt, count, seed=args.seed)
            result["sample"] = {"prompt_source": "train", "prompt": prompt, "generated": tokens, "limit": limit}
            np.save(output / "generated_frames.npy", tokenizer.decode(tokens), allow_pickle=False)
        report["variants"][variant] = result
        report["parameters"] = sum(p.numel() for p in model.parameters())
        checkpoint = {
            "format": report["format"], "version": 1, "variant": variant, "config": config,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "tokenizer": tokenizer.state(), "dataset_sha256": dataset_sha,
            "code_sha256": code_sha, "splits": provenance,
            "evaluation": report["evaluation"], "best_epoch": best_epoch,
        }
        torch.save(checkpoint, output / f"{variant}.pt")
        starts = result["test_window_starts"]
        mid = "" if not starts or starts["mid_scan_bits"] is None else (
            f"; full windows from the scan start {starts['scan_start_bits']:.4f}, from mid-scan {starts['mid_scan_bits']:.4f}")
        print(f"{variant}: {score['bits_per_token']:.4f} bits/token{mid}, best epoch {best_epoch}", flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def parser():
    p = argparse.ArgumentParser(description="Train a causal signal-token transformer and matched surrogate controls.")
    p.add_argument("--manifest", type=Path, default=ROOT / "web/data/slices/index.json")
    p.add_argument("--output", type=Path, default=ROOT / "runs/signal-lm")
    p.add_argument("--family", choices=("voyager", "offsource"), default="voyager")
    p.add_argument("--allow-within-session", action="store_true", help="Explicitly allow a labelled whole-scan smoke test")
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--context", type=int, default=15)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--position", choices=("absolute", "rope"), default="absolute",
                   help="learned absolute position embeddings, or rotary (RoPE), which depend only on relative offsets")
    p.add_argument("--train-stride", type=int, default=None,
                   help="frames between training-window starts (default: the context, no overlap); 1 trains on every start")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
