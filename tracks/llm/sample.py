"""Reload a trained checkpoint and continue a signal slice without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Tokenizer, load_slices
from .model import SignalLM, generate


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eiathome-signal-lm" or checkpoint.get("version") != 1:
        raise ValueError("Not a supported signal-LM checkpoint")
    model = SignalLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, Tokenizer(**checkpoint["tokenizer"]), checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--slice-id", required=True)
    p.add_argument("--prompt-frames", type=int, default=8)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    matches = [r for r in load_slices(args.manifest) if r["entry"]["id"] == args.slice_id]
    if len(matches) != 1:
        p.error("--slice-id must match exactly one manifest entry")
    record = matches[0]
    if not 1 <= args.prompt_frames <= min(len(record["frames"]), model.config["context"]):
        p.error("Prompt length must fit the slice and checkpoint context")
    prompt = tokenizer.encode(record["frames"][:args.prompt_frames]).tolist()
    try:
        tokens = generate(model, prompt, args.tokens, args.seed, args.temperature)
    except ValueError as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "frames.npy", tokenizer.decode(tokens), allow_pickle=False)
    (args.output / "sample.json").write_text(json.dumps({
        "checkpoint_dataset": checkpoint["dataset_sha256"], "variant": checkpoint["variant"],
        "source": record["entry"], "prompt": prompt, "generated": tokens,
        "seed": args.seed, "temperature": args.temperature,
        "note": "Codebook-centroid continuation in noise-sigma units; not a decoded message.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
