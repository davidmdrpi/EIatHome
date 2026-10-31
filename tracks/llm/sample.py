"""Reload a trained checkpoint and continue a signal slice without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_slices, tokenizer_from_state
from .model import SignalLM, generate
from .runtime import atomic_output, require_new_output, training_runtime


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eiathome-signal-lm" or checkpoint.get("version") != 1:
        raise ValueError("Not a supported signal-LM checkpoint")
    model = SignalLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer_from_state(checkpoint["tokenizer"]), checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--slice-id", required=True)
    p.add_argument("--prompt-frames", type=int, default=8)
    p.add_argument("--tokens", type=int, default=None,
                   help="Default: fill the trained context. Only RoPE checkpoints continue past it, sliding a window of at most context tokens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        require_new_output(args.output)
        with training_runtime(1, "cpu"):
            model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
            record = load_slices(args.manifest, slice_id=args.slice_id)[0]
            if not 1 <= args.prompt_frames <= min(len(record["frames"]), model.config["context"]):
                raise ValueError("Prompt length must fit the slice and checkpoint context")
            prompt = tokenizer.encode(record["frames"][:args.prompt_frames]).tolist()
            tokens = generate(model, prompt, args.tokens, args.seed, args.temperature)
            with atomic_output(args.output) as output:
                np.save(output / "frames.npy", tokenizer.decode(tokens), allow_pickle=False)
                (output / "sample.json").write_text(json.dumps({
                    "checkpoint_dataset": checkpoint["dataset_sha256"], "variant": checkpoint["variant"],
                    "source": record["entry"], "prompt": prompt, "generated": tokens,
                    "seed": args.seed, "temperature": args.temperature,
                    "position": model.config["position"],
                    "note": ("Codebook-centroid continuation; past the trained context each step conditions on the last "
                             "context tokens (RoPE). Qualitative: not a forecast or a decoded message."
                             if len(prompt) + len(tokens) > model.config["context"] + 1 else
                             "Codebook-centroid continuation within the trained context; not a decoded message."),
                }, indent=2) + "\n")
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
