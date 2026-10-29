"""A shared anchor for merging volunteer runs (issue #20, milestone 1): a pinned codebook and a
deterministic starting checkpoint θ₀.

Two runs can be merged by parameter arithmetic only if a token means the same frames in both,
and both started from the same weights. Then each run's change, its task vector τ = θ − θ₀, is
in the same coordinates as every other run's. An anchor fixes both:

- the codebook: k-means with K tokens, fitted once on the training scans of a declared corpus,
  and pinned twice: by the SHA-256 of its state (the digest train.py reports for any tokenizer),
  and by the SHA-256 of its whole record, including the slices it was fitted on. A
  rebuild matches it to float32 rounding, with the same tokens, but not always bit for bit:
  numpy's SIMD kernels (AVX-512 or not) can move a centroid by one rounding;
- θ₀: SignalLM weights drawn from a PCG64 stream with uniform initializers only, so it rebuilds
  bit-exactly anywhere. It's committed as raw little-endian float32, decoded in the model's own
  parameter order: its header must give exactly that layout.

Both hashes are also pinned here, so an edited file is refused, not trusted.

    python -m tracks.llm.anchor build --anchor v1            # rebuild, and compare with the committed files
    python -m tracks.llm.anchor build --anchor v1 --write    # (re)write them
    python -m tracks.llm.anchor delta runs/w1/real.pt --output runs/w1/tau.npz
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Tokenizer, digest, load_slices, split_records, unique_records
from .model import SignalLM, validate_config

ROOT = Path(__file__).resolve().parents[2]
ANCHOR_DIR = Path(__file__).resolve().parent / "anchors"

# Each anchor is a fixed definition: changing one makes a new version, never an edit.
ANCHORS = {
    "v1": {
        "id": "signal-lm-anchor/v1",
        # The 49-frame slices of the 6-second product: one 2020 session, three scans per family.
        "manifest": "tracks/llm/data/slices/index.json",
        "product": "6_sec_integration",
        "families": ("voyager", "offsource"),
        "codebook_seed": 0,
        "theta_seed": 0,
        # Small enough that a task vector compressed for a result card fits its 16 KB limit
        # (milestone 2): 26,496 parameters, 63% of them in the feed-forward blocks.
        "model": dict(vocab_size=16, context=15, width=32, layers=2, heads=4, dropout=0.1, position="rope"),
        "codebook_sha256": "19ce2a0dcbb116854f6cebc3388a0d2e9c5a37e403688dd90af00a02df297aec",
        "codebook_record_sha256": "22bf3d37b1ab89698bb64bfc481a7b1617bc2b6e29dbcba0e4d5fbe26fa05b00",
        "theta0_sha256": "9f3093e53322d17a892b5f0576b7c173084fc68fd4f3f2c43594dbd2cecc7287",
    },
}


def codebook_digest(state):
    """The tokenizer digest train.py reports, so any run's tokenizer compares with an anchor's."""
    return digest(json.dumps(state, sort_keys=True).encode())


def codebook_record_digest(codebook):
    """The digest of a codebook's whole record but its own tokenizer digest: format, anchor, the
    slices it was fitted on (which the holdout guard relies on) and the tokenizer itself."""
    return digest(json.dumps({k: v for k, v in codebook.items() if k != "sha256"}, sort_keys=True).encode())


def fit_records(spec, records=None):
    """The slices an anchor's codebook is fitted on: the training split of each of its
    families, under the same holdout train.py uses, so validation and test never shape it."""
    records = load_slices(ROOT / spec["manifest"]) if records is None else records
    fit = []
    for family in spec["families"]:
        partitions, _ = split_records(records, family, allow_within_session=True)
        fit += partitions["train"]
    return unique_records(fit)


def build_codebook(spec, records=None):
    fit = fit_records(spec, records)
    if any(r["entry"].get("source", {}).get("product") != spec["product"] for r in fit):
        raise ValueError(f"The codebook's slices must all come from the {spec['product']} product")
    tokenizer = Tokenizer.fit([r["frames"] for r in fit], spec["model"]["vocab_size"], spec["codebook_seed"])
    state = tokenizer.state()
    return {
        "format": "eiathome-signal-lm-codebook", "version": 1, "anchor": spec["id"],
        "fit": {
            "manifest": spec["manifest"], "product": spec["product"], "families": list(spec["families"]),
            "split": "train", "seed": spec["codebook_seed"], "frames": sum(len(r["frames"]) for r in fit),
            "slices": [{"id": r["entry"]["id"], "family": r["entry"]["family"], "sha256": r["sha256"],
                        "frame_sha256": r["frame_hash"]} for r in fit],
        },
        "tokenizer": state,
        "sha256": codebook_digest(state),
    }


# A rebuilt codebook agrees with the committed one when every centroid, the mean and the scale
# are within this (float32 rounding, accumulated over k-means) and every frame gets the same token.
CODEBOOK_TOLERANCE = 1e-5


def compare_codebook(state, anchor, frames):
    """How a rebuilt codebook's state compares with the committed, pinned one, on these frames."""
    rebuilt = Tokenizer(**state)
    committed = anchor.tokenizer
    difference = max(float(np.abs(rebuilt.centroids - committed.centroids).max()),
                     abs(rebuilt.mean - committed.mean), abs(rebuilt.scale - committed.scale))
    same_tokens = bool(np.array_equal(rebuilt.encode(frames), committed.encode(frames)))
    return {"exact": codebook_digest(state) == anchor.spec["codebook_sha256"], "max_difference": difference,
            "same_tokens": same_tokens, "agrees": difference <= CODEBOOK_TOLERANCE and same_tokens}


def build_theta0(spec):
    """θ₀ as {name: float32 array}, in the model's parameter order. Every random draw is a PCG64
    uniform double scaled by an exactly rounded bound, so the bytes are the same on any platform
    (normal draws would depend on the maths library). The initializers follow PyTorch's
    defaults, with uniform in place of normal: embeddings uniform(−√3, √3), unit variance like
    N(0, 1); linear layers uniform(±1/√fan_in); attention input projections Xavier uniform with
    zero bias, and output projections zero bias, as in RotaryBlock; layer norms one and zero."""
    config = spec["model"]
    if config["position"] != "rope":
        raise ValueError("Anchors use RotaryBlock (position 'rope'), whose parameters this initializes")
    validate_config(**config)
    model = SignalLM(**config)
    owner = {}
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            owner[f"{module_name}.{param_name}" if module_name else param_name] = (module_name, module, param_name)
    rng = np.random.Generator(np.random.PCG64(spec["theta_seed"]))
    uniform = lambda shape, bound: ((rng.random(shape) * 2.0 - 1.0) * bound).astype(np.float32)

    tensors = OrderedDict()
    for name, value in model.state_dict().items():
        shape = tuple(value.shape)
        module_name, module, param = owner[name]
        if isinstance(module, nn.Embedding):
            tensors[name] = uniform(shape, np.sqrt(3.0))
        elif isinstance(module, nn.LayerNorm):
            tensors[name] = np.ones(shape, np.float32) if param == "weight" else np.zeros(shape, np.float32)
        elif isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            if module_name.endswith(".qkv") and param == "weight":
                tensors[name] = uniform(shape, np.sqrt(6.0 / (fan_in + fan_out)))
            elif module_name.endswith((".qkv", ".out")) and param == "bias":
                tensors[name] = np.zeros(shape, np.float32)
            else:
                tensors[name] = uniform(shape, 1.0 / np.sqrt(fan_in))
        else:
            raise ValueError(f"No anchor initializer for {name} ({type(module).__name__})")
    return tensors


def canonical_layout(config):
    """Where each parameter sits in θ₀'s bytes: the model's own order, contiguous."""
    offset, layout = 0, []
    for name, value in SignalLM(**config).state_dict().items():
        count = int(value.numel())
        layout.append({"name": name, "shape": list(value.shape), "offset": offset, "count": count})
        offset += count
    return layout, offset


def theta0_files(spec, tensors):
    """The committed form of θ₀: raw little-endian float32 in parameter order, and a header."""
    layout, parameters = canonical_layout(spec["model"])
    if [(t["name"], tuple(t["shape"])) for t in layout] != [(k, v.shape) for k, v in tensors.items()]:
        raise ValueError("θ₀'s tensors aren't in the model's parameter order")
    data = b"".join(np.ascontiguousarray(t, dtype="<f4").tobytes() for t in tensors.values())
    header = {
        "format": "eiathome-signal-lm-anchor", "version": 1, "anchor": spec["id"], "model": spec["model"],
        "seed": spec["theta_seed"], "dtype": "float32-le", "parameters": parameters, "tensors": layout,
        "init": "PCG64 uniform: embeddings ±√3; linear ±1/√fan_in; qkv Xavier, zero bias; out zero bias; layer norms 1, 0",
        "sha256": digest(data),
    }
    return header, data


@dataclass
class Anchor:
    name: str
    spec: dict
    tokenizer: Tokenizer
    theta0: OrderedDict  # name -> float32 torch tensor
    codebook: dict

    def identity(self):
        return {"anchor": self.spec["id"], "codebook_sha256": self.spec["codebook_sha256"],
                "codebook_record_sha256": self.spec["codebook_record_sha256"], "theta0_sha256": self.spec["theta0_sha256"]}

    def model(self):
        model = SignalLM(**self.spec["model"])
        model.load_state_dict(self.theta0)
        return model

    def check_data(self, records, partitions):
        """Refuses data the anchor doesn't fit: slices of another product, and validation or test
        frames the codebook was fitted on. Returns how many control-family frames the codebook
        saw (other families are scored as a control, never trained on)."""
        products = {r["entry"].get("source", {}).get("product") for r in records}
        if products != {self.spec["product"]}:
            raise ValueError(f"Anchor {self.name} is defined on the {self.spec['product']} product; this manifest's slices "
                             f"are {', '.join(sorted(str(p) for p in products))}. Its codebook would tokenize them meaninglessly.")
        fitted = {s["frame_sha256"] for s in self.codebook["fit"]["slices"]}
        held_out = [r for name in ("validation", "test") for r in partitions[name] if r["frame_hash"] in fitted]
        if held_out:
            raise ValueError(f"Anchor {self.name}'s codebook was fitted on frames this run holds out "
                             f"({', '.join(r['entry']['id'] for r in held_out)}), so its scores wouldn't be held out")
        return sum(len(r["frames"]) for r in records if r["frame_hash"] in fitted)


def load_anchor(name="v1", directory=None):
    """The committed anchor, verified against the hashes pinned in ANCHORS."""
    if name not in ANCHORS:
        raise ValueError(f"Unknown anchor {name!r}; known: {', '.join(ANCHORS)}")
    spec = ANCHORS[name]
    folder = Path(directory) if directory is not None else ANCHOR_DIR / name
    codebook = json.loads((folder / "codebook.json").read_text())
    if codebook.get("format") != "eiathome-signal-lm-codebook" or codebook.get("anchor") != spec["id"]:
        raise ValueError(f"{folder / 'codebook.json'} isn't anchor {name}'s codebook")
    if not codebook_digest(codebook["tokenizer"]) == codebook["sha256"] == spec["codebook_sha256"]:
        raise ValueError(f"Anchor {name}'s codebook doesn't match its pinned SHA-256")
    # The record's digest covers the slices it was fitted on, which the holdout guard trusts.
    if codebook_record_digest(codebook) != spec["codebook_record_sha256"]:
        raise ValueError(f"Anchor {name}'s codebook record (its fitted slices) doesn't match its pinned SHA-256")
    header = json.loads((folder / "theta0.json").read_text())
    data = (folder / "theta0.bin").read_bytes()
    if header.get("format") != "eiathome-signal-lm-anchor" or header.get("version") != 1 or header.get("anchor") != spec["id"] or header.get("model") != spec["model"]:
        raise ValueError(f"{folder / 'theta0.json'} isn't anchor {name}'s θ₀")
    if not digest(data) == header["sha256"] == spec["theta0_sha256"]:
        raise ValueError(f"Anchor {name}'s θ₀ doesn't match its pinned SHA-256")
    # The bytes are decoded in the model's own order: a header can't move them, only describe them.
    layout, parameters = canonical_layout(spec["model"])
    values = np.frombuffer(data, dtype="<f4")
    if header.get("dtype") != "float32-le" or header.get("tensors") != layout or header.get("parameters") != parameters or len(values) != parameters:
        raise ValueError(f"Anchor {name}'s θ₀ header doesn't give the model's layout")
    theta0 = OrderedDict((t["name"], torch.from_numpy(values[t["offset"]:t["offset"] + t["count"]].reshape(t["shape"]).copy()))
                         for t in layout)
    return Anchor(name, spec, Tokenizer(**codebook["tokenizer"]), theta0, codebook)


def task_vector(state, anchor):
    """τ = θ − θ₀ for a state dict of the anchor's architecture, in float64 and parameter order."""
    if list(state) != list(anchor.theta0) or any(tuple(state[k].shape) != tuple(v.shape) for k, v in anchor.theta0.items()):
        raise ValueError("The weights don't have the anchor's parameters, so they can't be compared with θ₀")
    return OrderedDict((k, state[k].detach().cpu().double() - v.double()) for k, v in anchor.theta0.items())


def checkpoint_task_vector(path, anchor):
    """τ for a saved checkpoint, refusing one trained from another anchor or tokenized otherwise."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eiathome-signal-lm" or checkpoint.get("version") != 1:
        raise ValueError("Not a supported signal-LM checkpoint")
    if checkpoint.get("anchor") != anchor.identity():
        raise ValueError(f"The checkpoint wasn't trained from anchor {anchor.name} (train with --anchor {anchor.name})")
    if codebook_digest(checkpoint["tokenizer"]) != anchor.spec["codebook_sha256"]:
        raise ValueError(f"The checkpoint's tokenizer isn't anchor {anchor.name}'s codebook")
    return task_vector(checkpoint["model"], anchor), checkpoint


def flatten(vector):
    """One 1-D array in parameter order: the coordinates merging works in."""
    return np.concatenate([v.reshape(-1).numpy() for v in vector.values()])


def apply(anchor, vector, scale=1.0):
    """θ₀ + scale·τ, as a float32 state dict for the anchor's model."""
    return OrderedDict((k, (v.double() + scale * vector[k].double()).float()) for k, v in anchor.theta0.items())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "delta"])
    p.add_argument("checkpoint", nargs="?", type=Path, help="delta: a checkpoint trained with --anchor")
    p.add_argument("--anchor", default="v1", choices=list(ANCHORS))
    p.add_argument("--write", action="store_true", help="build: write the files instead of checking them")
    p.add_argument("--output", type=Path, help="delta: a new .npz file for τ (float32, by parameter name)")
    args = p.parse_args(argv)
    spec = ANCHORS[args.anchor]
    try:
        if args.command == "build":
            codebook = build_codebook(spec)
            header, data = theta0_files(spec, build_theta0(spec))
            folder = ANCHOR_DIR / args.anchor
            if args.write:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "codebook.json").write_text(json.dumps(codebook, indent=1) + "\n")
                (folder / "theta0.json").write_text(json.dumps(header, indent=1) + "\n")
                (folder / "theta0.bin").write_bytes(data)
                print(f"wrote {folder}: codebook {codebook['sha256']} (record {codebook_record_digest(codebook)}), "
                      f"θ₀ {header['sha256']} ({header['parameters']:,} parameters)")
                return
            committed = load_anchor(args.anchor)
            if data != (folder / "theta0.bin").read_bytes():
                raise ValueError("The rebuilt θ₀ differs from the committed one")
            if codebook["fit"] != committed.codebook["fit"]:
                raise ValueError("The rebuilt codebook was fitted on different slices from the committed one")
            frames = np.concatenate([r["frames"] for r in load_slices(ROOT / spec["manifest"])])
            check = compare_codebook(codebook["tokenizer"], committed, frames)
            if not check["agrees"]:
                raise ValueError(f"The rebuilt codebook differs from the committed one (by up to {check['max_difference']:.3g}"
                                 f"{'' if check['same_tokens'] else ', with different tokens'})")
            how = "exactly" if check["exact"] else f"to within {check['max_difference']:.2g}, with the same tokens"
            print(f"anchor {args.anchor} rebuilds: θ₀ exactly ({header['sha256'][:16]}), the codebook {how}. "
                  f"Runs use the committed codebook ({committed.spec['codebook_sha256'][:16]}).")
            return
        if args.checkpoint is None or args.output is None:
            p.error("delta needs a checkpoint and --output")
        if args.output.exists():
            p.error(f"{args.output} already exists")
        anchor = load_anchor(args.anchor)
        tau, _ = checkpoint_task_vector(args.checkpoint, anchor)
        flat = flatten(tau)
        np.savez(args.output, __anchor__=np.array(json.dumps(anchor.identity())),
                 **{name: v.float().numpy() for name, v in tau.items()})
        print(f"τ: {len(flat):,} parameters, L2 norm {np.linalg.norm(flat):.4f}, max |τ| {np.abs(flat).max():.4f}")
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
