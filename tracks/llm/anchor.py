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

from .data import PeakTokenizer, Tokenizer, digest, load_slices, split_records, tokenizer_from_state, unique_records
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
    # Milestone 3: v1's slices plus the Rosetta control (tracks/llm/synthetic.py), so one
    # codebook covers the families a mixture of experts upcycles. 32 tokens: the smallest of
    # 16, 24, 32 and 48 for which a Rosetta training frame's token gives its FSK symbol for at
    # least 95% of frames, and neither real family's training frames use fewer tokens than
    # under v1 (Voyager 12, off-source 6). The second condition was added after seeing that at
    # 16 tokens every off-source frame gets the same one. 27,520 parameters.
    "v2": {
        "id": "signal-lm-anchor/v2",
        "manifest": "tracks/llm/data/slices/index-rosetta.json",
        "product": ["6_sec_integration", "synthetic"],
        "families": ("voyager", "offsource", "rosetta"),
        "codebook_seed": 0,
        "theta_seed": 0,
        "model": dict(vocab_size=32, context=15, width=32, layers=2, heads=4, dropout=0.1, position="rope"),
        "codebook_sha256": "57465a05c3e9997a8d5a3510c12e1b1c76d5d18ab80d7516c9ee3a3a93b3298d",
        "codebook_record_sha256": "7b041732635e4a12e7d6a3c760d0f5698c94219b4ec25063552a11ed42a6101b",
        "theta0_sha256": "ac600b9458ab5f7d329d0ba0732d142ddc6591262384c8a5974f41f43b4880d0",
    },
    # Milestone 3b: Voyager, a longer Rosetta and an RFI sweep (index-moe-v2.json, made by
    # synthetic.py), tokenized by each frame's brightest channel instead of k-means. k-means
    # spent spare centroids splitting Rosetta's symbols by their noise (about 1 bit per token,
    # at any flux, K or initialization tried); the peak tokenizer gives each FSK symbol exactly
    # one token. Threshold 6 sigma, 2-channel bins: 33 tokens, the only setting tried. It maps
    # off-source noise to one token ("no line"), so off-source isn't one of v3's families.
    "v3": {
        "id": "signal-lm-anchor/v3",
        "manifest": "tracks/llm/data/slices/index-moe-v2.json",
        "product": ["6_sec_integration", "synthetic"],
        "families": ("voyager", "rosetta", "sweep"),
        "tokenizer": {"kind": "peak", "threshold": 6.0, "width": 2, "channels": 64},
        "theta_seed": 0,
        "model": dict(vocab_size=33, context=15, width=32, layers=2, heads=4, dropout=0.1, position="rope"),
        "codebook_sha256": "17d422eda0950f22b4ed96afc06f8770be96a56869a6d9a242269f85d0748215",
        "codebook_record_sha256": "51ce459876f09e068e7bd6ba1777236d8c792fa2f1eaa03c1f1ccff5b2857a65",
        "theta0_sha256": "83124fecefc322b17e9f1326ecafad80f15346cfcb5e0d10d5dd6f840039fa15",
    },
}


def products(spec):
    """The data products an anchor's codebook is defined on."""
    return set(spec["product"]) if isinstance(spec["product"], list) else {spec["product"]}


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
    if "tokenizer" in spec:  # a fixed rule (v3's peak tokenizer): nothing is fitted
        state = PeakTokenizer(**spec["tokenizer"]).state()
        if state["kind"] != "peak" or PeakTokenizer(**state).vocab_size != spec["model"]["vocab_size"]:
            raise ValueError("The tokenizer's vocabulary must be the model's")
        return {
            "format": "eiathome-signal-lm-codebook", "version": 1, "anchor": spec["id"],
            "fit": {"manifest": spec["manifest"], "product": spec["product"], "families": list(spec["families"]),
                    "split": None, "seed": None, "frames": 0, "slices": []},
            "tokenizer": state,
            "sha256": codebook_digest(state),
        }
    fit = fit_records(spec, records)
    if any(r["entry"].get("source", {}).get("product") not in products(spec) for r in fit):
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
    rebuilt = tokenizer_from_state(state)
    committed = anchor.tokenizer
    if isinstance(committed, PeakTokenizer) or isinstance(rebuilt, PeakTokenizer):
        same = state == committed.state()  # a fixed rule rebuilds exactly or not at all
        return {"exact": same, "max_difference": 0.0 if same else float("inf"), "same_tokens": same, "agrees": same}
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
        seen = {r["entry"].get("source", {}).get("product") for r in records}
        if not seen <= products(self.spec):
            raise ValueError(f"Anchor {self.name} is defined on the {self.spec['product']} product; this manifest's slices "
                             f"are {', '.join(sorted(str(p) for p in seen))}. Its codebook would tokenize them meaninglessly.")
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
    return Anchor(name, spec, tokenizer_from_state(codebook["tokenizer"]), theta0, codebook)


def difference(state, base):
    """state − base, in float64 and parameter order, for two state dicts of the same model."""
    if list(state) != list(base) or any(tuple(state[k].shape) != tuple(v.shape) for k, v in base.items()):
        raise ValueError("The weights don't have the anchor's parameters, so they can't be compared with θ₀")
    return OrderedDict((k, state[k].detach().cpu().double() - v.detach().cpu().double()) for k, v in base.items())


def task_vector(state, anchor):
    """τ = θ − θ₀ for a state dict of the anchor's architecture, in float64 and parameter order."""
    return difference(state, anchor.theta0)


def weights_digest(state):
    """The SHA-256 of weights stored as θ₀ is: float32 little-endian, in parameter order. For θ₀
    itself, it's the anchor's pinned digest."""
    return digest(b"".join(np.ascontiguousarray(v.detach().cpu().numpy(), dtype="<f4").tobytes() for v in state.values()))


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
