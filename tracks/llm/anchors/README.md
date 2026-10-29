# Shared anchors

An anchor is what every run must share before runs can be merged
([issue #20](https://github.com/davidmdrpi/EIatHome/issues/20)): a pinned codebook, so a token
means the same frames in every run, and a starting checkpoint θ₀, so every run's change
τ = θ − θ₀ is in the same coordinates. Each folder is one anchor, defined in `ANCHORS` in
[`anchor.py`](../anchor.py), which also pins both files' SHA-256. A changed definition is a new
version, never an edit.

| File | Contents |
|---|---|
| `codebook.json` | The k-means tokenizer (mean, scale, K centroids of 64 channels), the slices it was fitted on, and the SHA-256 of its state (the digest `train.py` reports for any tokenizer) |
| `theta0.bin` | θ₀: raw little-endian float32, the model's parameters in order |
| `theta0.json` | θ₀'s model settings, seed, each tensor's name, shape and offset, and the SHA-256 of `theta0.bin` |

## v1

- **Data:** the 49-frame slices of the 6-second product (`tracks/llm/data/slices`): one 2020
  session, three scans per family.
- **Codebook:** K = 16, fitted on the training split of both families (the first scan of each
  under the scan holdout `train.py` uses): 294 frames. No validation or test scan shaped it.
- **Model:** `SignalLM` with rotary positions, context 15, width 32, 2 layers, 4 heads, dropout
  0.1: 26,496 parameters, 63% of them in the feed-forward blocks. That's small enough that a
  task vector compressed for a result card fits its 16 KB limit (milestone 2).
- **θ₀:** PyTorch's default initializers, but drawn as PCG64 uniform doubles scaled by exactly
  rounded bounds (embeddings uniform ±√3, so unit variance, instead of normal). The bytes
  rebuild the same on any platform.

Check that the committed files rebuild, or write them for a new version:

```bash
python -m tracks.llm.anchor build --anchor v1
python -m tracks.llm.anchor build --anchor v1 --write   # only when defining a new anchor
```

θ₀ rebuilds bit for bit anywhere. The codebook rebuilds to float32 rounding, with the same token
for every frame, but not always bit for bit: numpy's SIMD kernels differ by CPU. Without AVX-512,
for example, a centroid moves by up to 1.8 × 10⁻⁷, which changes its digest. So the check
compares the rebuilt codebook's values and tokens, not its digest. Runs always use the committed
codebook, which is checked against its pinned hash.
