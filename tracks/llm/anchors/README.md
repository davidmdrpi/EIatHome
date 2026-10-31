# Shared anchors

An anchor is what every run must share before runs can be merged
([issue #20](https://github.com/davidmdrpi/EIatHome/issues/20)): a pinned codebook, so a token
means the same frames in every run, and a starting checkpoint θ₀, so every run's change
τ = θ − θ₀ is in the same coordinates. Each folder is one anchor, defined in `ANCHORS` in
[`anchor.py`](../anchor.py), which also pins the files' hashes. A changed definition is a new
version, never an edit.

| File | Contents |
|---|---|
| `codebook.json` | The k-means tokenizer (mean, scale, K centroids of 64 channels), the slices it was fitted on, and the SHA-256 of its state (the digest `train.py` reports for any tokenizer). Two digests are pinned: the state's, and the whole record's, which covers the fitted slices the holdout guard relies on |
| `theta0.bin` | θ₀: raw little-endian float32, the model's parameters in order. Its SHA-256 is pinned |
| `theta0.json` | θ₀'s model settings, seed, each tensor's name, shape and offset. The bytes are decoded in the model's own order, so the header must give exactly that layout; it can't move a tensor |

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

## v2

- **Data:** v1's slices plus the Rosetta control, synthetic 16-tone FSK text
  (`tracks/llm/data/slices/index-rosetta.json`, made by [`synthetic.py`](../synthetic.py)), for
  the mixture of experts (milestone 3).
- **Codebook:** K = 32, fitted on the training scan of each of the three families: 441 frames.
  - **Why 32:** it's the smallest of 16, 24, 32 and 48 for which two conditions hold:
    - a Rosetta training frame's token gives its symbol for at least 95% of frames (96%);
    - neither real family's training frames use fewer tokens than under v1: Voyager uses 24
      (12 under v1), off-source 6 (6).
  - **Added after looking:** the second condition, after seeing that at 16 tokens every
    off-source frame gets the same one.
  - **A cost:** k-means splits frequent symbols by their noise. Given its symbol, a Rosetta
    token still carries about 1 bit of noise.
- **Model:** v1's, with 32 tokens instead of 16: 27,520 parameters.
- **θ₀:** drawn as v1's is, from the same seed.

Check that the committed files rebuild, or write them for a new version:

```bash
python -m tracks.llm.anchor build --anchor v1   # or v2
python -m tracks.llm.anchor build --anchor v1 --write   # only when defining a new anchor
```

θ₀ rebuilds bit for bit anywhere. The codebook rebuilds to float32 rounding, with the same token
for every frame, but not always bit for bit: numpy's SIMD kernels differ by CPU. Without AVX-512,
for example, a centroid moves by up to 1.8 × 10⁻⁷, which changes its digest. So the check
compares the rebuilt codebook's values and tokens, not its digest. Runs always use the committed
codebook, which is checked against its pinned hash.
