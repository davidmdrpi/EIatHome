# SETI signal language-model training

Train a GPT-style, decoder-only transformer **from scratch on radio signal tokens**. This
extends Track B beyond the browser's GRU token model: configurable causal self-attention,
training-only tokenization, validation-selected checkpoints, test likelihood, surrogate
training runs, and checkpoint-based signal continuation. Runs locally on CPU or CUDA.

The vocabulary represents spectrogram frames, not words. The default model has about 0.8M
parameters, deliberately small for the current data. Width, layers, heads and vocabulary are
configurable for larger datasets; this is an LLM-style training pipeline, not a pretrained
text LLM or an alien-language translator. It does not fine-tune a text model on numeric strings.

## Run on the published data

From the repository root, in a Python 3.10+ environment:

```bash
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tracks/llm/requirements.txt
python -m tracks.llm.train --allow-within-session --output runs/voyager-smoke
python -m tracks.llm.train --family offsource --allow-within-session --output runs/offsource-smoke
```

For CUDA, install a PyTorch build appropriate for your GPU instead of the CPU wheel and pass
`--device cuda`. CPU is the default. Outputs must be new directories; previous runs are never
silently overwritten. Invalid settings are rejected before creating output directories. Each
run writes into a temporary sibling directory, renamed to the final output only after all
variants, checkpoints and the report succeed. Failed runs clean up their temporary output,
so the same output path can be retried. Library calls restore PyTorch settings and RNG state
on both success and failure. No account, remote model service or external data download is needed
after installation: these commands use the committed, verified slices.

**Longer windows.** [`data/slices`](data/slices/README.md) holds the same cadence cut from
Breakthrough Listen's 6-second product: 49 time steps per scan instead of 16, at the same
channel width. Train on it, with rotary position embeddings and a training window at every
start:

```bash
python -m tracks.llm.train --manifest tracks/llm/data/slices/index.json --allow-within-session \
  --position rope --train-stride 1 --output runs/voyager-6s-rope
```

**The current eighteen slices come from one session.** Without `--allow-within-session`,
training refuses to run until the chosen family spans at least **three independent sessions**
(one each for train, validation and test). The explicit smoke mode holds out whole scans,
keeps carrier/sidebands together, and reports `within_session_scan_holdout`. It cannot establish
generalization across observing sessions. With the Live Lab's 16-frame slices each split has
48 frames and 45 next-token targets; with the 49-frame slices, 147 frames and 144 targets.
More epochs do not make either dataset larger.

For three or more sessions, export a manifest using `tools/extract_slices.py`, then:

```bash
python -m tracks.llm.train --manifest data/more-slices/index.json --output runs/session-test
```

Groups are chronological, approximately 60/20/20, with at least one group per partition.
Validation and test each receive `max(1, round(group_count * 0.2))` whole groups; training gets
the remainder. For nine sessions this is 5/2/2. The report records actual group, slice, frame
and target counts in `split_sizes`, with `split_unit` distinguishing scans from sessions.
Scans connected by gaps of at most twelve hours belong to one session, including transitive
chains. Identical file bytes or identical cropped windows link groups even when timestamps
disagree. Copies within a split or the other-family control are counted once. All families participate in finding session
boundaries; only the chosen family trains the model. This conservative rule does not establish
independence of targets, observing hardware, or adjacent frequency bands.

## What the pipeline learns

1. Verify each slice's SHA-256, shape, finite samples and start time. Take one fixed central
   64-channel window, with no shifted copies, oversampling, or concatenation across slices.
2. Split observations **before fitting anything**. Apply signed `log1p` to signal amplitude;
   fit the scalar normalization and k-means centroids on training frames only. Each frame
   becomes the nearest centroid's integer token. Validation/test cannot alter the vocabulary.
3. Train three transformers with identical architecture, initialization and optimizer settings:
   real order, independently shuffled order within each slice, and Markov-1 sequences of the
   same lengths. The Markov generator is fitted exclusively to real training transitions, with
   fixed add-one smoothing; it never uses validation/test transitions. These are statistical
   controls, not alternative physical observations.
4. Minimize categorical next-token cross-entropy using an explicit causal attention mask,
   AdamW, gradient clipping, and validation early stopping. Chunk each slice independently.
   Training windows start every `--train-stride` frames (default: the context, so they don't
   overlap; 1 starts one at every frame). Validation and test always use non-overlapping
   chunks and count every next-token target once, including short tails. Padding is excluded
   from loss.
5. Reload the best validation weights and score test tokens. Report bits/token, perplexity,
   target count and training-fitted unigram/Markov-1 baselines. The real model is also scored
   on the other real family as a separately labelled control; it may share observing sessions.
6. Score full-length windows (context + 1 frames) by where they start in the slice:
   `test_window_starts` gives bits/token for each start, the scan-start window (start 0) and
   all later starts pooled (`mid_scan_bits`). The windows have the same length and the same
   contexts within them, so a gap between the two shows a model that depends on where a
   window starts. A frame is scored once per window containing it: these are per-window
   scores, not a second sequence likelihood. With 16-frame slices and context 15 there is
   only the scan-start window.

**Positions.** `--position absolute` (the default) learns one embedding per position: a model
trained on windows that start at a scan's first frame has only seen frame *p* at position *p*.
`--position rope` uses rotary position embeddings in a block that's otherwise PyTorch's
pre-norm encoder layer: attention then depends only on relative offsets. Neither extrapolates
past the trained context. RoPE is what makes sliding a window of at most `context` tokens
safe, and `--train-stride 1` is what shows the model content from mid-scan starts.

Default: 16 tokens, context 15, width 128, 4 layers, 4 heads, dropout 0.1, absolute positions,
training stride 15, 50 maximum epochs, patience 8, learning rate 0.0003, batch size 32, seed
42. Use `--help` for all options. The 16-frame slices cannot test dependencies longer than 15
steps. A context longer than the longest training sequence's next-token span is rejected: it
would leave positions untrained. Larger collections are currently loaded into memory;
this implementation is for extracted slice collections, not streaming multi-gigabyte HDF5.
With `--train-stride 1`, an epoch has about `context` times more training windows, so
early stopping happens after fewer epochs. Comparing strides therefore also changes the number
of optimizer updates and how often each target is trained on, not only where windows start.

## Outputs and inference

- `report.json`: configuration, source entries and hashes for every split, independent-session
  flag, actual split sizes, dataset/code fingerprints, package versions, loss histories, best epochs, metrics,
  control provenance and a qualitative sample.
- `real.pt`, `shuffled.pt`, `markov.pt`: selected model weights, complete model configuration,
  fitted tokenizer, split provenance, fingerprints and evaluation label. These are inference
  checkpoints; optimizer state for interrupted-run resume is not included.
- `generated_frames.npy`: codebook-centroid frames in approximate input noise-sigma units,
  starting from a training prompt. Absolute-position models stop at the trained context: with
  context 15 and an 8-frame prompt, 8 frames (positions 8 through 15). RoPE models continue to
  the prompt slice's length (41 frames for a 49-frame slice), sliding a window of at most
  `context` tokens. Either way it's a qualitative continuation, not a forecast.

Continue a particular signal with the saved tokenizer and model:

```bash
python -m tracks.llm.sample runs/voyager-smoke/real.pt \
  --manifest web/data/slices/index.json \
  --slice-id voyager1-59046-81310-carrier \
  --prompt-frames 8 --tokens 8 --seed 42 --output runs/continuation
```

This produces `sample.json` and `frames.npy`; it never retrains the tokenizer. Checkpoints load
with `weights_only=True`, and checkpoints saved before `--position` existed load as absolute.
Sampling verifies only the selected slice's bytes and shape; broken or unavailable unrelated
entries do not prevent inference. With `--tokens` omitted, generation fills the remaining
trained context. The prompt always starts at position zero, and every prediction uses at most
`context` input tokens. For absolute-position checkpoints, a longer `--tokens` is rejected
before generation: sliding would move mid-slice frames onto position alignments the model
never trained on. RoPE checkpoints accept any length: past the context, each step conditions
on the last `context` tokens. Long continuations are qualitative (`sample.json` says so); they
repeat what the model learned about short windows and aren't evidence of longer structure.
Generated tokens decode to centroid spectra, so fine spectral detail and phase are not recovered. Power-spectrogram prediction does not
decode Voyager telemetry or establish natural language. A predictable drifting carrier can
produce a strong score without linguistic structure.

## Validation and interpretation

The [recorded smoke results](SMOKE_RESULTS.md) include both published families, their
baselines, the exact environment, and dataset/code fingerprints. [Position results](POSITION_RESULTS.md) compare
absolute and rotary positions, with and without offset training windows, on the 49-frame slices:
on windows that start mid-scan, absolute positions trained on non-overlapping windows do worst,
and training on every start removes the difference.

```bash
python -m unittest discover -s tests -p 'test_signal_lm.py' -v
```

CI covers integrity checks, session/copy grouping, explicit smoke-mode consent, training-only
tokenization, split-local surrogates, causal attention in training and inference, padded loss,
learning a known next-token rule, checkpoint/generation round trips, and an end-to-end check
that mutating test data changes neither the fitted tokenizer nor selected model weights.
For positions: shifting a RoPE window's start leaves its attention unchanged; the RoPE block
equals PyTorch's encoder layer without the rotation; long RoPE continuations slide over the
last `context` tokens and stay in the vocabulary; absolute models still refuse to slide;
overlapping training windows leave test scoring at one count per target; and window-start
scores compare equal-length windows.
Behavioural tests create fixed synthetic manifests; only the integrity check reads the current
published collection, without assuming its size or number of sessions. Regressions also cover
context-bounded generation, tiny sampling temperatures, atomic output on late failure,
process-state restoration, split rounding, control de-duplication and sampling error messages.

For research, collect more independent sessions, repeat predeclared configurations across
seeds, and add the broader controls in [the experiment protocol](../../docs/experiments.md).
This command does not implement the protocol's full S0–S4 or Rosetta validation suite and
must not be used alone to claim linguistic content. Tokenizers fitted on different families
define different prediction problems; compare each model with its own baselines/surrogates,
not just raw bits/token across separately trained codebooks. CPU determinism is tested;
floating-point results need not match across PyTorch versions or devices.

Implementation references: PyTorch's [causal attention mask semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
and [restricted checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
