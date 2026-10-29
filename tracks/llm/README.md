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

**The Live Lab's slices come from two sessions.** Voyager 1 has one on-source scan from 2016
and three from 2020; the off-source controls are 2020's three scans. Without
`--allow-within-session`, training refuses to run until the chosen family spans at least
**three independent sessions** (one each for train, validation and test). The explicit smoke
mode holds out whole scans, keeps carrier/sidebands together, and reports
`within_session_scan_holdout`. It cannot establish generalization across observing sessions.
With the Live Lab's 16-frame slices, validation and test each get one 2020 scan (48 frames and
45 next-token targets); training gets the rest (Voyager: the 2016 scan and the first 2020 scan,
96 frames and 90 targets; off-source: 48 and 45). The 49-frame slices come from 2020 only, with
147 frames and 144 targets per split. More epochs do not make either dataset larger.
`load_slices` reads manifest versions 1 and 2; version 2 adds each slice's pointing on the sky,
which training doesn't use.

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

## Training from a shared anchor, so runs can be merged

[Issue #20](https://github.com/davidmdrpi/EIatHome/issues/20) merges small runs from many
volunteers into stronger models. Adding or averaging weights only makes sense if every run
tokenizes the same way and starts from the same weights. Without that, token 3 means different
frames in different runs, and their hidden units start in different places. `--anchor v1` fixes
both (see [anchors/](anchors/README.md)):

```bash
python -m tracks.llm.train --anchor v1 --manifest tracks/llm/data/slices/index.json \
  --allow-within-session --seed 42 --output runs/w42
python -m tracks.llm.anchor delta runs/w42/real.pt --output runs/w42/tau.npz   # τ = θ − θ₀
```

- **Tokens:** the run tokenizes with the anchor's pinned codebook instead of fitting its own, so
  its report's tokenizer digest is the anchor's.
- **Start:** the model starts from the anchor's θ₀ and takes its size. A conflicting size option
  is refused. The seed still sets data order and dropout.
- **Refused data:**
  - another data product (v1 is defined on the 49-frame 6-second slices; the Live Lab's 16-frame
    slices are another product);
  - validation or test frames the codebook was fitted on.
- **Recorded:** the anchor's hashes go in the report and checkpoints. The report also counts the
  manifest's frames the codebook saw: the first scan of each family, the control family's too.
- **Task vectors:** `tracks.llm.anchor.checkpoint_task_vector` gives a checkpoint's τ in
  parameter order, refusing one trained from another anchor or with another tokenizer.
  `flatten` and `apply` convert between τ and weights.

**What seeds alone give.** Three runs from `v1` (seeds 42, 43, 44, default training, Voyager)
score 1.92 bits per token on the test scan, against 3.78 for θ₀ untrained. Their task vectors
are almost the same: cosine similarity 0.996, and 98.5% of parameters change in the same
direction. With the same data and the same start, a seed only changes data order and dropout,
so seed runs give a merge little diversity to combine. That's a reason to try runs on
different data, milestone 2's shards; it doesn't show that merging seed runs can't help.
Milestone 2 measures both. All three still improved at the 50-epoch limit.

### Merging shard workers

Milestone 2 starts with the plainest merge: rounds of workers trained on disjoint shards, each
checked and gated on validation, then averaged by training samples
([merge.py](merge.py); results in [MERGE_RESULTS.md](MERGE_RESULTS.md)).

```bash
M=tracks/llm/data/slices/index.json
for i in 1 2 3; do
  python -m tracks.llm.train --anchor v1 --manifest $M --allow-within-session --variants real \
    --shard $i/3 --output runs/r1-w$i
done
python -m tracks.llm.merge round --output runs/round1 runs/r1-w*/real.pt
# Round 2: the workers start from round 1's merge.
python -m tracks.llm.train ... --start runs/round1/merged.pt --shard 1/3 --output runs/r2-w1
python -m tracks.llm.merge round --start runs/round1/merged.pt --output runs/round2 runs/r2-w*/real.pt
python -m tracks.llm.merge replay runs/round1/round.json   # recompute the round from its record
python -m tracks.llm.merge score runs/round1/merged.pt     # test and the control, for reporting only
```

- **Workers:** `--shard i/n` trains on every n-th training slice in a fixed order by content,
  so shards are disjoint and don't depend on the manifest's order. Validation and test aren't
  sharded. `--start` continues from a merged round (round 1 starts from θ₀). `--variants real`
  skips the surrogate controls. Each worker records its round, its start's digest, its shard,
  its seed and its number of training targets.
- **Refused before the gate:**
  - another anchor, tokenizer or model size;
  - another round or start;
  - a surrogate stream;
  - non-finite changes, or any change over 10;
  - a split that isn't the round's;
  - the same shard and seed twice, which counts once.
- **The gate:** each worker, then the average, must score within 0.02 bits per token of the
  incumbent on validation. A round never reads the test split.
- **The record:** `round.json` names every contributor by file hash, its fate and why, the
  weights and rule, and the merged weights' digest. `replay` recomputes the round from the same
  files and refuses any difference. A merged checkpoint loads with `sample.py`, and only starts
  a round if its weights match its recorded digest.

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
