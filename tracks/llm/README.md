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
silently overwritten. No account, remote model service or external data download is needed
after installation: these commands use the committed, verified slices.

**The current eighteen slices come from one session.** Without `--allow-within-session`,
training refuses to run until the chosen family spans at least **three independent sessions**
(one each for train, validation and test). The explicit smoke mode holds out whole scans,
keeps carrier/sidebands together, and reports `within_session_scan_holdout`. It cannot establish
generalization across observing sessions. Each current split has only 48 frames and 45
next-token targets. More epochs do not make that dataset larger.

For three or more sessions, export a manifest using `tools/extract_slices.py`, then:

```bash
python -m tracks.llm.train --manifest data/more-slices/index.json --output runs/session-test
```

Groups are chronological, approximately 60/20/20, with at least one group per partition.
Scans connected by gaps of at most twelve hours belong to one session, including transitive
chains. Identical file bytes or identical cropped windows link groups even when timestamps
disagree. Copies within a split are counted once. All families participate in finding session
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
   AdamW, gradient clipping, and validation early stopping. Chunk each slice independently;
   count every next-token target once, including short tails. Padding is excluded from loss.
5. Reload the best validation weights and score test tokens. Report bits/token, perplexity,
   target count and training-fitted unigram/Markov-1 baselines. The real model is also scored
   on the other real family as a separately labelled control; it may share observing sessions.

Default: 16 tokens, context 15, width 128, 4 layers, 4 heads, dropout 0.1, 50 maximum epochs,
patience 8, learning rate 0.0003, batch size 32, seed 42. Use `--help` for all options. The current
16-frame slices cannot test dependencies longer than 15 steps. Increasing `--context` alone
does not create longer observations. Larger collections are currently loaded into memory;
this implementation is for extracted slice collections, not streaming multi-gigabyte HDF5.

## Outputs and inference

- `report.json`: configuration, source entries and hashes for every split, independent-session
  flag, dataset/code fingerprints, package versions, loss histories, best epochs, metrics,
  control provenance and a qualitative sample.
- `real.pt`, `shuffled.pt`, `markov.pt`: selected model weights, complete model configuration,
  fitted tokenizer, split provenance, fingerprints and evaluation label. These are inference
  checkpoints; optimizer state for interrupted-run resume is not included.
- `generated_frames.npy`: 32 generated codebook-centroid frames, in approximate input
  noise-sigma units, starting from a training prompt. This is a qualitative continuation.

Continue a particular signal with the saved tokenizer and model:

```bash
python -m tracks.llm.sample runs/voyager-smoke/real.pt \
  --manifest web/data/slices/index.json \
  --slice-id voyager1-59046-81310-carrier \
  --prompt-frames 8 --tokens 32 --seed 42 --output runs/continuation
```

This produces `sample.json` and `frames.npy`; it never retrains the tokenizer. Checkpoints load
with `weights_only=True`. The sample uses a sliding context and decodes to centroid spectra,
so fine spectral detail and phase are not recovered. Power-spectrogram prediction does not
decode Voyager telemetry or establish natural language. A predictable drifting carrier can
produce a strong score without linguistic structure.

## Validation and interpretation

The [recorded smoke results](SMOKE_RESULTS.md) include both published families, their
baselines, the exact environment, and dataset/code fingerprints.

```bash
python -m unittest discover -s tests -p 'test_signal_lm.py' -v
```

CI covers integrity checks, session/copy grouping, explicit smoke-mode consent, training-only
tokenization, split-local surrogates, causal attention in training and inference, padded loss,
learning a known next-token rule, checkpoint/generation round trips, and an end-to-end check
that mutating test data changes neither the fitted tokenizer nor selected model weights.

For research, collect more independent sessions, repeat predeclared configurations across
seeds, and add the broader controls in [the experiment protocol](../../docs/experiments.md).
This command does not implement the protocol's full S0–S4 or Rosetta validation suite and
must not be used alone to claim linguistic content. Tokenizers fitted on different families
define different prediction problems; compare each model with its own baselines/surrogates,
not just raw bits/token across separately trained codebooks. CPU determinism is tested;
floating-point results need not match across PyTorch versions or devices.

Implementation references: PyTorch's [causal attention mask semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
and [restricted checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
