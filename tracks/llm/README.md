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

Milestone 2: rounds of workers trained on disjoint shards, each checked and gated, then merged
by a recorded rule ([merge.py](merge.py); results in [MERGE_RESULTS.md](MERGE_RESULTS.md)).
The default rule is the plain baseline, an average by training targets.

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

# DARE-TIES with a greedy soup and DiLoCo's outer step (every round of a chain uses the same one):
python -m tracks.llm.merge round --merge ties --dare-p 0.9 --soup greedy \
  --outer-lr 0.7 --outer-momentum 0.9 --output runs/round1 runs/r1-w*/real.pt
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
  - a validation or test split that isn't the round's, or a manifest other than the one the
    workers trained on (its slices' digest);
  - training slices that aren't the worker's shard's, or a target count that isn't theirs: the
    round recounts targets from the shard, and weights by its own count;
  - the same shard and seed twice, which counts once;
  - with DARE, a compressed delta over a result card's 16 KB.
- **Refused outright:**
  - workers divided into different numbers of shards, whose shards would overlap;
  - a chain that changes family, dataset or split: `train.py --start` and `merge round
    --start` both refuse a merge from another family's runs, or from another manifest;
  - a chain that changes its outer step.
- **The rule:**
  - **DARE** (`--dare-p p`, `--dare-method random|magnitude`): each delta is compressed as a
    volunteer would send it. A fraction p of its entries is dropped; the rest are stored as
    int8 with one scale per tensor, and their indices as varint gaps, in compact JSON. The round
    gates and merges what the payload rebuilds, and records each payload's size. A random drop
    rescales the survivors by 1/(1 − p), so the expected delta is unchanged. Its stream is
    seeded by the worker's weights, so a replay redraws it. Dropping by magnitude keeps the
    survivors unscaled, since they're already the large entries. At p = 0.9 a delta is about
    8 KB.
  - **TIES** (`--merge ties`, `--ties-keep k`): trim each delta to its largest fraction k,
    elect each parameter's sign as the sign of the weighted sum, and average only the nonzero
    entries that agree with it. A delta that's zero at a parameter doesn't pull the others
    toward zero there. `--ties-mean all` (milestone 5) divides the agreeing entries' weighted
    sum by every contributor's weight instead. After DARE it removes the factor TIES's own mean
    keeps: DARE's 1/(1 − p) keeps a sum over all contributors right on average, while a mean
    over the survivors alone keeps that factor on every entry. It's unbiased only where the
    contributors agree in sign; where they disagree, sign election still biases it. A round
    recorded before the setting existed replays with TIES's own mean.
  - **Greedy soup** (`--soup greedy`): the accepted workers are tried best first by
    validation. Each is kept only if the merge scores lower on validation with it than without
    it. A worker that barely moved from the incumbent can't dilute the others unless it helps.
  - **DiLoCo** (`--outer-lr`, `--outer-momentum`): the merged delta τ̄ is a pseudo-gradient
    for an outer Nesterov step: m ← μm + τ̄, θ ← θ + lr(τ̄ + μm). The momentum is saved in the
    merged checkpoint, with its digest, and carried to the next round. lr 1 with no momentum is
    θ + τ̄.
- **Weights:** each merged shard's training targets, shared equally by the workers on that
  shard (several seeds of one shard count its slices once).
- **The gate:** each worker, and the merge after the outer step, must score within 0.02 bits
  per token of the incumbent on validation. A round never reads the test split or the
  reporting control.
- **The selection control** (the other family's validation scan) is recorded for the
  incumbent, each worker and the merge, as a diagnostic, not a veto: its bits, and how much the
  model gains from the real order of its tokens over 19 shuffled orders, with a permutation
  p-value. Its likelihood alone can't show that a model found structure there: matching token
  frequencies or calibration lowers it too. On the published off-source scans, neither number
  separates a model trained on them from a Voyager model, so neither is a veto (see
  [MERGE_RESULTS.md](MERGE_RESULTS.md)). The other family's test scan is the reporting control,
  used only by `score`.
- **DARE's rebuilt delta** is held to the same finite and size bounds as the worker's own,
  since rescaling by 1/(1 − p) multiplies the kept entries.
- **The record:** `round.json` names every contributor by file hash, with its fate and why. It
  also holds each soup trial, the weights, the whole rule, and the digests of the merged
  weights and momentum. `replay` recomputes the round from the same files and refuses any
  difference. A merged checkpoint loads with `sample.py`, and only starts a round if its
  weights and momentum match their recorded digests.
- **Not a defense against adversaries:** the validation data and the control are public, so a
  delta crafted to pass the gate could pass it.

### A mixture of experts across families

Milestone 3 upcycles one specialist per family into a mixture of experts, Branch-Train-MiX
style ([moe.py](moe.py); results in [MERGE_RESULTS.md](MERGE_RESULTS.md)). It uses anchor v2,
whose codebook also covers the Rosetta control: synthetic 16-tone FSK text from
[synthetic.py](synthetic.py), published in `data/slices/index-rosetta.json`.

```bash
M=tracks/llm/data/slices/index-rosetta.json
for f in voyager offsource rosetta; do
  python -m tracks.llm.train --anchor v2 --manifest $M --family $f --allow-within-session \
    --variants real --train-stride 1 --output runs/$f
done
python -m tracks.llm.moe upcycle --finetune router+backbone --output runs/moe \
  runs/voyager/real.pt runs/offsource/real.pt runs/rosetta/real.pt
python -m tracks.llm.moe_bench --output runs/moe-bench   # the milestone 3 benchmark
```

- **Specialists:** workers trained from θ₀ (not a later round) on distinct families of one
  dataset. Each is checked against the manifest's split, like a merge round's workers. Its
  record must also show its family's whole training split, not a shard, held-out slices or
  another family's, with that split's number of targets.
- **The mixture:**
  - each specialist's feed-forward layers become one expert in every block;
  - the backbone (embedding, attention, layer norms, head) is θ₀ plus the specialists'
    averaged deltas (`--merge ties` for TIES);
  - a router per block, initialized from a PCG64 stream, picks two experts per token, with
    gates softmax(top2(x·W_g)). Each expert runs only on the tokens routed to it, so a token
    passes through the parameters the report counts as active.
- **Finetuning** (`--finetune router`, `router+backbone` or `none`): on all the families'
  training scans together, with the experts frozen.
  - The load-balancing loss N Σ fᵢ Pᵢ is weighted 0.01.
  - The epoch is chosen on the families' validation scans together.
- **The report:** each family's validation and test bits, which experts each family's tokens
  go to, the total parameters and those a token uses, and the checkpoint `moe.pt`.

**Milestone 3b** tests the two bottlenecks milestone 3 found: experts with little to learn,
and a merged backbone from specialists that branched from random weights. It uses anchor v3,
whose peak tokenizer makes each Rosetta symbol exactly one token, on `index-moe-v2.json`, which
adds a longer Rosetta and an RFI sweep. Specialists can branch from a warm start: θ₀ trained a
few epochs on every family together.

```bash
M=tracks/llm/data/slices/index-moe-v2.json
python -m tracks.llm.moe warmup --anchor v3 --output runs/warm          # 10 epochs, the families together
python -m tracks.llm.train --anchor v3 --manifest $M --family rosetta --allow-within-session \
  --variants real --train-stride 1 --warm runs/warm/warm.pt --output runs/rosetta-warm
python -m tracks.llm.moe upcycle --anchor v3 --warm runs/warm/warm.pt --output runs/moe-warm ...
python -m tracks.llm.moe_ablation --output runs/moe-ablation             # the milestone 3b benchmark
```

- **The warm start's lineage:**
  - a warm start records the slices it trained on, and is refused unless they are exactly its
    families' training splits under the manifest it names;
  - `train.py --warm` refuses a family the warm start didn't cover, or `--warm` with
    `--start`;
  - `upcycle --warm` takes only specialists that all branched from that warm start, and builds
    the backbone as the warm start plus their averaged deltas from it.
- **Dense rivals of the mixture's size:** `moe.matched_config` gives the width (a multiple of
  8), then the feed-forward size, closest to a parameter count. `SignalLM` takes an optional
  `ffn` size; the anchors leave it out.

The Live Lab's token transformer is a different model, in TF.js, so it has an anchor of its own
(`token-anchor/v1`, [web/js/anchor.js](../../web/js/anchor.js)) and a merge work unit in the
browser (milestones 4 and 5; results in [MERGE_RESULTS.md](MERGE_RESULTS.md)).

### Routing whole sequences to the specialist bank

Milestone 5 ([route_bench.py](route_bench.py)) keeps milestone 3b's warm specialists whole and
picks one for each sequence without the family label, from the tokens already seen, so no
target helps choose the model that predicts it. The routers score each specialist before each
target:

- by the sequence's first token, or by every token so far, using its family's token
  frequencies (naive Bayes over 99 numbers);
- by the specialists' own likelihood of the tokens so far, followed throughout or committed
  after 4 or 8 tokens;
- or as a posterior-weighted mixture of all three specialists.

One router is chosen on validation by the mean over families, and the concatenated choice is
recorded beside it. The benchmark also retrains milestone 3b's dense rivals and chooses their
epoch both ways, the selection check that milestone 3b's review asked for.

```bash
python -m tracks.llm.route_bench --output runs/route-bench   # about half an hour on a CPU
```

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
