# Weight-space programming: merging volunteers' workers by editing their weights

`signal-lm-wsp-bench/v0` ([wsp_bench.py](wsp_bench.py)) asks whether an editor that reads the
workers' weights can merge them into a model that predicts new observing sessions better than
any one of them. The editor can be an LLM or a numerical search. It starts from the session
benchmark ([BENCHMARK.md](BENCHMARK.md)): there, no one-shot merge beat the best worker.

Every method gets the same cohort (the published base and four workers of each seed), the same
edit language, the same validator, and the same 32 looks at validation. Each pick is chosen on
validation only, and every pick is published as an edit artifact that `verify` replays.

**Status.** This is the development run, on v0 (whose test scores have been seen before). The
search controls and the specialist bank have been run. The LLM editor is implemented and tested
against a scripted client, but it has not run yet: this environment has no API key of the
experimenter's own (below). Its rows say "not run", and nothing here should be read as a result
for it. The fresh cohort that gives the real comparison is cut only after the editor is final
([What's next](#whats-next-the-fresh-cohort)).

## Summary (v0, development)

Bits per token on the 7 test sessions (lower is better), mean over seeds 42, 43 and 44. The last
two columns compare each arm with the best worker: the mean difference on test, and on how many of
the 21 test sessions (7 per seed) it does better or worse.

| Arm | Validation | Test | Worst test session | Test − best worker | Sessions better / worse |
|---|---:|---:|---:|---:|---:|
| best worker, chosen on validation (a free reference) | 1.558 | **1.344** | **2.42** | — | — |
| plain average (a free reference) | 1.648 | 1.500 | 3.06 | +0.156 | 8 / 13 |
| **random search**: 32 random mixes of the workers per group | 1.558 | 1.344 | 2.42 | 0 | picked the best worker |
| **coordinate search**: 32 group-by-group trials from the average | 1.553 | 1.364 | 2.43 | +0.020 | 6 / 15 |
| **WSP editor** (Claude): 32 programs | not run | | | | |
| **WSP editor, anonymized**: 32 programs | not run | | | | |
| specialist bank: the 4 workers mixed per sequence (4× inference; not a merge) | 1.534 | 1.357 | 2.55 | +0.013 | 6 / 15 |
| *From the session benchmark, the same cohort:* | | | | | |
| one-shot TIES, keeping 20% (warm start from the base) | 1.607 | 1.490 | 3.04 | | |
| one-shot greedy soup (keeps only the best worker) | 1.558 | 1.344 | 2.42 | | |
| 10 rounds of plain averaging | 1.558 | 1.358 | 2.58 | | |
| 10 rounds, average + DiLoCo | 1.550 | 1.385 | 2.92 | | |
| one model on all 23 training sessions, from the base | 1.510 | **1.329** | 2.53 | | |
| first-order Markov chain fitted on the training sessions | 1.612 | 1.488 | 3.06 | | |

**What it shows: nothing beats the best worker on new sessions yet.**

- **Validation gains didn't hold on test.** Coordinate search and the specialist bank both beat
  the best worker on validation, by 0.005 and 0.024 bits. Both lost on test, by 0.020 and 0.013,
  and both did worse on 15 of the 21 test sessions. With 7 validation sessions, a gain of a few
  hundredths of a bit there says little about new sessions. That's the bar an editor has to clear.
- **Random mixes never come close.** None of the 32 random programs per seed beats the best
  worker on validation: the best scores 1.62 against 1.56. So the random arm's pick is worker 5
  alone, a free reference. Mixing in the other workers' deltas costs more than it adds, as the
  plain average showed.
- **Coordinate search makes the same pick on every seed.** It takes worker 5's weights for both
  feed-forward blocks, block 1's attention and the head. It takes the plain average for the
  embedding and block 0's attention, and worker 3's final norm (worker 3 hardly moved from the
  base: its best epoch was its first). It used all 32 queries before finishing its first pass
  over the groups, and was still improving.
- **The bank and the merges fail on different sessions.** On the 2022-04 session (HIP 8962),
  where the best worker does worse than the base, the specialist bank does much better than the
  best worker (1.58 against 1.86). It's worse on 5 of the other 6. A router that knew when to
  fall back would have room to work with; a posterior mixture over a 16-token window doesn't find it.
- **Seeds agree to within 0.01 bits**, as in the session benchmark: they show that the result
  reproduces, not that it would hold on other sessions.

## The protocol

**Edit programs.** Every method proposes programs in one declarative language:

| Operation | What it does | Bounds |
|---|---|---|
| `start` | the plain average (merge.py's, weighted by training targets) or the base | — |
| `combine` group | the group's tensors become base + Σ wₖ τₖ (τₖ = worker k − base) | each wₖ in [-1, 2]; Σ\|wₖ\| ≤ 4 |
| `trim` group | keep the largest fraction of each tensor's current delta from the base | keep in [0.05, 1] |
| `low_rank` tensor | add scale × the best rank-r approximation of one worker's delta to one matrix | r from 1 to 4; scale in [-2, 2] |
| `reset` group | the group back to the base | — |

There are 7 groups: the embedding, the attention and feed-forward halves of each block, the final
norm and the head. A program has at most 32 operations, and its weights must be finite and within
10 of the base.

**The validator** is the same whoever proposes. It checks a program's shape, names and bounds,
applies it in float64 and stores float32, and hashes the result. A refused program returns its
reason and costs nothing (a session ends after more than 32 refusals). The language reproduces the
references exactly: `{"start": "average"}` is bit for bit merge.py's plain average (the session
benchmark's one-shot merge), and a worker's weights in every group are that worker's published
weights. The tests check both.

**The budget and the pick.** Each method scores up to 32 programs on validation. It sees the mean
bits per token over the validation targets and each validation session's bits. Every method
also gets 6 reference points free: the base, the plain average and each worker alone. Its pick is
the lowest validation mean among the references and everything it scored; ties go to the earlier.
Test is scored once, after the pick. The editor can't change the rule: it proposes, and the rule
picks.

**The controls.**

- **Random search**: per group, a Dirichlet(½) mix of the four workers scaled by U[0.5, 1.5],
  seed 20261107.
- **Coordinate search**: from the plain average, each group in turn (the head first) tries each
  worker alone, the base and the average extrapolated by 1.5, and keeps any that improves
  validation.
- **The specialist bank**: route_bench.py's posterior mixture of the four workers, with a uniform
  prior. Each token's probability mixes the workers' by how well each predicted the sequence so
  far. It's scored on the same sets but isn't a candidate: it runs four models.

**The editor** (`llm`). Claude, through the Messages API, in a tool loop the harness runs. Its
first message holds:

- the model's layout and code (model.py's `SignalLM`);
- each contributor's training sessions (date, target, band, centre frequency, hits), training
  targets and best epoch;
- the validation sessions;
- per-tensor statistics of every worker's delta (norm, pairwise cosines, the share of entries
  where all four agree in sign, top singular values);
- the references' validation scores, and its budget.

It has three tools:

- `evaluate_program` scores a program through the same oracle as the controls.
- `read_tensor` returns any tensor's values (free).
- `finish` ends the session.

The session ends on `finish`, a spent budget, a turn with no tool call, a refusal or a truncated
turn, or 48 turns. Settings:

- model `claude-opus-5-5` with adaptive thinking at effort `high`, and 16,000 output tokens per turn;
- prompt caching;
- the API's server-side refusal fallback (`fallbacks: "default"`). The model that served each
  turn is recorded.

The experimenter's own key (`ANTHROPIC_API_KEY`) is required; the editor always calls
api.anthropic.com. Every session's transcript, tokens (input, cache writes, cache reads, output)
and dollar cost at list price ($4 and $20 per million input and output tokens, $5 for cache writes
and $0.20 for cache reads) are published with its picks.

**The anonymized ablation** (`llm-anonymized`) is the same editor, with every name hidden:

- the sessions' targets, dates, bands and frequencies;
- the telescope;
- what the tokens are;
- the family names (`stars`, `blind-window`).

Sessions and contributors are just numbered. The statistics and scores are the same.

**Pinned.** `PROTOCOL_SHA256` covers all of this:

- the edit language, its bounds and the budget;
- the selection rule and the controls;
- the editor's model, settings, prompt, tools and context, and the model code it reads.

A test fails whenever the code no longer matches the pin. A run on any benchmark but v0 refuses any
other protocol.

## Costs (v0, per seed, one CPU thread)

- **Training.** The cohort every method shares cost 3.4 s for the base and 4.9 s for the four
  workers (the session benchmark's run).
- **Communication.** Each worker sends a float32 delta of 110 KB; every method here reads all
  four (441 KB). The bank must also keep all four for inference.
- **Search.** 32 validation evaluations take about 0.6 s, for either search. Each pick's
  test scoring adds about 0.1 s.
- **Inference.** One model needs 0.048 ms per test token; the bank needs 0.19 ms (4×).
- **The editor.** Not run yet. The report records its tokens, dollars and seconds per session.

## What's next: the fresh cohort

v0's test sessions have been looked at, so v0 can only rank methods during development. The real
comparison scores a frozen protocol on sessions nobody has looked at, once:

1. **Develop the editor on v0.** This needs the experimenter's API key. Each change to the prompt,
   tools or context moves `PROTOCOL_SHA256`, and the test makes every move deliberate.
2. **Freeze.** Commit and push the final pin before any new session is drawn.
3. **Cut sessions-v1.** Draw new sessions, never one v0 drew. Draw more than v0's 60 if the index
   allows: more independent sessions, not more seeds, are what narrow the uncertainty. Then cut
   them, split them, pin the split and publish the cohort:

   ```bash
   python -m tracks.llm.sessions_corpus draw --output tracks/llm/data/sessions-v1/pool.json \
     --seed 20261108 --id llm-sessions/v2 --exclude tracks/llm/data/sessions/pool.json
   python tools/extract_slices.py --pool tracks/llm/data/sessions-v1/pool.json \
     --out tracks/llm/data/sessions-v1 --width 64 --product fine_resolution
   python -m tracks.llm.session_bench split --benchmark v1 --write   # then pin SPLIT_SHA256_V1
   python -m tracks.llm.session_bench run --benchmark v1 --output runs/sessions-v1 --publish
   ```

4. **Score every arm once** with `wsp_bench run --benchmark v1 --methods random coordinate
   specialist llm llm-anonymized --publish`, and publish what it shows, better or worse.

Until step 2, `session_bench` refuses v1: it has no split.

## Not done here

- **The recovery evaluation.** Sensitivity at fixed false-alarm rates belongs to the recovery
  benchmark ([tracks/recovery](../recovery/RESULTS.md)). Its detectors run on their own windows
  with their own tokenizers. Scoring these picks there needs a detector built on anchor v4's
  models, which is separate work.
- **Training-time repair.** The edits here change weights only; no method trains. The language's
  `trim`, `low_rank` and `reset` are the bounded repairs.

## Replay

```bash
python -m tracks.llm.wsp_bench verify          # every published pick and every query replayed from the
                                               # published cohort; the picks and the bank rescored
python -m tracks.llm.wsp_bench context --seed 42 [--anonymized]   # exactly what the editor reads first
python -m tracks.llm.wsp_bench run --output runs/wsp               # the controls again (about 20 s)
python -m tracks.llm.wsp_bench run --output runs/wsp-llm --methods llm llm-anonymized   # needs ANTHROPIC_API_KEY
                                                                   # and `pip install anthropic`
```

[benchmarks/wsp-v0](benchmarks/wsp-v0) holds:

- `report.json`: every arm on every seed, with the per-session scores, each method's validation
  curve over its queries, the costs, the environment and the protocol hash;
- `edits/seed*/<method>.json`: each pick's edit artifact. It records the input weights' digests,
  the program, the output weights' digest and every query the method spent, each with its own
  digest.
