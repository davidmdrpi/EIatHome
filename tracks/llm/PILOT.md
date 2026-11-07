# The volunteer checkpoint pilot

`signal-lm-pilot/v0` ([pilot.py](pilot.py)) is the first time EI@Home merges model updates that
different volunteers trained on their own computers. Until now, the Live Lab's issue cards have
carried summaries, never weights. Every merge in [MERGE_RESULTS.md](MERGE_RESULTS.md) and
[BENCHMARK.md](BENCHMARK.md) merged workers one person trained.

The pilot's goal is one independently validated merge built from different volunteers'
checkpoints. Its setting is the session benchmark's, so the merged model is scored on observing
sessions none of the volunteers trained on.

**Status: open, no contributions yet.** The current model is round 1, the session benchmark's
published seed-42 base. Round 2 is open to shards 2 to 5. `python -m tracks.llm.pilot status`
shows the live state from [pilot/v0/ledger.json](pilot/v0/ledger.json).

## How to contribute

You need Python 3.10 or later, the CPU build of PyTorch, a clone of this repository, and a GitHub
account. Training takes a few seconds on one CPU thread.

```bash
git pull                                                   # the pilot's current round
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tracks/llm/requirements.txt
python -m tracks.llm.pilot status                          # which shards need contributions
python -m tracks.llm.pilot contribute --output my-run      # trains the least-covered shard
gh issue create --repo davidmdrpi/EIatHome --title "$(cat my-run/title.txt)" --body-file my-run/issue.md
```

Without `gh`, open a new issue with the title in `my-run/title.txt` and paste `my-run/issue.md`
as its body.

`contribute` trains one of the session benchmark's session shards from the current model, with
the benchmark's settings: up to 1,000 epochs, patience 50, lr 3e-4, batch 32, one CPU thread. The
seed is random unless you give one. It writes four files:

- `artifact.json`: the delta artifact (below);
- `card.json`: its summary;
- `issue.md`: the issue body, holding both, about 40 KB (GitHub's limit is 65,536 characters);
- `title.txt`: the issue's title.

The coordinator replies to your issue with a receipt.

## What a contribution is

The artifact (`eiathome-signal-lm-delta`, version 2) is a JSON document holding the change your
run made to the current model, and everything needed to check where it came from:

| Field | What it pins |
|---|---|
| `pilot` | `signal-lm-pilot/v0` |
| `anchor` | anchor v4: its id, and the digests of its codebook, its record and θ₀ |
| `tokenizer` | the peak tokenizer's kind and digest |
| `dataset` | the sessions corpus manifest, its digest, the split's pin, the family (`stars`) and the evaluation (`independent_sessions`) |
| `round` | the round number and the digest of the weights it starts from (the current model) |
| `shard` | the shard's index, count (5) and unit (sessions), a digest of its slices, its training targets |
| `training` | seed, settings, best epoch, epochs run, seconds and environment |
| `scales`, `values` | the delta: int8 values with one float32 scale per tensor, in the anchor's parameter order |
| `delta_sha256` | the SHA-256 of the float32 delta those rebuild |

The artifact's id is the SHA-256 of its text.

Quantizing to int8 moves a worker's validation score by less than 1e-4 bits, and keeps the
artifact to about 40 KB, small enough to travel inside an issue. What the coordinator evaluates
and merges is exactly the delta the artifact rebuilds.

The issue also carries a card: a summary of the round, shard, seed, the artifact's id and size,
and your own run's validation score. The card is the receipt's counterpart; it is never trusted.

## What the coordinator checks

`.github/workflows/ingest-checkpoints.yml` runs `tools/ingest_issues.mjs --kind checkpoint`, which
calls `python -m tracks.llm.pilot ingest` on each new issue titled "Checkpoint contribution: …".
Nothing in the issue is executed. The artifact and card are parsed as JSON and checked:

1. **It decodes.** The artifact rebuilds the delta its digest names, and the card names this
   artifact (its id and size) and the same round, shard and seed.
2. **Its identities are the pilot's next round.** That means the same anchor, tokenizer, dataset
   and split, the next round's number and start, an open shard with that shard's slices and
   training targets, and the pilot's training settings. A contribution trained from an older
   round is refused: pull and run `contribute` again.
3. **It counts once.** Each volunteer counts once per shard per round. An artifact recorded
   before, or a run of the same shard with the same seed as one already recorded, is a duplicate.
4. **It is independently evaluated.** The coordinator rebuilds the worker (the current model plus
   the delta) as a checkpoint and runs [merge.py](merge.py)'s own round on it alone. That checks
   the round, the start, the shard's training slices and target count, and the bound on the
   delta. It also gates the worker on the validation sessions against the current model: a
   worker more than 0.02 bits worse is rejected.
5. **It is replicated.** The coordinator retrains the claimed shard with the claimed seed itself.
   The contribution is replicated when that run is within 0.02 bits of it on validation and
   their deltas' cosine is at least 0.9.

A contribution that passes 1 to 4 is stored and recorded, whether or not it replicates. Only
replicated ones are merged.

Retraining every contribution is only affordable because the model is tiny. A larger model would
need spot checks or quorum, as BOINC does.

In a rehearsal with simulated volunteers (the tests), honest runs replicated with a cosine above
0.9999. A shard-2 run sent as shard 4 passed the gate, since it does improve validation, but its
replication cosine was 0.13. So it was recorded and never merged.

The receipt names where the artifact is stored, the coordinator's validation score next to the
current model's, the replication result, and how far the round is from a merge.

## Merging, influence and rollback

A maintainer merges the round with the **Checkpoint pilot round** workflow (`pilot.yml`, action
`merge`), or with `python -m tracks.llm.pilot merge --by <login>` and a pull request. The merge:

- needs eligible contributions from at least **two volunteers**. Eligible means validated,
  replicated, for the open round from the current model, not excluded and not merged already.
- is merge.py's round with its plain rule: the shard-weighted average of the deltas, gated
  against the current model like each worker.
- **caps each shard's influence.** Each shard weighs its share of the training targets, split
  equally among its contributions, so ten volunteers on one shard weigh what one would. Together
  with the once-per-volunteer rule, no volunteer and no shard can outvote the rest.
- is recorded whole in the ledger: merge.py's round record, with each contribution's status,
  reason and weight. An accepted merge's weights are stored under
  `pilot/v0/rounds/<digest>.bin` (raw float32, as the session benchmark publishes them, with a
  JSON header) and become the current model. A rejected merge changes nothing.

**Rollback** (`pilot.yml` action `rollback`, or `python -m tracks.llm.pilot rollback --to-round N
--reason …`) makes an earlier round the current model again. Nothing is deleted. The rounds after
it stay in the ledger, marked by the rollback event, and the contributions they merged become
eligible again unless excluded.

**Exclusion** (`exclude <artifact id> --reason …`) keeps a contribution out of later merges. A
contribution already in the current model stays there until its round is rolled back.

## Storage and replay

| Path | What's there |
|---|---|
| `pilot/v0/ledger.json` | every event in order: submissions (with the coordinator's scores and replication), exclusions, rounds, rollbacks |
| `pilot/v0/submissions/<id>.json` | each recorded artifact, exactly as submitted, named by its SHA-256 |
| `pilot/v0/rounds/<digest>.bin`, `.json` | each accepted merge's weights and header |

Issue cards and replies are the receipts. The ledger and the stored artifacts are the record.

`python -m tracks.llm.pilot verify` (run in CI) replays the whole ledger:

- each artifact, under its id, with its identities and the round that was current when it came;
- each evaluation, rescored to 1e-6 bits;
- each merge, from exactly the contributions eligible then. It is merged again from the stored
  artifacts and must reproduce its record and its weights bit for bit.

It also refuses any stored file the ledger doesn't name. `verify --replicate` retrains every
contribution too.

## Limits

- **Tiny.** 27,584 parameters, a few seconds of training per contribution, four open shards.
  The pilot tests the loop: the artifacts, the checks, the caps and the rollback. It does not
  test a model anyone needs.
- **Python only.** The Live Lab's work units run in the browser, on the 2020 Voyager slices with
  the browser anchor. Contributing from the browser would need anchor v4 and the sessions corpus
  there. Until then, contributions come from a clone.
- **Retraining as replication.** That works at this scale only; see above.
- **The plain average.** On the session benchmark it trailed the best single worker (1.500 test
  bits against 1.344; [BENCHMARK.md](BENCHMARK.md)). The pilot uses it because it's the simplest
  rule that combines everyone's contributions. The merge rules and the editor in
  [WSP_RESULTS.md](WSP_RESULTS.md) can be tried on the pilot's stored contributions later,
  offline.
