# Aggregating rounds of volunteers' training: restarts, soups and the best round

`signal-lm-aggregation-bench/v0` ([aggregate_bench.py](aggregate_bench.py)) reruns the session
benchmark's ten-round chains ([BENCHMARK.md](BENCHMARK.md)) from the same published bases, with the
same training and `merge.py`'s unchanged rules and gate, and changes only what a chain does around
each round. It started from a review of the rounds the session benchmark published.

## What the review found

1. **A refused merge deadlocks DiLoCo.** `merge.py` gates every merge against the incumbent, and a
   refused round leaves the incumbent as it was, with its outer momentum. That momentum is what
   made the step overshoot, so the next round's step overshoots too, and the one after.

   On sessions-v1 this happened on every seed. The first DiLoCo round was accepted. Its momentum
   then carried 0.9 of that round's merged delta, which was the workers' whole converged step. All
   nine later merges scored 1.35 to 1.37 bits on validation against the incumbent's 1.32, and all
   nine were refused. So v1's "10 rounds, average + DiLoCo" row (1.135) scores **one outer step**,
   and what read as "DiLoCo is worse than plain rounds on v1" was this deadlock.

   On v0 no DiLoCo round was refused, so its row there is what it says.

2. **A chain reports its last round, not its best.** The gate accepts a merge up to 0.02 bits worse
   than the incumbent on validation, so a chain can drift. Each worker keeps its best epoch; the
   chain keeps whatever came last. v0's DiLoCo chains peaked at round 8 or 9 and gave some back.

3. **The greedy soup was only tried one-shot.** On v0 it was the one-shot merge that didn't dilute
   the useful worker (1.344 against the plain average's 1.500), and on v1 it kept all four. But
   both chains average every worker in every round. In v0's plain chain, three of the four workers
   hardly move in round 1 and still take three quarters of the weight, so the chain needs ten
   rounds to get where the best worker got in one.

The volunteer pilot ([PILOT.md](PILOT.md)) merges with the plain rule, without momentum, so the
deadlock can't happen there.

## What was added

- **Restart.** When the gate refuses a merge that carried outer momentum, the momentum is cleared
  and the same workers are merged again, with no retraining. This is the adaptive restart of
  accelerated gradient methods (O'Donoghue and Candès, 2015): drop the momentum when the objective
  rises. If the merge without momentum is refused too, the incumbent continues with its momentum
  cleared, so the next round starts afresh.
- **Greedy soup in every round.** `merge.py`'s `--soup greedy`, unchanged, as the chain's rule: a
  round keeps only the workers whose deltas improve the merge on validation.
- **The best round.** Every chain is scored at its last round and at its accepted round with the
  lowest validation bits (the base counts; ties go to the earlier). The gate has already scored
  every round on validation, so this costs nothing more.

Five arms, each ten rounds from the published base, on seeds 42 to 44: the session benchmark's two
chains again as references, DiLoCo with restarts, greedy-soup rounds, and greedy-soup rounds with
DiLoCo and restarts.

**The references reproduce.** On both cohorts and every seed, the two reference chains come out
with the weights the session benchmark published scores for, round decisions included. `verify`
checks that.

## Results

Bits per token on the test sessions (lower is better), the mean over seeds 42, 43 and 44.
"Against plain rounds" counts the session scores (test sessions × 3 seeds) where the arm's last
round does better or worse than the plain average's. Those are 7 or 15 independent sessions scored
three times, not 21 or 45 independent observations.

### sessions-v0 (7 test sessions; development: its test scores had been seen)

| Arm | Rounds accepted, per seed | Validation | Test | Worst test session | Against plain rounds | Best round: validation | Test | Rounds picked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| average (the session benchmark's) | 10, 10, 10 | 1.558 | 1.358 | 2.58 | — | 1.558 | 1.358 | 10, 10, 10 |
| average + DiLoCo (the session benchmark's) | 10, 10, 10 | 1.550 | 1.385 | 2.92 | 16 / 5 | 1.545 | 1.366 | 8, 8, 9 |
| average + DiLoCo, restarting | 10, 10, 10 | 1.550 | 1.385 | 2.92 | 16 / 5 | 1.545 | 1.366 | 8, 8, 9 |
| **greedy soup** | 10, 10, 10 | 1.494 | **1.335** | **2.54** | 16 / 5 | 1.494 | 1.336 | 9, 8, 10 |
| greedy soup + DiLoCo, restarting | 10, 10, 10 | 1.499 | 1.340 | 2.86 | 17 / 4 | 1.471 | 1.346 | 6, 6, 7 |
| *From the session benchmark:* best worker | | 1.558 | 1.344 | 2.42 | | | | |
| one model on all 23 training sessions, from the base / from θ₀ | | 1.510 / 1.473 | 1.329 / 1.338 | 2.53 / 2.51 | | | | |

### sessions-v1 (15 test sessions; run once, after the arms were fixed)

| Arm | Rounds accepted, per seed | Validation | Test | Worst test session | Against plain rounds | Best round: validation | Test | Rounds picked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| average (the session benchmark's) | 10, 10, 10 | 1.313 | **1.107** | 6.58 | — | 1.312 | 1.107 | 7, 10, 10 |
| average + DiLoCo (the session benchmark's) | **1, 1, 1** | 1.317 | 1.135 | 6.68 | 12 / 33 | 1.317 | 1.135 | 1, 1, 1 |
| **average + DiLoCo, restarting** | 10, 10, 10 | 1.317 | 1.109 | 6.37 | 31 / 14 | 1.311 | 1.108 | 3, 2, 5 |
| greedy soup | 10, 10, 10 | 1.314 | 1.109 | 6.74 | 19 / 26 | 1.314 | 1.108 | 5, 2, 3 |
| greedy soup + DiLoCo, restarting | 10, 10, 10 | 1.328 | 1.122 | 6.61 | 23 / 22 | 1.314 | 1.118 | 1, 2, 2 |
| *From the session benchmark:* best worker | | 1.361 | 1.170 | | | | | |
| one model on all 44 training sessions, from the base / from θ₀ | | 1.303 / 1.308 | 1.092 / 1.092 | | | | | |

**What it shows:**

- **The restart removes the deadlock, and with it DiLoCo's deficit on v1.** Each seed restarted
  once, at its second round, and then every round was accepted. DiLoCo with restarts scores 1.109
  on v1, level with plain rounds (1.107), where the deadlocked chain scored 1.135. It's better
  than plain rounds on 31 of the 45 session scores and worse on 14, with the same mean. On v0 no
  merge was refused, so the restart never fired and the two DiLoCo arms are the same chain.
- **So DiLoCo's momentum neither helps nor hurts on v1.** It's still worse than plain rounds on
  v0's test sessions (1.385 against 1.358) despite a better validation score.
- **Greedy-soup rounds are the best chain on v0 and level with the best on v1.** On v0 they score
  1.335 against plain rounds' 1.358, better on every seed (by 0.022 to 0.024 bits) and on 16 of 21
  session scores, and their validation is 0.064 bits lower. That's level with the best worker
  (1.344) and with one model trained on all the data (1.329 and 1.338), and it's the first merge
  on v0 that isn't behind the best worker. On v1, where every worker is useful, they score 1.109
  against 1.107: 19 session scores better, 26 worse. The soup kept one or two of v0's four workers
  in every round; on v1 it kept all four in the first round and one to three after.
- **v1 doesn't confirm a gain for the soup; it shows it costs nothing there.** The arms were chosen
  on v0, whose test scores had been seen. On fresh sessions the soup's rounds are 0.003 bits
  behind plain rounds, well inside what 15 sessions can resolve. What the two cohorts support
  together is that a soup in every round protects a chain when volunteers' data differ in value
  (v0) and gives up nothing measurable when they don't (v1).
- **The soup and DiLoCo's momentum don't combine.** On v1 that arm is the worst chain that ran
  (1.122). On v0 its seeds range from 1.298 to 1.372. A soup's merged delta changes membership
  from round to round, and momentum carries the last round's members into the next.
- **Picking the best round changes little.** It moves test by 0.001 bits or less for the plain
  average, the soup, and DiLoCo with restarts on v1. It helps v0's DiLoCo chain (1.385 to 1.366)
  and v1's soup with DiLoCo (1.122 to 1.118), and hurts v0's soup with DiLoCo (1.340 to 1.346).
  With 7 or 15 validation sessions, the lowest of ten validation scores is partly luck, as the WSP
  controls found ([WSP_RESULTS.md](WSP_RESULTS.md)). It's reported for every arm; it isn't a
  reliable improvement.
- **No chain closes the gap to pooling the data on v1.** The best chains score 1.107 to 1.109
  against 1.092 for one model on all 44 training sessions.

**Costs.** Every arm trains the same 40 workers per seed: about 30 s on v0 and 53 s on v1, on one
CPU thread, and 40 float32 deltas of 110 KB. A soup scores up to four trial merges per round on
validation, and a restart adds one merge; a merge with its gate takes under a second.

## What had been seen, and when

- v0's test scores had been seen for every session-benchmark arm. The arms here were fixed after
  one trial of all five on v0's seed 42, and none was changed afterwards.
- The deadlock was found by reading v1's published report: its per-round validation decisions,
  beside the test scores the session benchmark had already published for its own arms. So the
  restart was designed knowing that v1's DiLoCo chain was stuck, but not what a freed chain would
  score.
- v1 was then run once with the five arms as fixed. Nothing was tuned on it: DiLoCo's lr and
  momentum are the session benchmark's, the soup is `merge.py`'s, and the gate's tolerance is
  unchanged.
- Both runs were then repeated with the arms unchanged, after edits to how the module loads the
  published cohort and to the output path the report records. Training is deterministic: every
  repeat gave the same weights and scores, and the published reports are the last one's.

**The WSP protocol is untouched.** This benchmark is a new module. It changes nothing
`PROTOCOL_SHA256` covers (`merge.py`, the scoring code, the published cohorts), and a test checks
the hash is still the pinned one. The session benchmark's own chain loop and its published
reports are unchanged too, so they still replay; BENCHMARK.md now says what v1's DiLoCo row is.

## What to use

- **A chain with outer momentum should restart on a refused merge.** Without it, one refusal ends
  the chain while every later round's training is spent.
- **Greedy-soup rounds without momentum are the safest default of those tested**: best on v0,
  level on v1.
- **The pilot's rule is unchanged.** It merges with the plain average and no momentum, and it's
  open for contributions under that rule. Moving it to greedy-soup rounds is a change of its
  protocol, for the experimenter to make.

## Replay

```bash
python -m tracks.llm.aggregate_bench verify                  # the published report against the session benchmark's
python -m tracks.llm.aggregate_bench verify --benchmark v1
python -m tracks.llm.aggregate_bench run --output runs/aggregation-v0                  # about 8 minutes on one CPU thread
python -m tracks.llm.aggregate_bench run --benchmark v1 --output runs/aggregation-v1   # about 14 minutes
```

[benchmarks/aggregation-v0](benchmarks/aggregation-v0) and
[benchmarks/aggregation-v1](benchmarks/aggregation-v1) each hold `report.json`: every arm on every
seed, with each round's contributors, weights, decision and validation score, each restart and the
refused merge it followed, both picks' scores (per test session too), the costs, the environment
and the code digests.

**What `verify` checks**, without retraining:

- the report is on the benchmark's pinned split, seeds, rounds and arms, and starts from the
  published bases;
- the two reference chains are the session benchmark's: the same decision in every round and the
  same final weights;
- every restart is in an arm that restarts, and is counted;
- each best round is the accepted round with the lowest validation bits, and each last score is
  the last accepted round's;
- the summary is what the seeds give.

`run` retrains everything and stops a publication whose reference chains don't reproduce the
session benchmark's.
