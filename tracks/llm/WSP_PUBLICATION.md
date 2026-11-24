# Frozen WSP v1 publication

Status: **pending**, not an experimental result. An agent-environment dry run found no
`ANTHROPIC_API_KEY`; no arm ran. Its [record](wsp-attempts/dry-run-001.json) is retained
as a preflight check, not an experimenter run or a verified publication attempt.
The editor remains the untuned design frozen before sessions-v1 was drawn. This work changes
neither the protocol nor the cohort, and does not promote WSP into the live volunteer pilot.

## Execute

From the repository root, install the CPU PyTorch build, `tracks/llm/requirements.txt`, and
`anthropic`. Provide an experimenter-owned `ANTHROPIC_API_KEY` through the process environment;
do not put credentials in git or in transcripts. Unset `ANTHROPIC_AUTH_TOKEN`.

In Claude Code cloud, configure `ANTHROPIC_API_KEY` in the environment settings, where
sessions and commands using that environment can read it; network secrets are not injected
into requests to `api.anthropic.com` ([cloud environment documentation](https://code.claude.com/docs/en/cloud-environments#requests-that-never-get-the-secret)).
Alternatively, run from your own machine. The frozen client pins the public API URL and
refuses `ANTHROPIC_AUTH_TOKEN`; it does not use the session's `ANTHROPIC_BASE_URL`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tools.wsp_publication run \
  --output runs/wsp-v1-attempt-002
```

This runs random, coordinate, specialist, llm and llm-anonymized in that order on seeds 42–44,
using the existing 32-query budget and frozen selection rule. It uses a new attempt directory,
never overwrites one, never retries the experiment automatically, and does not publish directly.
The Anthropic SDK's existing transport retry behavior is unchanged. Budget and API charges
remain those of the frozen editor, not a new dollar cap imposed by this wrapper.

Before any arm runs, the runner refuses what the frozen harness would refuse: no
`ANTHROPIC_API_KEY`, an `ANTHROPIC_AUTH_TOKEN` set, no `anthropic` package, or a protocol other
than the pinned one. Each is recorded as a preflight failure.

`attempt.json` records start/end, stage and outcome, and once verified the SHA-256 of the run's
`report.json`. Exceptions record their type, not potentially
sensitive exception messages. A killed process may leave `running` or `verifying`; this means
interrupted, not successful. Preserve every attempt directory and review failures before any
restart. The frozen harness saves completed editor sessions only: an exception inside a session
can lose that session's in-memory transcript. This wrapper does not fix that limitation or claim
to recover partial API costs. Retain provider usage records for interrupted sessions.

If the pinned API model/settings cannot execute, record the failure. Do not substitute a model,
change the prompt, or tune against v1 and label it the same frozen experiment. A changed protocol
requires a new cohort under the existing plan.

## Publish and replay

Only a complete run that passes the following check is ready for publication:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tools.wsp_publication check \
  --folder runs/wsp-v1-attempt-002/results
```

The check requires all five arms and all three seeds, edit artifacts, and matching editor
transcript metadata. Then the original offline verifier rebuilds and rescores the selections.
It uses no API key. It does not attest that an API transcript is authentic or replay the LLM's
reasoning; it checks recorded artifacts and numerical selections.

Copy the verified `results` directory to `tracks/llm/benchmarks/wsp-v1` (which must not already
exist), and its `attempt.json` to `tracks/llm/wsp-attempts/<attempt>.json`. Include records of
failed attempts and all available completed artifacts/transcripts, reviewed for accidental
secrets. Then run `python -m tools.wsp_publication check` (no `--folder`). It checks the
publication as above, and also requires a verified attempt record naming its `report.json` by
SHA-256, so the published files are one verified attempt's, unedited. CI uses the same check,
allowing only the entire publication directory to be absent while this PR is pending. An existing
but partial or unattested directory fails. Remove `--allow-absent` from CI when publishing results.

## Results to report

- Validation-selected picks against training-target-weighted FedAvg, the validation-selected
  worker, random search and coordinate search. Keep the four-model specialist bank separate
  from single-model merges. Hindsight-best worker is a descriptive reference, never a selector.
- Test bits, per-session differences, worst session and blind-window controls. Report seeds
  individually and averaged. Fifteen sessions measured across three seeds are fifteen independent
  session units, not forty-five independent observations.
- Editor versus anonymized editor, validation queries used, termination reason, actual served
  models, tokens, recorded dollar cost, wall time and inference cost.
- Publish negative and null results. A complete verified comparison finishes the experiment;
  it does not require an LLM win. Any uncertainty analysis added after seeing published baseline
  scores must be labeled exploratory, without changing the frozen selection rule.

Keep the PR draft until the full publication, offline replay and result discussion are present.
The next operational milestone remains a merge from at least two real volunteers; the current
pilot ledger is empty and this coordinator-run experiment does not satisfy that milestone.
