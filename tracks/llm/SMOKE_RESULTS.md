# Published-data smoke run

These are functional checks on the one-session July 2020 data, not independent-session
results or evidence of language. No hyperparameter search was performed. Both runs use
the defaults documented in [README.md](README.md): seed 42, 799,360 parameters, 50 maximum
epochs with validation early stopping, and an explicit within-session scan holdout.

Commands (from the repository root):

```bash
python -m tracks.llm.train --allow-within-session --output runs/voyager-review-fix
python -m tracks.llm.train --family offsource --allow-within-session --output runs/offsource-review-fix
```

| Training family / stream | Transformer bits/token | Unigram bits/token | Markov-1 bits/token | Best epoch | Test targets |
|---|---:|---:|---:|---:|---:|
| Voyager / real | 1.2236 | 3.6866 | 2.7344 | 22 | 45 |
| Voyager / shuffled | 3.8993 | 3.6702 | 3.7541 | 8 | 45 |
| Voyager / markov | 4.1244 | 4.1689 | 4.0436 | 2 | 45 |
| Off-source / real | 3.8926 | 3.6545 | 4.0443 | 6 | 45 |
| Off-source / shuffled | 3.7408 | 3.6395 | 3.9176 | 7 | 45 |
| Off-source / markov | 4.0117 | 4.1224 | 4.0323 | 2 | 45 |

Uniform prediction is 4 bits/token. Each variant is evaluated on its own corresponding
test stream; unigram and Markov baselines are fitted to that variant’s training stream.
The two families fit separate codebooks, so their raw bits/token are not directly comparable.
The Voyager run predicts signal-frame dynamics better than its baselines and shuffled
control in this scan holdout. The off-source model does not beat its unigram baseline.
A drifting carrier is predictable without linguistic content. Only 45 test targets per run
and a single shared observing session make these results preliminary.

Environment:

- Python 3.12.14; NumPy 2.3.5; PyTorch 2.14.1+cpu; CPU, one thread.
- Dataset fingerprint: `c177da9075e20ce4ef3165e20c2669f30e26db5f351efb6427d1ec605281d339`.
- Training code fingerprint: `802101756afbf8335c30f98d0f6af873ca072c671df472c67974f8601b436402` (data, model, training and runtime modules).

Both runs also wrote loadable inference checkpoints and generated frames. The standalone
sampling command in README.md was exercised and produced a finite 8 × 64 frame array,
bounded by the trained context. The earlier 32-frame continuation is superseded: it slid position alignments beyond the training distribution.
Binary checkpoints are intentionally not committed; the commands reproduce them.

Regenerated after the review fixes. Both reports record one scan group, three distinct
slices, 48 frames and 45 targets in each split. The reported four-decimal metrics are
unchanged by the optimized tokenizer and surrogate implementations.
