"""Learned detectors, trained on the train scan only (see the benchmark protocol in README.md).

Two of the project's models, each giving two statistics:

- next-frame GRU (Track A's model, ported from TF.js and checked against it): predicts each
  frame from the frames before it;
- token model (tracks/llm): k-means tokens of frames and a causal transformer over them.

surprise: how badly the model predicts the window (mean squared residual; bits per token).
gain: how much better it predicts than a model of no structure (predicting empty frames; the
token frequencies). A carrier can be very predictable and a glitch very surprising: the two
statistics measure different things.

Training follows Track A's work units (track-a/v0): signal-free windows plus synthetic drifting
tones, injected with the train scan's own seeds. Tones are a class seen in training; the other
signal classes and the decoys aren't. No labels are used: both models only predict.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tracks.llm.data import Tokenizer
from tracks.llm.model import SignalLM, fit as fit_token_model
from tracks.llm.runtime import training_runtime

from .inject import STRENGTHS, draw, render

FIT_FRACTION = 0.75  # of the train scan's windows (3,072 of 4,096); the rest are for early stopping
TONE_FRACTION = 0.5
GRU = dict(hidden=64, lr=3e-3, batch=64, epochs=30, patience=5)  # Track A: hidden 16-64, Adam 3e-3, batch 64
TOKENS = dict(vocab=16, width=128, layers=4, heads=4, dropout=0.1, lr=3e-4, batch=32, epochs=30, patience=5)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def state_digest(module):
    return digest(*(v.detach().cpu().numpy() for v in module.state_dict().values()))


def training_windows(train, seed, fit_fraction=FIT_FRACTION, tone_fraction=TONE_FRACTION):
    """Disjoint fit and development windows from the train scan, each with drifting tones injected
    into tone_fraction of its windows at the design strengths. Returns (fit, dev, info)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    fit_count = round(fit_fraction * len(train))
    frames, channels = train.shape[1:]
    parts = {}
    for name, idx in (("fit", order[:fit_count]), ("dev", order[fit_count:])):
        windows = np.array(train[idx], dtype=np.float32)
        for i in rng.choice(len(idx), round(tone_fraction * len(idx)), replace=False):
            spec = draw(rng, "tone", float(rng.choice(STRENGTHS)), frames, channels)
            windows[i] += render(spec, frames, channels).astype(np.float32)
        parts[name] = windows
    info = {"scan": "train", "seed": seed, "fit_windows": len(parts["fit"]), "dev_windows": len(parts["dev"]),
            "tone_fraction": tone_fraction, "tone_strengths": list(STRENGTHS),
            "windows_sha256": digest(parts["fit"], parts["dev"])}
    return parts["fit"], parts["dev"], info


def track_a_normalise(windows):
    """Track A's per-snippet normalization (normalise in web/js/signals.js): subtract the median,
    divide by 1.4826 x the median absolute deviation, then asinh. As there, the median of an
    even count is the upper of the two middle values."""
    x = np.asarray(windows, dtype=np.float32)
    flat = x.reshape(len(x), -1)
    middle = flat.shape[1] // 2
    median = np.sort(flat, axis=1)[:, middle]
    scale = 1.4826 * np.sort(np.abs(flat - median[:, None]), axis=1)[:, middle]
    scale[scale == 0] = 1.0
    return np.arcsinh((x - median[:, None, None]) / scale[:, None, None]).astype(np.float32)


def hard_sigmoid(x):
    """TF.js's hardSigmoid, the default gate activation of its GRU."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class NextFrame(nn.Module):
    """Track A's predictor as TF.js computes it (buildModel in web/js/engine.worker.js):
    tf.layers.gru, then a dense layer giving the next frame.

    Not PyTorch's GRU, which differs in four ways:
    - TF.js has one bias per gate, not separate input and recurrent biases;
    - its gates use a hard sigmoid;
    - it applies the reset gate before the recurrent product;
    - it initializes the input kernel Glorot-uniform and the recurrent kernel orthogonally
      (biases zero).
    Weights use TF.js's layout: kernel (channels, 3 x units) and recurrent kernel
    (units, 3 x units), with gates in the order z, r, candidate. tests/track_a_gru.json holds a
    reference that this and TF.js itself must both reproduce."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.units = hidden
        self.kernel = nn.Parameter(torch.empty(channels, 3 * hidden))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden, 3 * hidden))
        self.bias = nn.Parameter(torch.zeros(3 * hidden))
        self.head = nn.Linear(hidden, channels)
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform, as the engine sets
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.xavier_uniform_(self.head.weight)  # tf.layers.dense's default Glorot uniform
        nn.init.zeros_(self.head.bias)

    def states(self, frames):
        u = self.units
        projected = frames @ self.kernel + self.bias
        h = frames.new_zeros(frames.shape[0], u)
        out = []
        for t in range(frames.shape[1]):
            xz, xr, xh = projected[:, t].split(u, dim=-1)
            hz, hr = (h @ self.recurrent_kernel[:, :2 * u]).split(u, dim=-1)
            z, r = hard_sigmoid(xz + hz), hard_sigmoid(xr + hr)
            candidate = torch.tanh(xh + (r * h) @ self.recurrent_kernel[:, 2 * u:])
            h = z * h + (1 - z) * candidate
            out.append(h)
        return torch.stack(out, dim=1)

    def forward(self, frames):
        return self.head(self.states(frames))


def fit_gru(fit, dev, seed, hidden, lr, batch, epochs, patience):
    """Mean-squared-error training to predict frames 1..T-1 from 0..T-2, keeping the epoch with the
    lowest development loss. Returns (model, history, best_epoch)."""
    torch.manual_seed(seed)
    x, x_dev = torch.from_numpy(track_a_normalise(fit)), torch.from_numpy(track_a_normalise(dev))
    model = NextFrame(x.shape[2], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)  # tf.train.adam's epsilon
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, history = math.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.0
        for start in range(0, len(x) - batch + 1, batch):  # whole batches only, as Track A
            xb = x[order[start:start + batch]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb[:, :-1]), xb[:, 1:])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        seen = len(x) - len(x) % batch
        model.eval()
        with torch.no_grad():
            dev_loss = F.mse_loss(model(x_dev[:, :-1]), x_dev[:, 1:]).item()
        history.append({"epoch": epoch, "train_mse": total / seen, "dev_mse": dev_loss})
        if dev_loss < best:
            best, best_epoch = dev_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch


@torch.no_grad()
def gru_residuals(model, windows, batch=512):
    """Per window: the model's mean squared prediction residual, and that of predicting empty
    frames (Track A-normalized units)."""
    residual, null = [], []
    for start in range(0, len(windows), batch):
        z = torch.from_numpy(track_a_normalise(windows[start:start + batch]))
        target = z[:, 1:]
        residual.append(((target - model(z[:, :-1])) ** 2).mean(dim=(1, 2)))
        null.append((target ** 2).mean(dim=(1, 2)))
    return torch.cat(residual).numpy().astype(np.float64), torch.cat(null).numpy().astype(np.float64)


def fit_tokens(fit, dev, seed, vocab, width, layers, heads, dropout, lr, batch, epochs, patience):
    """A tokenizer fitted on the fit windows' frames, and tracks/llm's model over each whole window.
    Returns (tokenizer, model, unigram, history, best_epoch)."""
    frames, channels = fit.shape[1:]
    tokenizer = Tokenizer.fit([fit.reshape(-1, channels)], vocab, seed)
    encode = lambda w: tokenizer.encode(w.reshape(-1, channels)).reshape(len(w), frames)
    fit_seqs, dev_seqs = list(encode(fit)), list(encode(dev))
    config = dict(vocab_size=vocab, context=frames - 1, width=width, layers=layers, heads=heads,
                  dropout=dropout, position="absolute")
    model, history, best_epoch = fit_token_model(fit_seqs, dev_seqs, config, epochs, batch, lr, patience, seed, "cpu")
    counts = np.bincount(np.concatenate(fit_seqs), minlength=vocab) + 1.0  # add-one, as the Markov baseline
    return tokenizer, model.eval(), counts / counts.sum(), history, best_epoch


@torch.no_grad()
def token_bits(tokenizer, model, unigram, windows, batch=256):
    """Per window: bits per token of frames 1..T-1 under the model, and under the token
    frequencies."""
    n, frames, channels = windows.shape
    tokens = tokenizer.encode(np.asarray(windows).reshape(-1, channels)).reshape(n, frames)
    model_bits = []
    for start in range(0, n, batch):
        x = torch.from_numpy(tokens[start:start + batch])
        logp = F.log_softmax(model(x[:, :-1]), dim=-1)
        nll = -logp.gather(2, x[:, 1:, None]).squeeze(2)
        model_bits.append(nll.mean(dim=1) / math.log(2))
    unigram_bits = -np.log2(unigram[tokens[:, 1:]]).mean(axis=1)
    return torch.cat(model_bits).numpy().astype(np.float64), unigram_bits


def learned_detectors(train, seed=0, gru=GRU, tokens=TOKENS):
    """Train both models on the train scan's windows and return their four detectors, each with
    a config recording what it is and what it was trained on, and the trained state."""
    fit, dev, info = training_windows(train, seed)
    with training_runtime(1, "cpu"):
        gru_model, gru_history, gru_best = fit_gru(fit, dev, seed, **gru)
        tokenizer, token_model, unigram, token_history, token_best = fit_tokens(fit, dev, seed, **tokens)
    gru_common = {"model": "Track A next-frame GRU: TF.js's GRU cell and initializers, ported (tracks/recovery/learned.py)",
                  "preprocessing": "per-window median/MAD, asinh (as web/js/signals.js)",
                  "parameters": sum(p.numel() for p in gru_model.parameters()), "hyperparameters": dict(gru),
                  "training": info, "best_epoch": gru_best, "dev_mse": gru_history[gru_best - 1]["dev_mse"],
                  "state_sha256": state_digest(gru_model)}
    token_common = {"model": "tracks/llm SignalLM over k-means frame tokens, context = the whole window",
                    "parameters": sum(p.numel() for p in token_model.parameters()), "hyperparameters": dict(tokens),
                    "training": info, "best_epoch": token_best,
                    "dev_bits": token_history[token_best - 1]["validation_bits"],
                    "tokenizer_sha256": hashlib.sha256(repr(tokenizer.state()).encode()).hexdigest(),
                    "state_sha256": state_digest(token_model)}

    def gru_surprise(windows):
        return gru_residuals(gru_model, windows)[0]

    def gru_gain(windows):
        residual, null = gru_residuals(gru_model, windows)
        return null - residual

    def token_surprise(windows):
        return token_bits(tokenizer, token_model, unigram, windows)[0]

    def token_gain(windows):
        model_bits, unigram_bits = token_bits(tokenizer, token_model, unigram, windows)
        return unigram_bits - model_bits

    gru_surprise.config = {**gru_common, "statistic": "mean squared prediction residual"}
    gru_gain.config = {**gru_common, "statistic": "residual of predicting empty frames minus the model's"}
    token_surprise.config = {**token_common, "statistic": "bits per token under the model"}
    token_gain.config = {**token_common, "statistic": "bits per token under token frequencies minus under the model"}
    detectors = {"gru-surprise": gru_surprise, "gru-gain": gru_gain,
                 "token-surprise": token_surprise, "token-gain": token_gain}
    state = {"gru": gru_model.state_dict(), "token": token_model.state_dict(), "tokenizer": tokenizer.state(),
             "unigram": unigram.tolist(), "training": info,
             "history": {"gru": gru_history, "token": token_history}}
    return detectors, state
