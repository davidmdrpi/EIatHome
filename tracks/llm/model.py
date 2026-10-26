"""GPT-style causal transformer and token-weighted next-frame likelihood."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SignalLM(nn.Module):
    def __init__(self, vocab_size=16, context=15, width=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        if min(vocab_size, context, width, layers, heads) < 1 or width % heads:
            raise ValueError("Positive model dimensions required; width must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.config = dict(vocab_size=vocab_size, context=context, width=width,
                           layers=layers, heads=heads, dropout=dropout)
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        # Independent layers: avoid cloned encoder layers starting with identical parameters.
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        if not 1 <= n <= self.config["context"]:
            raise ValueError("Input length exceeds the checkpoint's context")
        x = self.token(tokens) + self.position(torch.arange(n, device=tokens.device))
        mask = torch.ones(n, n, device=tokens.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(self.norm(x))


def windows(sequences, context):
    """No cross-slice boundaries; each target occurs once, including short tails."""
    return [(s[start:start + context], s[start + 1:start + context + 1])
            for s in sequences for start in range(0, len(s) - 1, context)]


def batches(examples, batch_size, device, order=None):
    order = range(len(examples)) if order is None else order
    for start in range(0, len(examples), batch_size):
        chosen = [examples[i] for i in order[start:start + batch_size]]
        n = max(len(y) for _, y in chosen)
        x = torch.zeros(len(chosen), n, dtype=torch.long, device=device)
        y = torch.full_like(x, -100)
        for i, (a, b) in enumerate(chosen):
            # Inputs can include one unused final token in the last chunk; trim to targets.
            x[i, :len(b)] = torch.as_tensor(a[:len(b)], device=device)
            y[i, :len(b)] = torch.as_tensor(b, device=device)
        yield x, y


@torch.no_grad()
def evaluate(model, sequences, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for x, y in batches(windows(sequences, model.config["context"]), batch_size, device):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum")
        total += loss.item()
        count += (y != -100).sum().item()
    if not count:
        raise ValueError("Evaluation requires at least one next-token target")
    bits = total / count / math.log(2)
    return {"bits_per_token": bits, "perplexity": 2 ** bits, "targets": count}


def fit(sequences, validation, config, epochs, batch_size, lr, patience, seed, device):
    torch.manual_seed(seed)
    model = SignalLM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    examples = windows(sequences, config["context"])
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in batches(examples, batch_size, device, rng.permutation(len(examples))):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = (y != -100).sum().item()
            total += loss.item() * n
            count += n
        val = evaluate(model, validation, batch_size)["bits_per_token"]
        history.append({"epoch": epoch, "train_bits": total / count / math.log(2), "validation_bits": val})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


@torch.no_grad()
def generate(model, prompt, count=32, seed=0, temperature=1.0):
    if not prompt or count < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need a nonempty prompt, nonnegative length, and positive finite temperature")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens = list(prompt)
    for _ in range(count):
        x = torch.tensor([tokens[-model.config["context"]:]], device=device)
        probabilities = torch.softmax(model(x)[0, -1] / temperature, dim=-1)
        tokens.append(torch.multinomial(probabilities, 1, generator=rng).item())
    return tokens[len(prompt):]
