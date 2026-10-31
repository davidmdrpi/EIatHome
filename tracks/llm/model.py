"""GPT-style causal transformer and token-weighted next-frame likelihood."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


POSITIONS = ("absolute", "rope")


def validate_config(vocab_size, context, width, layers, heads, dropout, position="absolute", ffn=None):
    if min(vocab_size, context, width, layers, heads, 1 if ffn is None else ffn) < 1 or width % heads:
        raise ValueError("Positive model dimensions required; width must be divisible by heads")
    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")
    if position not in POSITIONS:
        raise ValueError(f"Position encoding must be one of {', '.join(POSITIONS)}")
    if position == "rope" and (width // heads) % 2:
        raise ValueError("Rotary position embeddings need an even width per head")


def rotary(x, start=0):
    """Rotary position embedding (RoPE): rotate feature pairs by angles proportional to position.
    Query-key dot products then depend only on relative offsets, not on where a window starts."""
    n, dim = x.shape[-2], x.shape[-1]
    inverse = 10000.0 ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float64) / dim)
    angles = torch.arange(start, start + n, device=x.device, dtype=torch.float64)[:, None] * inverse
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class RotaryBlock(nn.Module):
    """nn.TransformerEncoderLayer(norm_first=True, activation="gelu") with RoPE applied to the
    queries and keys, and causal attention. Without the rotation it computes the same function."""

    def __init__(self, width, heads, dropout, ffn=None):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        hidden = 4 * width if ffn is None else ffn
        self.feed = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        # Initialized like nn.MultiheadAttention's projections.
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out.bias)

    def attend(self, x, start=0):
        """The block's first half: x plus its causal attention output."""
        b, n, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(rotary(q, start), rotary(k, start), v, is_causal=True,
                                                  dropout_p=self.dropout if self.training else 0.0)
        return x + self.drop1(self.out(attended.transpose(1, 2).reshape(b, n, width)))

    def forward(self, x, start=0):
        x = self.attend(x, start)
        return x + self.drop2(self.feed(self.norm2(x)))


class SignalLM(nn.Module):
    def __init__(self, vocab_size=16, context=15, width=128, layers=4, heads=4, dropout=0.1, position="absolute", ffn=None):
        super().__init__()
        validate_config(vocab_size, context, width, layers, heads, dropout, position, ffn)
        self.config = dict(vocab_size=vocab_size, context=context, width=width,
                           layers=layers, heads=heads, dropout=dropout, position=position)
        if ffn is not None:  # the feed-forward size, when it isn't 4 × width
            self.config["ffn"] = ffn
        self.token = nn.Embedding(vocab_size, width)
        if position == "absolute":
            self.position = nn.Embedding(context, width)
            # Independent layers: avoid cloned encoder layers starting with identical parameters.
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                width, heads, 4 * width if ffn is None else ffn, dropout, activation="gelu", batch_first=True, norm_first=True)
                for _ in range(layers)])
        else:
            self.blocks = nn.ModuleList([RotaryBlock(width, heads, dropout, ffn) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        if not 1 <= n <= self.config["context"]:
            raise ValueError("Input length exceeds the checkpoint's context")
        x = self.token(tokens)
        if self.config["position"] == "rope":
            for block in self.blocks:
                x = block(x)
        else:
            x = x + self.position(torch.arange(n, device=tokens.device))
            mask = torch.ones(n, n, device=tokens.device, dtype=torch.bool).triu(1)
            for block in self.blocks:
                x = block(x, src_mask=mask)
        return self.head(self.norm(x))


def windows(sequences, context, stride=None):
    """Chunks of each sequence, never across sequences. With the default stride (the context),
    each target occurs once, including short tails: scoring always uses that. A smaller stride
    gives overlapping training windows that start mid-sequence."""
    stride = context if stride is None else stride
    if stride < 1:
        raise ValueError("Window stride must be positive")
    return [(s[start:start + context], s[start + 1:start + context + 1])
            for s in sequences for start in range(0, len(s) - 1, stride)]


def batches(examples, batch_size, device, order=None):
    order = range(len(examples)) if order is None else order
    for start in range(0, len(examples), batch_size):
        chosen = [examples[i] for i in order[start:start + batch_size]]
        n = max(len(y) for _, y in chosen)
        x = np.zeros((len(chosen), n), dtype=np.int64)
        y = np.full_like(x, -100)
        for i, (a, b) in enumerate(chosen):
            # Inputs can include one unused final token in the last chunk; trim to targets.
            x[i, :len(b)] = a[:len(b)]
            y[i, :len(b)] = b
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


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


def fit(sequences, validation, config, epochs, batch_size, lr, patience, seed, device, stride=None, init_state=None):
    """Train on windows `stride` frames apart (default: the context, so no overlap); validation
    always scores each target once. init_state, if given, replaces the seeded initialization
    (a shared anchor θ₀): the seed then only sets data order and dropout."""
    torch.random.default_generator.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = SignalLM(**config)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    examples = windows(sequences, config["context"], stride)
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
def generate(model, prompt, count=None, seed=0, temperature=1.0):
    """Sample a continuation. By default, fill the trained context. Absolute-position models
    stop there. RoPE models can continue: past the context they condition on the last `context`
    tokens, a sliding window whose attention depends only on relative offsets."""
    if not prompt or (count is not None and count < 0) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Need a nonempty prompt, nonnegative length, and positive finite temperature")
    context = model.config["context"]
    capacity = context + 1 - len(prompt)
    count = capacity if count is None else count
    rope = model.config.get("position", "absolute") == "rope"
    if capacity < 1 or (count > capacity and not rope):
        raise ValueError(f"Continuation exceeds the trained context; this prompt allows at most {max(0, capacity)} tokens"
                         + ("" if capacity < 1 else " (only RoPE checkpoints continue further)"))
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens = list(prompt)
    for _ in range(count):
        # The window always starts at position 0. Absolute models never slide (count is bounded),
        # so frame 0 stays at position 0; RoPE models slide over the last `context` tokens.
        x = torch.tensor([tokens[-context:]], device=device)
        logits = model(x)[0, -1].double()
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits")
        # Centre before dividing: tiny positive temperatures approach greedy decoding safely.
        probabilities = torch.softmax((logits - logits.max()) / temperature, dim=-1)
        tokens.append(torch.multinomial(probabilities, 1, generator=rng).item())
    return tokens[len(prompt):]
