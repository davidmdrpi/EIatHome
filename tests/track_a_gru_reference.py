"""Track A's next-frame model as TF.js computes it, in plain NumPy: the reference that both the
PyTorch port (tracks/recovery/learned.py) and TF.js itself (tests/browser.test.mjs) are checked
against, through tests/track_a_gru.json.

The equations are TF.js's GRUCell.call (web/vendor/tfjs/tf.min.js), with its defaults: tanh,
hard-sigmoid gates, one bias, the reset gate applied before the recurrent product
(reset_after is unsupported), and gates ordered z, r, candidate. A dense layer then predicts
the next frame, as buildModel in web/js/engine.worker.js.

    python tests/track_a_gru_reference.py   # rewrite tests/track_a_gru.json
"""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).with_name("track_a_gru.json")


def hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def gru_states(x, kernel, recurrent_kernel, bias):
    units = recurrent_kernel.shape[0]
    h = np.zeros((x.shape[0], units))
    states = []
    for t in range(x.shape[1]):
        xz, xr, xh = np.split(x[:, t] @ kernel + bias, 3, axis=-1)
        hz, hr = np.split(h @ recurrent_kernel[:, :2 * units], 2, axis=-1)
        z, r = hard_sigmoid(xz + hz), hard_sigmoid(xr + hr)
        candidate = np.tanh(xh + (r * h) @ recurrent_kernel[:, 2 * units:])
        h = z * h + (1 - z) * candidate
        states.append(h)
    return np.stack(states, axis=1)


def reference_case(seed=11, channels=4, units=3, batch=2, frames=6):
    """Small fixed weights and inputs, large enough that the hard sigmoid clips, and the outputs."""
    rng = np.random.default_rng(seed)
    f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)  # values exact in float32
    case = {
        "channels": channels, "units": units,
        "kernel": f32(rng.normal(0, 0.9, (channels, 3 * units))),
        "recurrent_kernel": f32(rng.normal(0, 0.9, (units, 3 * units))),
        "bias": f32(rng.normal(0, 0.5, 3 * units)),
        "dense_kernel": f32(rng.normal(0, 0.7, (units, channels))),
        "dense_bias": f32(rng.normal(0, 0.3, channels)),
        "input": f32(rng.normal(0, 2.0, (batch, frames, channels))),
    }
    states = gru_states(case["input"], case["kernel"], case["recurrent_kernel"], case["bias"])
    case["states"] = states
    case["prediction"] = states @ case["dense_kernel"] + case["dense_bias"]
    return case


def pre_activations(case):
    """Every gate pre-activation the reference computes, to check the case reaches the clipping."""
    kernel, recurrent, bias = case["kernel"], case["recurrent_kernel"], case["bias"]
    units = case["units"]
    out, h = [], np.zeros((case["input"].shape[0], units))
    for t in range(case["input"].shape[1]):
        xz, xr, _ = np.split(case["input"][:, t] @ kernel + bias, 3, axis=-1)
        hz, hr = np.split(h @ recurrent[:, :2 * units], 2, axis=-1)
        out += [xz + hz, xr + hr]
        h = case["states"][:, t]
    return np.concatenate([o.ravel() for o in out])


def as_json(case):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in case.items()}


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(as_json(reference_case())) + "\n")
    print(f"wrote {FIXTURE}")
