"""Scoped process settings and publication of complete training/sample outputs."""

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def require_new_output(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")


@contextmanager
def atomic_output(output):
    """Remove incomplete runs; expose the final directory only after all writes succeed."""
    output = Path(output)
    require_new_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        yield Path(tmp)
        require_new_output(output)
        Path(tmp).rename(output)  # sibling directory: same-filesystem atomic rename


@contextmanager
def training_runtime(threads, device):
    """Restore caller settings and RNG state on success and failure (including notebooks)."""
    old_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        # manual_seed affects every CUDA generator, so preserve them all for GPU runs.
        devices = list(range(torch.cuda.device_count())) if device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if old_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_workspace
