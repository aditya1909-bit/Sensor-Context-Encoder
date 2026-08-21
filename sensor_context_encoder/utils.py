"""Reproducibility, metrics, device, and serialization helpers."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import ACTIVITY_NAMES


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state needed to continue stochastic training."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])


def move_to_cpu(value: Any) -> Any:
    """Recursively detach tensors for portable checkpoint serialization."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "estimating"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this Python environment")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional arrays with equal shape")
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class_f1: list[float] = []
    for index in range(6):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return {
        "macro_f1": float(np.mean(per_class_f1)),
        "accuracy": float((labels == predictions).mean()),
        "per_class_f1": dict(zip(ACTIVITY_NAMES, per_class_f1, strict=True)),
        "confusion_matrix": confusion.tolist(),
    }


def make_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("A derangement requires at least two examples")
    generator = np.random.default_rng(seed)
    original = np.arange(size)
    while True:
        permutation = generator.permutation(size)
        if np.all(permutation != original):
            return permutation


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency_ms(
    operation: Callable[[], torch.Tensor],
    device: torch.device,
    runs: int = 50,
    warmup: int = 10,
) -> float | None:
    if runs <= 0:
        return None
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            operation()
        synchronize(device)
    return 1_000.0 * (time.perf_counter() - start) / runs


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
