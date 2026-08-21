"""Evaluate direct, matched-context, and shuffled-context conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sensor_context_encoder.constants import ACTIVITY_NAMES, MODEL_ID, MODEL_REVISION
from sensor_context_encoder.data import as_tensor_dataset, prepare_data
from sensor_context_encoder.models import DirectClassifier, FrozenContextClassifier
from sensor_context_encoder.utils import (
    benchmark_latency_ms,
    classification_metrics,
    make_derangement,
    resolve_device,
    set_seed,
    write_json,
)
from train import build_model


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint(checkpoint: dict[str, Any], expected_type: str) -> None:
    if checkpoint.get("model_type") != expected_type:
        raise ValueError(f"Expected a {expected_type} checkpoint")
    if expected_type == "context" and (
        checkpoint.get("model_id") != MODEL_ID
        or checkpoint.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("Context checkpoint uses an unexpected backbone or revision")


@torch.inference_mode()
def predict_direct(
    model: DirectClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_out: list[np.ndarray] = []
    predictions_out: list[np.ndarray] = []
    for features, labels in loader:
        logits = model(features.to(device))
        labels_out.append(labels.numpy())
        predictions_out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels_out), np.concatenate(predictions_out)


@torch.inference_mode()
def cache_context_embeddings(
    model: FrozenContextClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    for features, labels in loader:
        embeddings.append(model.encode_sensor(features.to(device)).cpu())
        labels_out.append(labels)
    return torch.cat(embeddings), torch.cat(labels_out)


@torch.inference_mode()
def predict_projected(
    model: FrozenContextClassifier,
    projected: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TensorDataset(projected, labels), batch_size=batch_size, shuffle=False)
    labels_out: list[np.ndarray] = []
    predictions_out: list[np.ndarray] = []
    for embedding_batch, label_batch in loader:
        logits = model.forward_from_projected(embedding_batch.to(device))
        labels_out.append(label_batch.numpy())
        predictions_out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels_out), np.concatenate(predictions_out)


def write_results(
    output_dir: Path,
    conditions: dict[str, dict[str, Any]],
    recommendation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"conditions": conditions, "recommendation": recommendation, "metadata": metadata}
    write_json(output_dir / "results.json", payload)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "macro_f1", "accuracy", "seed"))
        writer.writeheader()
        for condition, values in conditions.items():
            writer.writerow(
                {
                    "condition": condition,
                    "macro_f1": values["macro_f1"],
                    "accuracy": values["accuracy"],
                    "seed": values["seed"],
                }
            )

    markdown = [
        "# Required Results",
        "",
        "| Condition | Macro-F1 | Seed |",
        "|---|---:|---:|",
    ]
    labels = {
        "direct_sensor_classifier": "Direct sensor classifier",
        "context_embedding_model": (
            "Context-embedding model with structured objectives"
            if metadata["context_objectives_enabled"]
            else "Context-embedding model"
        ),
        "context_shuffled_embeddings": "Context model with shuffled embeddings",
    }
    for condition, values in conditions.items():
        markdown.append(f"| {labels[condition]} | {values['macro_f1']:.4f} | {values['seed']} |")
    markdown.extend(
        (
            "",
            f"Recommendation: **{recommendation['decision']}**. {recommendation['explanation']}",
            "",
        )
    )
    (output_dir / "results.md").write_text("\n".join(markdown), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    direct_checkpoint = load_checkpoint(args.direct_checkpoint)
    context_checkpoint = load_checkpoint(args.context_checkpoint)
    validate_checkpoint(direct_checkpoint, "direct")
    validate_checkpoint(context_checkpoint, "context")

    direct_stats = direct_checkpoint["normalization"]
    context_stats = context_checkpoint["normalization"]
    if direct_stats != context_stats:
        raise ValueError("Direct and context checkpoints do not use identical normalization")

    data = prepare_data(args.data_dir)
    if not np.allclose(data.mean, np.asarray(direct_stats["mean"], dtype=np.float32)) or not np.allclose(
        data.std, np.asarray(direct_stats["std"], dtype=np.float32)
    ):
        raise ValueError("Checkpoint normalization does not match the current data split")

    direct_model = build_model("direct", device, int(direct_checkpoint["config"]["seed"]))
    assert isinstance(direct_model, DirectClassifier)
    direct_model.load_state_dict(direct_checkpoint["state"])

    context_model = build_model("context", device, int(context_checkpoint["config"]["seed"]))
    assert isinstance(context_model, FrozenContextClassifier)
    context_model.load_trainable_state_dict(context_checkpoint["state"])

    direct_loader = DataLoader(
        as_tensor_dataset(data.test), batch_size=args.direct_batch_size, shuffle=False
    )
    context_loader = DataLoader(
        as_tensor_dataset(data.test), batch_size=args.context_batch_size, shuffle=False
    )
    labels, direct_predictions = predict_direct(direct_model, direct_loader, device)
    projected, context_labels = cache_context_embeddings(context_model, context_loader, device)
    context_labels_np, context_predictions = predict_projected(
        context_model, projected, context_labels, device, args.context_batch_size
    )
    if not np.array_equal(labels, context_labels_np):
        raise AssertionError("Direct and context test example order differs")

    permutation = make_derangement(len(projected), args.shuffle_seed)
    shuffled = projected[torch.from_numpy(permutation)]
    shuffled_labels, shuffled_predictions = predict_projected(
        context_model, shuffled, context_labels, device, args.context_batch_size
    )
    if not np.array_equal(labels, shuffled_labels):
        raise AssertionError("Shuffled evaluation changed label order")

    direct_metrics = classification_metrics(labels, direct_predictions)
    context_metrics = classification_metrics(labels, context_predictions)
    shuffled_metrics = classification_metrics(labels, shuffled_predictions)
    direct_metrics["seed"] = int(direct_checkpoint["config"]["seed"])
    context_metrics["seed"] = int(context_checkpoint["config"]["seed"])
    shuffled_metrics["seed"] = args.shuffle_seed

    batch_one = torch.from_numpy(data.test.features[:1]).to(device)
    direct_metrics["batch_one_latency_ms"] = benchmark_latency_ms(
        lambda: direct_model(batch_one), device, args.latency_runs
    )
    context_metrics["batch_one_latency_ms"] = benchmark_latency_ms(
        lambda: context_model(batch_one), device, args.latency_runs
    )
    shuffled_metrics["batch_one_latency_ms"] = context_metrics["batch_one_latency_ms"]

    parity_passed = context_metrics["macro_f1"] >= direct_metrics["macro_f1"] - 0.05
    dependence_passed = context_metrics["macro_f1"] - shuffled_metrics["macro_f1"] >= 0.20
    continue_development = parity_passed and dependence_passed
    recommendation = {
        "decision": "continue" if continue_development else "stop",
        "parity_gate_passed": parity_passed,
        "sensor_dependence_gate_passed": dependence_passed,
        "explanation": (
            "Both predefined continuation gates passed."
            if continue_development
            else "At least one predefined continuation gate failed."
        ),
    }
    conditions = {
        "direct_sensor_classifier": direct_metrics,
        "context_embedding_model": context_metrics,
        "context_shuffled_embeddings": shuffled_metrics,
    }
    metadata = {
        "activities": list(ACTIVITY_NAMES),
        "test_examples": len(labels),
        "direct_trainable_parameters": direct_checkpoint["trainable_parameters"],
        "context_trainable_parameters": context_checkpoint["trainable_parameters"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "context_objectives_enabled": bool(
            context_checkpoint["config"].get(
                "context_objectives", context_checkpoint["config"].get("frontier", False)
            )
        ),
        "context_objective_names": context_checkpoint["config"].get(
            "context_objective_names", context_checkpoint["config"].get("frontier_methods", [])
        ),
    }
    write_results(args.output_dir, conditions, recommendation, metadata)
    print(f"Results written to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=43)
    parser.add_argument("--direct-batch-size", type=int, default=128)
    parser.add_argument("--context-batch-size", type=int, default=16)
    parser.add_argument("--latency-runs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
