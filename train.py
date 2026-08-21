"""Train the direct classifier or frozen-LLM context model."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sensor_context_encoder.constants import (
    MODEL_ID,
    MODEL_REVISION,
    VALIDATION_SUBJECTS,
)
from sensor_context_encoder.context_objectives import (
    CONTEXT_OBJECTIVE_NAMES,
    ContextLossWeights,
    SensorViewAugmenter,
    compute_context_losses,
)
from sensor_context_encoder.data import PreparedData, as_tensor_dataset, prepare_data
from sensor_context_encoder.models import (
    EXPECTED_CONTEXT_PARAMETERS,
    EXPECTED_DIRECT_PARAMETERS,
    DirectClassifier,
    FrozenContextClassifier,
    SensorEncoder,
    count_trainable_parameters,
    trainable_parameters,
)
from sensor_context_encoder.utils import (
    capture_rng_state,
    classification_metrics,
    cosine_warmup_multiplier,
    format_duration,
    move_to_cpu,
    resolve_device,
    restore_rng_state,
    set_seed,
    write_json,
)


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    data_dir: str
    seed: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    accumulation_steps: int
    num_workers: int
    device: str
    context_objectives: bool
    context_objective_names: tuple[str, ...]
    objective_warmup_epochs: int
    max_batches: int | None


def build_model(model_type: str, device: torch.device, seed: int) -> nn.Module:
    set_seed(seed)
    encoder = SensorEncoder()
    if model_type == "direct":
        model: nn.Module = DirectClassifier(encoder)
        expected = EXPECTED_DIRECT_PARAMETERS
    elif model_type == "context":
        model = FrozenContextClassifier(device=device, encoder=encoder)
        expected = EXPECTED_CONTEXT_PARAMETERS
    else:
        raise ValueError(f"Unknown model type {model_type!r}")
    model.to(device)
    actual = count_trainable_parameters(model)
    if actual != expected:
        raise AssertionError(f"Unexpected trainable parameter count: {actual:,} != {expected:,}")
    return model


def build_loaders(
    data: PreparedData,
    config: TrainingConfig,
    epoch: int = 1,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed + epoch - 1)
    train_loader = DataLoader(
        as_tensor_dataset(data.train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        as_tensor_dataset(data.validation),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    return train_loader, validation_loader


def verify_gradient_partition(model: nn.Module) -> None:
    if not isinstance(model, FrozenContextClassifier):
        return
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen backbone received gradients")
    for name, module in (
        ("encoder", model.encoder),
        ("projector", model.projector),
        ("head", model.head),
    ):
        if not any(parameter.grad is not None for parameter in module.parameters()):
            raise AssertionError(f"Trainable {name} did not receive gradients")


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    accumulation_steps: int,
    context_objectives: bool = False,
    view_augmenter: SensorViewAugmenter | None = None,
    objective_weights: ContextLossWeights | None = None,
    objective_scale: float = 1.0,
    max_batches: int | None = None,
    epoch: int = 1,
    total_epochs: int = 1,
    start_batch: int = 0,
    global_step: int = 0,
    elapsed_before: float = 0.0,
    run_started_at: float | None = None,
    accumulator: dict[str, Any] | None = None,
    checkpoint_every_steps: int = 1,
    checkpoint_callback: Callable[[int, int, float, float, dict[str, Any]], None]
    | None = None,
) -> tuple[dict[str, Any], int, float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    if run_started_at is None:
        run_started_at = time.perf_counter()
    if accumulator is None:
        accumulator = {
            "total_loss": 0.0,
            "total_classification_loss": 0.0,
            "auxiliary_loss_totals": {},
            "labels": [],
            "predictions": [],
        }
    auxiliary_loss_totals: dict[str, float] = accumulator["auxiliary_loss_totals"]
    batch_limit = min(len(loader), max_batches) if max_batches else len(loader)
    if start_batch < 0 or start_batch >= batch_limit:
        raise ValueError(f"Resume batch must be in [0, {batch_limit}), got {start_batch}")
    remainder = batch_limit % accumulation_steps
    total_batches = total_epochs * batch_limit
    progress = tqdm(
        total=batch_limit,
        initial=start_batch,
        desc=f"epoch {epoch}/{total_epochs}",
        leave=False,
    )
    eta_seconds = math.inf
    for batch_index, (features, labels) in enumerate(loader):
        if batch_index < start_batch:
            continue
        if batch_index >= batch_limit:
            break
        features = features.to(device)
        labels = labels.to(device)
        if context_objectives:
            if not isinstance(model, FrozenContextClassifier):
                raise TypeError("Context objectives require the frozen-LLM context model")
            if view_augmenter is None or objective_weights is None:
                raise ValueError("Context objectives must be configured")
            logits, representations, projected = model.forward_with_embeddings(features)
        else:
            logits = model(features)
            representations = projected = None
        classification_loss = criterion(logits, labels)
        loss = classification_loss
        if context_objectives:
            assert representations is not None and projected is not None
            auxiliary_total, auxiliary_losses = compute_context_losses(
                model,
                features,
                labels,
                representations,
                projected,
                view_augmenter,
                objective_weights,
            )
            loss = loss + objective_scale * auxiliary_total
            for name, value in auxiliary_losses.items():
                auxiliary_loss_totals[name] = auxiliary_loss_totals.get(name, 0.0) + float(
                    value.detach()
                ) * len(labels)
        accumulator["total_loss"] += float(loss.detach()) * len(labels)
        accumulator["total_classification_loss"] += float(classification_loss.detach()) * len(
            labels
        )

        in_final_group = remainder and batch_index >= batch_limit - remainder
        divisor = remainder if in_final_group else accumulation_steps
        (loss / divisor).backward()
        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batch_limit
        if should_step:
            if batch_index < accumulation_steps:
                verify_gradient_partition(model)
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        accumulator["labels"].extend(labels.detach().cpu().tolist())
        accumulator["predictions"].extend(logits.detach().argmax(dim=1).cpu().tolist())
        completed_batches = (epoch - 1) * batch_limit + batch_index + 1
        elapsed_seconds = elapsed_before + (time.perf_counter() - run_started_at)
        eta_seconds = (
            elapsed_seconds / completed_batches * (total_batches - completed_batches)
            if completed_batches
            else math.inf
        )
        progress.update(1)
        progress.set_postfix(
            loss=f"{float(loss.detach()):.3f}", eta=format_duration(eta_seconds)
        )
        next_batch = batch_index + 1
        if (
            should_step
            and checkpoint_callback is not None
            and global_step % checkpoint_every_steps == 0
            and next_batch < batch_limit
        ):
            checkpoint_callback(
                next_batch,
                global_step,
                elapsed_seconds,
                eta_seconds,
                accumulator,
            )

    progress.close()
    labels_array = np.asarray(accumulator["labels"], dtype=np.int64)
    predictions_array = np.asarray(accumulator["predictions"], dtype=np.int64)
    metrics = classification_metrics(labels_array, predictions_array)
    metrics["loss"] = accumulator["total_loss"] / len(labels_array)
    metrics["classification_loss"] = accumulator["total_classification_loss"] / len(labels_array)
    if auxiliary_loss_totals:
        metrics["auxiliary_losses"] = {
            name: value / len(labels_array) for name, value in auxiliary_loss_totals.items()
        }
        metrics["auxiliary_weight"] = objective_scale
    elapsed_seconds = elapsed_before + (time.perf_counter() - run_started_at)
    return metrics, global_step, elapsed_seconds, eta_seconds


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    labels_out: list[np.ndarray] = []
    predictions_out: list[np.ndarray] = []
    batch_limit = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, (features, labels) in enumerate(loader):
        if batch_index >= batch_limit:
            break
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, labels)) * len(labels)
        labels_out.append(labels.cpu().numpy())
        predictions_out.append(logits.argmax(dim=1).cpu().numpy())
    labels_array = np.concatenate(labels_out)
    predictions_array = np.concatenate(predictions_out)
    metrics = classification_metrics(labels_array, predictions_array)
    metrics["loss"] = total_loss / len(labels_array)
    return metrics


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def build_training_state(
    optimizer: AdamW,
    scheduler: LambdaLR,
    *,
    epoch: int,
    next_batch: int,
    global_step: int,
    best_macro_f1: float,
    best_epoch: int,
    epochs_without_improvement: int,
    elapsed_seconds: float,
    eta_seconds: float | None,
    epoch_accumulator: dict[str, Any] | None,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "epoch_accumulator": epoch_accumulator,
        "status": status,
        "optimizer": move_to_cpu(optimizer.state_dict()),
        "scheduler": move_to_cpu(scheduler.state_dict()),
        "rng_state": move_to_cpu(capture_rng_state()),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    data: PreparedData,
    history: list[dict[str, Any]],
    best_epoch: int,
    training_state: dict[str, Any],
    write_progress: bool = False,
) -> None:
    if isinstance(model, FrozenContextClassifier):
        state: dict[str, Any] = model.trainable_state_dict()
    else:
        state = cpu_state_dict(model)
    checkpoint = {
        "model_type": config.model,
        "state": state,
        "config": asdict(config),
        "normalization": {"mean": data.mean.tolist(), "std": data.std.tolist()},
        "validation_subjects": sorted(VALIDATION_SUBJECTS),
        "model_id": MODEL_ID if config.model == "context" else None,
        "model_revision": MODEL_REVISION if config.model == "context" else None,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "history": history,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "training_state": training_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    if write_progress:
        eta = training_state["eta_seconds"]
        write_json(
            path.parent / "progress.json",
            {
                "status": training_state["status"],
                "epoch": training_state["epoch"],
                "next_batch": training_state["next_batch"],
                "global_step": training_state["global_step"],
                "best_epoch": training_state["best_epoch"],
                "best_macro_f1": training_state["best_macro_f1"],
                "elapsed_seconds": training_state["elapsed_seconds"],
                "elapsed": format_duration(training_state["elapsed_seconds"]),
                "eta_seconds": eta if eta is None or math.isfinite(eta) else None,
                "eta": format_duration(eta),
                "resumable_from": str(path),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )


def load_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    if isinstance(model, FrozenContextClassifier):
        model.load_trainable_state_dict(state)
    else:
        model.load_state_dict(state)


def optimizer_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    config: TrainingConfig,
    data: PreparedData,
) -> None:
    saved_config = dict(checkpoint.get("config", {}))
    if "context_objectives" not in saved_config:
        saved_config["context_objectives"] = bool(saved_config.get("frontier", False))
    saved_config.setdefault("objective_warmup_epochs", 0)
    required_matches = (
        "model",
        "seed",
        "epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "accumulation_steps",
        "context_objectives",
        "objective_warmup_epochs",
        "max_batches",
    )
    mismatches = [
        key
        for key in required_matches
        if saved_config.get(key) != asdict(config).get(key)
    ]
    if mismatches:
        raise ValueError(f"Resume configuration differs for: {', '.join(mismatches)}")
    if "training_state" not in checkpoint:
        raise ValueError("Checkpoint predates resumable training state; use it for evaluation only")
    normalization = checkpoint.get("normalization", {})
    if not np.allclose(data.mean, normalization.get("mean")) or not np.allclose(
        data.std, normalization.get("std")
    ):
        raise ValueError("Resume checkpoint normalization does not match the current data")


def train(args: argparse.Namespace) -> Path:
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = dict(resume_checkpoint.get("config", {}))
        if "context_objectives" not in saved:
            saved["context_objectives"] = bool(saved.get("frontier", False))
        saved.setdefault("objective_warmup_epochs", 0)
        for name in (
            "model",
            "seed",
            "epochs",
            "patience",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "accumulation_steps",
            "num_workers",
            "context_objectives",
            "objective_warmup_epochs",
            "max_batches",
        ):
            if name not in saved:
                raise ValueError(f"Resume checkpoint is missing config field {name!r}")
            setattr(args, name, saved[name])
        if "data_dir" in saved:
            args.data_dir = Path(saved["data_dir"])
    elif args.model is None:
        raise ValueError("--model is required unless --resume is provided")

    if args.context_objectives is None:
        args.context_objectives = args.model == "context"

    device = resolve_device(args.device)
    if args.context_objectives and args.model != "context":
        raise ValueError("Context objectives are available only for --model context")
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
        print("Using MPS: float32 trainable modules with a float16 frozen backbone")
    batch_size = args.batch_size or (128 if args.model == "direct" else 16)
    accumulation_steps = args.accumulation_steps or (1 if args.model == "direct" else 8)
    checkpoint_every_steps = args.checkpoint_every_steps or (
        25 if args.model == "direct" else 1
    )
    if checkpoint_every_steps < 1:
        raise ValueError("--checkpoint-every-steps must be positive")
    if args.objective_warmup_epochs < 0:
        raise ValueError("--objective-warmup-epochs must be non-negative")
    config = TrainingConfig(
        model=args.model,
        data_dir=str(args.data_dir.resolve()),
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_workers=args.num_workers,
        device=str(device),
        context_objectives=args.context_objectives,
        context_objective_names=CONTEXT_OBJECTIVE_NAMES if args.context_objectives else (),
        objective_warmup_epochs=args.objective_warmup_epochs,
        max_batches=args.max_batches,
    )
    set_seed(config.seed)
    data = prepare_data(args.data_dir)
    sizing_loader, _ = build_loaders(data, config, epoch=1)
    model = build_model(config.model, device, config.seed)
    optimizer = AdamW(
        trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches_per_epoch = (
        min(len(sizing_loader), args.max_batches) if args.max_batches else len(sizing_loader)
    )
    steps_per_epoch = math.ceil(batches_per_epoch / config.accumulation_steps)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = max(1, math.ceil(total_steps * 0.05))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(
            cosine_warmup_multiplier,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )

    if args.resume is not None and args.output_dir is None:
        output_dir = args.resume.parent
    else:
        output_dir = args.output_dir or Path("runs") / config.model
    best_checkpoint_path = output_dir / "best.pt"
    latest_checkpoint_path = output_dir / "latest.pt"
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    start_batch = 0
    global_step = 0
    elapsed_before = 0.0
    epoch_accumulator: dict[str, Any] | None = None
    if args.resume is not None:
        assert resume_checkpoint is not None
        validate_resume_checkpoint(resume_checkpoint, config, data)
        load_model_state(model, resume_checkpoint["state"])
        training_state = resume_checkpoint["training_state"]
        optimizer.load_state_dict(training_state["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(training_state["scheduler"])
        start_epoch = int(training_state["epoch"])
        start_batch = int(training_state["next_batch"])
        global_step = int(training_state["global_step"])
        best_macro_f1 = float(training_state["best_macro_f1"])
        best_epoch = int(training_state["best_epoch"])
        epochs_without_improvement = int(training_state["epochs_without_improvement"])
        elapsed_before = float(training_state["elapsed_seconds"])
        epoch_accumulator = training_state["epoch_accumulator"]
        history = resume_checkpoint["history"]
        restore_rng_state(training_state["rng_state"])
        print(
            f"Resuming {args.resume} at epoch {start_epoch}, batch {start_batch}, "
            f"global step {global_step}; elapsed {format_duration(elapsed_before)}"
        )
    else:
        initial_state = build_training_state(
            optimizer,
            scheduler,
            epoch=1,
            next_batch=0,
            global_step=0,
            best_macro_f1=best_macro_f1,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            elapsed_seconds=0.0,
            eta_seconds=None,
            epoch_accumulator=None,
        )
        save_checkpoint(
            latest_checkpoint_path,
            model,
            config,
            data,
            history,
            best_epoch,
            initial_state,
            write_progress=True,
        )

    if start_epoch > config.epochs:
        print(f"Training is already complete. Best checkpoint: {best_checkpoint_path}")
        if best_checkpoint_path.exists():
            return best_checkpoint_path
        assert args.resume is not None
        return args.resume

    view_augmenter = SensorViewAugmenter() if config.context_objectives else None
    objective_weights = ContextLossWeights() if config.context_objectives else None
    run_started_at = time.perf_counter()
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_loader, validation_loader = build_loaders(data, config, epoch=epoch)
            resume_batch = start_batch if epoch == start_epoch else 0
            resume_accumulator = epoch_accumulator if epoch == start_epoch else None
            objective_scale = (
                min(1.0, epoch / max(1, config.objective_warmup_epochs))
                if config.context_objectives
                else 0.0
            )

            def checkpoint_during_epoch(
                next_batch: int,
                current_global_step: int,
                elapsed_seconds: float,
                eta_seconds: float,
                accumulator: dict[str, Any],
                current_epoch: int = epoch,
                current_best_macro_f1: float = best_macro_f1,
                current_best_epoch: int = best_epoch,
                current_epochs_without_improvement: int = epochs_without_improvement,
            ) -> None:
                state = build_training_state(
                    optimizer,
                    scheduler,
                    epoch=current_epoch,
                    next_batch=next_batch,
                    global_step=current_global_step,
                    best_macro_f1=current_best_macro_f1,
                    best_epoch=current_best_epoch,
                    epochs_without_improvement=current_epochs_without_improvement,
                    elapsed_seconds=elapsed_seconds,
                    eta_seconds=eta_seconds,
                    epoch_accumulator=accumulator,
                )
                save_checkpoint(
                    latest_checkpoint_path,
                    model,
                    config,
                    data,
                    history,
                    current_best_epoch,
                    state,
                    write_progress=True,
                )

            training_metrics, global_step, _, _ = run_training_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config.accumulation_steps,
                context_objectives=config.context_objectives,
                view_augmenter=view_augmenter,
                objective_weights=objective_weights,
                objective_scale=objective_scale,
                max_batches=args.max_batches,
                epoch=epoch,
                total_epochs=config.epochs,
                start_batch=resume_batch,
                global_step=global_step,
                elapsed_before=elapsed_before,
                run_started_at=run_started_at,
                accumulator=resume_accumulator,
                checkpoint_every_steps=checkpoint_every_steps,
                checkpoint_callback=checkpoint_during_epoch,
            )
            validation_metrics = evaluate_loader(
                model, validation_loader, device, max_batches=args.max_batches
            )
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": training_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            improved = validation_metrics["macro_f1"] > best_macro_f1 + 1e-4
            if improved:
                best_macro_f1 = validation_metrics["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            elapsed_seconds = elapsed_before + (time.perf_counter() - run_started_at)
            completed_batches = epoch * batches_per_epoch
            total_batches = config.epochs * batches_per_epoch
            eta_seconds = elapsed_seconds / completed_batches * (
                total_batches - completed_batches
            )
            early_stopped = epochs_without_improvement >= config.patience
            status = (
                "early_stopped"
                if early_stopped
                else "complete"
                if epoch == config.epochs
                else "running"
            )
            state = build_training_state(
                optimizer,
                scheduler,
                epoch=epoch + 1,
                next_batch=0,
                global_step=global_step,
                best_macro_f1=best_macro_f1,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                elapsed_seconds=elapsed_seconds,
                eta_seconds=0.0 if status != "running" else eta_seconds,
                epoch_accumulator=None,
                status=status,
            )
            if improved:
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    config,
                    data,
                    history,
                    best_epoch,
                    state,
                )
            save_checkpoint(
                latest_checkpoint_path,
                model,
                config,
                data,
                history,
                best_epoch,
                state,
                write_progress=True,
            )
            print(
                f"epoch={epoch} train_f1={training_metrics['macro_f1']:.4f} "
                f"validation_f1={validation_metrics['macro_f1']:.4f} "
                f"elapsed={format_duration(elapsed_seconds)} "
                f"max_epoch_eta={format_duration(0.0 if status != 'running' else eta_seconds)}"
            )
            start_batch = 0
            epoch_accumulator = None
            if early_stopped:
                print(f"Early stopping after epoch {epoch}")
                break
    except KeyboardInterrupt:
        print(
            f"Interrupted. Resume from the last atomic checkpoint with: "
            f"python train.py --resume {latest_checkpoint_path} --device {device.type}"
        )
        raise

    print(
        f"Best checkpoint: {best_checkpoint_path} "
        f"(epoch {best_epoch}, macro-F1 {best_macro_f1:.4f})"
    )
    return best_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("direct", "context"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--with-context-objectives",
        action="store_true",
        dest="context_objectives",
        help="Enable the six auxiliary context objectives as a research ablation",
    )
    parser.set_defaults(context_objectives=False)
    parser.add_argument(
        "--objective-warmup-epochs",
        type=int,
        default=3,
        help="Ramp standard context-objective weights over this many epochs (default: 3)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Limit each epoch for a smoke run; do not use for reportable results",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Atomic latest-checkpoint cadence in optimizer steps (default: context 1, direct 25)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume exactly from latest.pt; saved training settings are restored automatically",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
