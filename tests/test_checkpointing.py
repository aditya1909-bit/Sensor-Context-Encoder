from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sensor_context_encoder.data import ArraySplit, PreparedData
from sensor_context_encoder.models import DirectClassifier
from train import TrainingConfig, build_training_state, save_checkpoint


def make_data() -> PreparedData:
    split = ArraySplit(
        features=np.zeros((2, 128, 9), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        subjects=np.array([1, 2], dtype=np.int64),
    )
    return PreparedData(
        train=split,
        validation=split,
        test=split,
        mean=np.zeros(9, dtype=np.float32),
        std=np.ones(9, dtype=np.float32),
    )


def test_atomic_checkpoint_contains_complete_resume_state(tmp_path: Path) -> None:
    model = DirectClassifier()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    config = TrainingConfig(
        model="direct",
        data_dir=str(tmp_path),
        seed=42,
        epochs=30,
        patience=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        accumulation_steps=1,
        num_workers=0,
        device="cpu",
        context_objectives=False,
        context_objective_names=(),
        objective_warmup_epochs=3,
        max_batches=None,
    )
    accumulator = {
        "total_loss": 1.5,
        "total_classification_loss": 1.2,
        "auxiliary_loss_totals": {},
        "labels": [0, 1],
        "predictions": [0, 1],
    }
    state = build_training_state(
        optimizer,
        scheduler,
        epoch=3,
        next_batch=16,
        global_step=100,
        best_macro_f1=0.8,
        best_epoch=2,
        epochs_without_improvement=1,
        elapsed_seconds=900.0,
        eta_seconds=3_600.0,
        epoch_accumulator=accumulator,
    )
    checkpoint_path = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config,
        make_data(),
        history=[],
        best_epoch=2,
        training_state=state,
        write_progress=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["training_state"]["next_batch"] == 16
    assert checkpoint["training_state"]["optimizer"]
    assert checkpoint["training_state"]["scheduler"]
    assert checkpoint["training_state"]["rng_state"]
    assert checkpoint["training_state"]["epoch_accumulator"] == accumulator
    assert not (tmp_path / "latest.pt.tmp").exists()

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "running"
    assert progress["eta"] == "1h 00m"
    assert progress["resumable_from"] == str(checkpoint_path)
