"""UCI HAR inertial-signal loading and subject-wise splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .constants import (
    EXPECTED_TEST_SUBJECTS,
    SIGNAL_NAMES,
    VALIDATION_SUBJECTS,
    dataset_root,
)


@dataclass(frozen=True)
class ArraySplit:
    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    train: ArraySplit
    validation: ArraySplit
    test: ArraySplit
    mean: np.ndarray
    std: np.ndarray


def load_official_split(root: str | Path, split: str) -> ArraySplit:
    """Load only the nine windowed inertial signals for one official split."""

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    root = Path(root)
    signal_dir = root / split / "Inertial Signals"
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for signal_name in SIGNAL_NAMES:
        path = signal_dir / f"{signal_name}_{split}.txt"
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 128:
            raise ValueError(f"{path} must have shape [samples, 128], got {values.shape}")
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(
                f"All inertial signals must share one shape; {path} has {values.shape}, "
                f"expected {expected_shape}"
            )
        arrays.append(values)

    features = np.stack(arrays, axis=-1)
    labels = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64).reshape(-1) - 1
    subjects = np.loadtxt(
        root / split / f"subject_{split}.txt", dtype=np.int64
    ).reshape(-1)

    if len(features) != len(labels) or len(features) != len(subjects):
        raise ValueError("Feature, label, and subject counts do not match")
    if not np.isfinite(features).all():
        raise ValueError("Inertial signals contain non-finite values")
    if not np.isin(labels, np.arange(6)).all():
        raise ValueError("Labels must map from UCI values 1..6 to zero-based values 0..5")

    return ArraySplit(features=features, labels=labels, subjects=subjects)


def split_training_subjects(
    official_train: ArraySplit,
    validation_subjects: frozenset[int] = VALIDATION_SUBJECTS,
) -> tuple[ArraySplit, ArraySplit]:
    """Split official training data without allowing a subject to cross partitions."""

    validation_mask = np.isin(official_train.subjects, tuple(validation_subjects))
    train_mask = ~validation_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Both training and validation partitions must be non-empty")

    def select(mask: np.ndarray) -> ArraySplit:
        return ArraySplit(
            features=official_train.features[mask],
            labels=official_train.labels[mask],
            subjects=official_train.subjects[mask],
        )

    train = select(train_mask)
    validation = select(validation_mask)
    if set(train.subjects) & set(validation.subjects):
        raise AssertionError("Training and validation subjects overlap")
    return train, validation


def standardize_splits(
    train: ArraySplit,
    validation: ArraySplit,
    test: ArraySplit,
) -> PreparedData:
    """Standardize all partitions with statistics computed from training only."""

    mean = train.features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    if np.any(std < 1e-8):
        raise ValueError("At least one training signal has near-zero variance")

    def normalize(split: ArraySplit) -> ArraySplit:
        features = ((split.features - mean) / std).astype(np.float32, copy=False)
        return ArraySplit(features=features, labels=split.labels, subjects=split.subjects)

    return PreparedData(
        train=normalize(train),
        validation=normalize(validation),
        test=normalize(test),
        mean=mean,
        std=std,
    )


def prepare_data(data_dir: str | Path, validate_official: bool = True) -> PreparedData:
    """Load, validate, subject-split, and normalize the complete experiment data."""

    root = dataset_root(data_dir)
    official_train = load_official_split(root, "train")
    test = load_official_split(root, "test")
    train, validation = split_training_subjects(official_train)

    if validate_official:
        if len(official_train.features) != 7_352 or len(test.features) != 2_947:
            raise ValueError("Unexpected official UCI HAR split sizes")
        if len(train.features) != 5_551 or len(validation.features) != 1_801:
            raise ValueError("Unexpected subject-wise training/validation split sizes")
        if set(test.subjects) != EXPECTED_TEST_SUBJECTS:
            raise ValueError("Official test subject IDs do not match the expected split")
        if set(train.subjects) & set(test.subjects) or set(validation.subjects) & set(
            test.subjects
        ):
            raise ValueError("Test subjects overlap training or validation")
        for name, split in (("train", train), ("validation", validation), ("test", test)):
            if set(np.unique(split.labels)) != set(range(6)):
                raise ValueError(f"{name} does not contain all six activity classes")

    return standardize_splits(train, validation, test)


def as_tensor_dataset(split: ArraySplit) -> TensorDataset:
    """Convert a prepared array split into a PyTorch dataset."""

    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(split.features)),
        torch.from_numpy(np.ascontiguousarray(split.labels)),
    )
