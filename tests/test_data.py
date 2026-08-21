from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensor_context_encoder.constants import SIGNAL_NAMES
from sensor_context_encoder.data import (
    ArraySplit,
    load_official_split,
    split_training_subjects,
    standardize_splits,
)


def write_split(root: Path, split: str, width: int = 128) -> None:
    signal_dir = root / split / "Inertial Signals"
    signal_dir.mkdir(parents=True)
    samples = 3
    for index, signal in enumerate(SIGNAL_NAMES):
        values = np.full((samples, width), index, dtype=np.float32)
        np.savetxt(signal_dir / f"{signal}_{split}.txt", values)
    np.savetxt(root / split / f"y_{split}.txt", np.array([1, 2, 6]), fmt="%d")
    np.savetxt(root / split / f"subject_{split}.txt", np.array([1, 2, 3]), fmt="%d")


def test_loads_only_ordered_inertial_signals(tmp_path: Path) -> None:
    write_split(tmp_path, "train")
    split = load_official_split(tmp_path, "train")
    assert split.features.shape == (3, 128, 9)
    assert split.features.dtype == np.float32
    assert split.labels.tolist() == [0, 1, 5]
    for index in range(9):
        assert np.all(split.features[:, :, index] == index)


def test_rejects_wrong_window_width(tmp_path: Path) -> None:
    write_split(tmp_path, "test", width=127)
    with pytest.raises(ValueError, match="128"):
        load_official_split(tmp_path, "test")


def test_subject_split_is_disjoint() -> None:
    features = np.zeros((6, 128, 9), dtype=np.float32)
    official = ArraySplit(
        features=features,
        labels=np.arange(6),
        subjects=np.array([1, 1, 2, 2, 3, 3]),
    )
    train, validation = split_training_subjects(official, frozenset({2}))
    assert set(train.subjects) == {1, 3}
    assert set(validation.subjects) == {2}
    assert not set(train.subjects) & set(validation.subjects)


def test_normalization_uses_training_statistics_only() -> None:
    generator = np.random.default_rng(7)
    train_features = generator.normal(4.0, 2.0, size=(5, 128, 9)).astype(np.float32)
    validation_features = np.full((2, 128, 9), 100.0, dtype=np.float32)
    test_features = np.full((2, 128, 9), -100.0, dtype=np.float32)

    def split(features: np.ndarray) -> ArraySplit:
        return ArraySplit(features, np.zeros(len(features), dtype=np.int64), np.arange(len(features)))

    prepared = standardize_splits(
        split(train_features), split(validation_features), split(test_features)
    )
    np.testing.assert_allclose(prepared.mean, train_features.mean(axis=(0, 1)), rtol=1e-5)
    np.testing.assert_allclose(prepared.train.features.mean(axis=(0, 1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(prepared.train.features.std(axis=(0, 1)), 1.0, atol=1e-5)
    assert prepared.validation.features.mean() > 1.0
    assert prepared.test.features.mean() < -1.0
