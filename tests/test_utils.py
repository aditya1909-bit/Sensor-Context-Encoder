import numpy as np
import pytest
import torch

from sensor_context_encoder.utils import (
    capture_rng_state,
    classification_metrics,
    make_derangement,
    restore_rng_state,
)


def test_derangement_is_deterministic_and_has_no_fixed_rows() -> None:
    first = make_derangement(100, seed=43)
    second = make_derangement(100, seed=43)
    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(100))
    assert sorted(first.tolist()) == list(range(100))

    embeddings = np.arange(100 * 4).reshape(100, 4)
    shuffled = embeddings[first]
    for row in shuffled:
        assert any(np.array_equal(row, original) for original in embeddings)


def test_derangement_rejects_one_example() -> None:
    with pytest.raises(ValueError):
        make_derangement(1, seed=43)


def test_macro_f1_and_confusion_matrix() -> None:
    labels = np.arange(6)
    metrics = classification_metrics(labels, labels.copy())
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert np.array(metrics["confusion_matrix"]).trace() == 6


def test_rng_state_round_trip() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    state = capture_rng_state()
    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)
    restore_rng_state(state)
    torch.testing.assert_close(torch.rand(4), expected_torch)
    np.testing.assert_allclose(np.random.rand(4), expected_numpy)
