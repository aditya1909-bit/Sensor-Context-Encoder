from __future__ import annotations

import os

import pytest
import torch

from sensor_context_encoder.models import FrozenContextClassifier


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to download and test the pinned backbone",
)
def test_real_backbone_one_batch_gradient_partition() -> None:
    model = FrozenContextClassifier(device=torch.device("cpu"))
    model.train()
    loss = model(torch.randn(1, 128, 9)).sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.projector.parameters())
    assert any(parameter.grad is not None for parameter in model.head.parameters())
