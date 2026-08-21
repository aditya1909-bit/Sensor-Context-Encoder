"""Tests for context-training auxiliary objectives."""

from __future__ import annotations

import torch

from sensor_context_encoder.context_objectives import (
    CONTEXT_OBJECTIVE_NAMES,
    ContextLossWeights,
    SensorViewAugmenter,
    compute_context_losses,
    relational_semantic_distillation_loss,
    supervised_contrastive_loss,
    text_prototype_alignment_loss,
    token_manifold_matching_loss,
    vicreg_loss,
    visreg_loss,
)
from tests.test_models import make_context_model


def test_so3_views_preserve_vector_norms_without_other_augmentations() -> None:
    augmenter = SensorViewAugmenter(
        max_rotation_degrees=30.0,
        temporal_mask_length=0,
        jitter_std=0.0,
        scale_range=0.0,
    )
    inputs = torch.randn(4, 128, 9)
    augmented = augmenter.augment(inputs)
    original_norms = inputs.reshape(4, 128, 3, 3).norm(dim=-1)
    augmented_norms = augmented.reshape(4, 128, 3, 3).norm(dim=-1)
    torch.testing.assert_close(augmented_norms, original_norms, rtol=1e-5, atol=1e-5)


def test_individual_context_losses_are_finite() -> None:
    first = torch.randn(8, 256, requires_grad=True)
    second = torch.randn(8, 256, requires_grad=True)
    projected = torch.randn(8, 960, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    prototypes = torch.randn(6, 960)
    losses = (
        vicreg_loss(first, second),
        visreg_loss(first, second, num_projections=8),
        supervised_contrastive_loss(first, labels),
        text_prototype_alignment_loss(projected, labels, prototypes),
        relational_semantic_distillation_loss(projected, labels, prototypes),
        token_manifold_matching_loss(
            projected,
            torch.zeros(960),
            torch.full((960,), 0.02),
            torch.tensor(0.6),
        ),
    )
    total = sum(losses)
    assert all(torch.isfinite(loss) for loss in losses)
    total.backward()
    assert first.grad is not None
    assert projected.grad is not None


def test_combined_context_objective_updates_only_trainable_modules() -> None:
    torch.manual_seed(42)
    model = make_context_model()
    model.train()
    inputs = torch.randn(6, 128, 9)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits, representations, projected = model.forward_with_embeddings(inputs)
    auxiliary_total, losses = compute_context_losses(
        model,
        inputs,
        labels,
        representations,
        projected,
        SensorViewAugmenter(),
        ContextLossWeights(),
    )
    (torch.nn.functional.cross_entropy(logits, labels) + auxiliary_total).backward()
    assert tuple(losses) == CONTEXT_OBJECTIVE_NAMES[1:]
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.projector.parameters())
    assert any(parameter.grad is not None for parameter in model.head.parameters())
