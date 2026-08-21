"""Auxiliary objectives for training continuous sensor context embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .models import FrozenContextClassifier


CONTEXT_OBJECTIVE_NAMES = (
    "physics_aware_so3_views",
    "visreg_sliced_wasserstein",
    "supervised_contrastive",
    "text_prototype_alignment",
    "relational_semantic_distillation",
    "token_manifold_matching",
)


@dataclass(frozen=True)
class ContextLossWeights:
    visreg: float = 0.02
    supervised_contrastive: float = 0.10
    text_prototype: float = 0.10
    relational: float = 0.05
    token_manifold: float = 0.01


class SensorViewAugmenter:
    """Create two views with shared SO(3) transforms across physical vector triads."""

    def __init__(
        self,
        max_rotation_degrees: float = 15.0,
        temporal_mask_length: int = 16,
        jitter_std: float = 0.02,
        scale_range: float = 0.10,
    ) -> None:
        self.max_rotation_radians = torch.deg2rad(torch.tensor(max_rotation_degrees)).item()
        self.temporal_mask_length = temporal_mask_length
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.augment(inputs), self.augment(inputs)

    def augment(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (128, 9):
            raise ValueError(f"Sensor input must have shape [batch, 128, 9], got {inputs.shape}")
        batch_size = inputs.shape[0]
        device = inputs.device
        dtype = inputs.dtype

        axes = torch.randn(batch_size, 3, device=device, dtype=dtype)
        axes = F.normalize(axes, dim=-1)
        angles = (
            torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0
        ) * self.max_rotation_radians
        rotations = _rodrigues_rotation_matrices(axes, angles)
        grouped = inputs.reshape(batch_size, 128, 3, 3)
        augmented = torch.einsum("bij,btgj->btgi", rotations, grouped).reshape_as(inputs)

        if self.scale_range:
            scales = 1.0 + (
                torch.rand(batch_size, 1, 3, 1, device=device, dtype=dtype) * 2.0 - 1.0
            ) * self.scale_range
            augmented = (augmented.reshape(batch_size, 128, 3, 3) * scales).reshape_as(inputs)
        if self.jitter_std:
            augmented = augmented + torch.randn_like(augmented) * self.jitter_std
        if self.temporal_mask_length:
            starts = torch.randint(
                0,
                128 - self.temporal_mask_length + 1,
                (batch_size, 1),
                device=device,
            )
            positions = torch.arange(128, device=device).unsqueeze(0)
            mask = (positions >= starts) & (positions < starts + self.temporal_mask_length)
            augmented = augmented.masked_fill(mask.unsqueeze(-1), 0.0)
        return augmented


def _rodrigues_rotation_matrices(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    x, y, z = axes.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(-1, 3, 3)
    identity = torch.eye(3, device=axes.device, dtype=axes.dtype).expand(len(axes), -1, -1)
    sine = angles.sin().reshape(-1, 1, 1)
    cosine = angles.cos().reshape(-1, 1, 1)
    return identity + sine * skew + (1.0 - cosine) * torch.bmm(skew, skew)


def vicreg_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """VICReg invariance, variance, and covariance loss with standard coefficients."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg views must be equally shaped [batch, features] tensors")
    if len(first) < 2:
        return first.sum() * 0.0
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
    )
    denominator = max(len(first) - 1, 1)
    first_covariance = first_centered.T @ first_centered / denominator
    second_covariance = second_centered.T @ second_centered / denominator
    covariance = _off_diagonal(first_covariance).pow(2).mean()
    covariance = covariance + _off_diagonal(second_covariance).pow(2).mean()
    return 25.0 * invariance + 25.0 * variance + covariance


def visreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    num_projections: int = 64,
) -> torch.Tensor:
    """VISReg invariance plus sliced-Wasserstein scale/shape regularization."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VISReg views must be equally shaped [batch, features] tensors")
    if len(first) < 2:
        return first.sum() * 0.0
    invariance = F.mse_loss(first, second)
    regularization = 0.5 * (
        _visreg_regularization(first, num_projections)
        + _visreg_regularization(second, num_projections)
    )
    return invariance + regularization


def _visreg_regularization(
    representations: torch.Tensor,
    num_projections: int,
) -> torch.Tensor:
    mean = representations.mean(dim=0)
    centered = representations - mean
    std = centered.std(dim=0, unbiased=False)
    center_loss = mean.pow(2).mean()
    scale_loss = (1.0 - std).pow(2).mean()
    normalized = centered / std.detach().clamp_min(1e-4)

    directions = torch.randn(
        representations.shape[1],
        num_projections,
        device=representations.device,
        dtype=representations.dtype,
    )
    directions = F.normalize(directions, dim=0)
    sorted_projections = torch.sort(normalized @ directions, dim=0).values
    normal = NormalDist()
    quantiles = torch.tensor(
        [normal.inv_cdf((index + 1) / (len(representations) + 1)) for index in range(len(representations))],
        device=representations.device,
        dtype=representations.dtype,
    ).unsqueeze(1)
    shape_loss = (sorted_projections - quantiles).pow(2).mean()
    return center_loss + scale_loss + shape_loss


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def supervised_contrastive_loss(
    representations: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Pull same-activity sensor representations together and separate other classes."""

    normalized = F.normalize(representations, dim=-1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid_anchors = positive_mask.any(dim=1)
    if not valid_anchors.any():
        return representations.sum() * 0.0
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (
        log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_counts
    )
    return -mean_positive_log_probability[valid_anchors].mean()


def text_prototype_alignment_loss(
    projected: torch.Tensor,
    labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Align each continuous context with its frozen activity-text prototype."""

    logits = F.normalize(projected, dim=-1) @ F.normalize(text_prototypes, dim=-1).T
    return F.cross_entropy(logits / temperature, labels)


def relational_semantic_distillation_loss(
    projected: torch.Tensor,
    labels: torch.Tensor,
    text_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Match pairwise sensor-context geometry to frozen text-prototype geometry."""

    if len(projected) < 2:
        return projected.sum() * 0.0
    sensor_similarity = F.normalize(projected, dim=-1) @ F.normalize(projected, dim=-1).T
    label_prototypes = F.normalize(text_prototypes, dim=-1)[labels]
    semantic_similarity = label_prototypes @ label_prototypes.T
    mask = ~torch.eye(len(projected), dtype=torch.bool, device=projected.device)
    return F.smooth_l1_loss(sensor_similarity[mask], semantic_similarity[mask])


def token_manifold_matching_loss(
    projected: torch.Tensor,
    token_mean: torch.Tensor,
    token_std: torch.Tensor,
    token_norm_mean: torch.Tensor,
) -> torch.Tensor:
    """Match first/second moments and norm scale of frozen vocabulary embeddings."""

    projected_mean = projected.mean(dim=0)
    projected_std = projected.std(dim=0, unbiased=False)
    mean_loss = F.mse_loss(projected_mean, token_mean)
    std_loss = F.mse_loss(projected_std, token_std)
    projected_norm = projected.norm(dim=-1).mean().clamp_min(1e-6)
    norm_loss = (
        projected_norm.log() - token_norm_mean.to(projected_norm.dtype).clamp_min(1e-6).log()
    ).pow(2)
    return mean_loss + std_loss + norm_loss


def compute_context_losses(
    model: FrozenContextClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    representations: torch.Tensor,
    projected: torch.Tensor,
    views: SensorViewAugmenter,
    weights: ContextLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute auxiliary context losses without another language-model pass."""

    first_view, second_view = views(inputs)
    paired_representations = model.encoder(torch.cat((first_view, second_view), dim=0))
    first_representation, second_representation = paired_representations.chunk(2, dim=0)
    losses = {
        "visreg_sliced_wasserstein": visreg_loss(
            first_representation, second_representation
        ),
        "supervised_contrastive": supervised_contrastive_loss(representations, labels),
        "text_prototype_alignment": text_prototype_alignment_loss(
            projected, labels, model.activity_prototypes
        ),
        "relational_semantic_distillation": relational_semantic_distillation_loss(
            projected, labels, model.activity_prototypes
        ),
        "token_manifold_matching": token_manifold_matching_loss(
            projected,
            model.token_embedding_mean,
            model.token_embedding_std,
            model.token_embedding_norm_mean,
        ),
    }
    total = (
        weights.visreg * losses["visreg_sliced_wasserstein"]
        + weights.supervised_contrastive * losses["supervised_contrastive"]
        + weights.text_prototype * losses["text_prototype_alignment"]
        + weights.relational * losses["relational_semantic_distillation"]
        + weights.token_manifold * losses["token_manifold_matching"]
    )
    return total, losses
