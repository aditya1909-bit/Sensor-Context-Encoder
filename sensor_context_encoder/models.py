"""Shared sensor encoder and direct/frozen-language-model classifiers."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .constants import (
    ACTIVITY_NAMES,
    MODEL_HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
)

MAX_TRAINABLE_PARAMETERS = 10_000_000
EXPECTED_DIRECT_PARAMETERS = 1_091_398
EXPECTED_CONTEXT_PARAMETERS = 1_721_606


class ResidualBlock1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class SensorEncoder(nn.Module):
    output_size = 256

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 128),
            ResidualBlock1D(128, 256),
            ResidualBlock1D(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.normalization = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (128, 9):
            raise ValueError(f"Sensor input must have shape [batch, 128, 9], got {inputs.shape}")
        features = inputs.transpose(1, 2)
        features = self.blocks(self.stem(features))
        features = self.pool(features).squeeze(-1)
        return self.dropout(self.normalization(features))


class DirectClassifier(nn.Module):
    def __init__(self, encoder: SensorEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or SensorEncoder()
        self.head = nn.Linear(self.encoder.output_size, 6)
        assert_trainable_parameter_limit(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


class ContextProjector(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(SensorEncoder.output_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, MODEL_HIDDEN_SIZE),
            nn.LayerNorm(MODEL_HIDDEN_SIZE),
        )


class FrozenContextClassifier(nn.Module):
    """Insert one learned sensor embedding into a frozen causal transformer."""

    def __init__(
        self,
        device: torch.device,
        encoder: SensorEncoder | None = None,
        backbone: nn.Module | None = None,
        tokenizer: Any | None = None,
        backbone_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or SensorEncoder()
        self.projector = ContextProjector()
        self.head = nn.Linear(MODEL_HIDDEN_SIZE, 6)

        if backbone is None or tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            if backbone_dtype is None:
                backbone_dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
            backbone = AutoModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=backbone_dtype,
            )

        self.backbone = backbone.to(device)
        hidden_size = int(self.backbone.config.hidden_size)
        if hidden_size != MODEL_HIDDEN_SIZE:
            raise ValueError(f"Expected backbone hidden size 960, got {hidden_size}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        prefix_ids = self._token_ids(tokenizer, PROMPT_PREFIX, include_bos=True, device=device)
        suffix_ids = self._token_ids(tokenizer, PROMPT_SUFFIX, include_bos=False, device=device)
        embedding_table = self.backbone.get_input_embeddings()
        with torch.no_grad():
            prefix_embeddings = embedding_table(prefix_ids).detach()
            suffix_embeddings = embedding_table(suffix_ids).detach()
            activity_prototypes = torch.stack(
                [
                    embedding_table(
                        self._token_ids(
                            tokenizer, activity, include_bos=False, device=device
                        )
                    ).mean(dim=1).squeeze(0)
                    for activity in ACTIVITY_NAMES
                ]
            ).detach().float()
            embedding_weights = embedding_table.weight.detach().float()
            token_embedding_mean = embedding_weights.mean(dim=0)
            token_embedding_std = embedding_weights.std(dim=0, unbiased=False)
            token_embedding_norm_mean = embedding_weights.norm(dim=-1).mean()
        self.register_buffer("prefix_embeddings", prefix_embeddings, persistent=False)
        self.register_buffer("suffix_embeddings", suffix_embeddings, persistent=False)
        sequence_length = prefix_embeddings.shape[1] + 1 + suffix_embeddings.shape[1]
        self.register_buffer(
            "attention_mask_template",
            torch.ones(1, sequence_length, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer("activity_prototypes", activity_prototypes, persistent=False)
        self.register_buffer("token_embedding_mean", token_embedding_mean, persistent=False)
        self.register_buffer("token_embedding_std", token_embedding_std, persistent=False)
        self.register_buffer(
            "token_embedding_norm_mean", token_embedding_norm_mean, persistent=False
        )
        assert_trainable_parameter_limit(self, exclude_prefix="backbone.")

    @staticmethod
    def _token_ids(tokenizer: Any, text: str, include_bos: bool, device: torch.device) -> torch.Tensor:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if include_bos:
            if tokenizer.bos_token_id is None:
                raise ValueError("Tokenizer must define a BOS token")
            ids.insert(0, int(tokenizer.bos_token_id))
        if not ids:
            raise ValueError("Prompt segment tokenized to an empty sequence")
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    @property
    def backbone_dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def train(self, mode: bool = True) -> FrozenContextClassifier:
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_sensor(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(inputs))

    def compose_input_embeddings(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 2 or projected.shape[1] != MODEL_HIDDEN_SIZE:
            raise ValueError(f"Projected context must have shape [batch, 960], got {projected.shape}")
        batch_size = projected.shape[0]
        prefix = self.prefix_embeddings.expand(batch_size, -1, -1)
        suffix = self.suffix_embeddings.expand(batch_size, -1, -1)
        sensor = projected.unsqueeze(1).to(dtype=self.backbone_dtype)
        return torch.cat((prefix, sensor, suffix), dim=1)

    def forward_from_projected(self, projected: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.compose_input_embeddings(projected)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=self.attention_mask_template.expand(inputs_embeds.shape[0], -1),
            use_cache=False,
            return_dict=True,
        )
        final_hidden = outputs.last_hidden_state[:, -1, :].float()
        return self.head(final_hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_embeddings(inputs)
        return logits

    def forward_with_embeddings(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representations = self.encoder(inputs)
        projected = self.projector(representations)
        logits = self.forward_from_projected(projected)
        return logits, representations, projected

    def trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "encoder": _cpu_state_dict(self.encoder),
            "projector": _cpu_state_dict(self.projector),
            "head": _cpu_state_dict(self.head),
        }

    def load_trainable_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.projector.load_state_dict(state["projector"])
        self.head.load_state_dict(state["head"])


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


def count_trainable_parameters(module: nn.Module, exclude_prefix: str | None = None) -> int:
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and (exclude_prefix is None or not name.startswith(exclude_prefix))
    )


def assert_trainable_parameter_limit(module: nn.Module, exclude_prefix: str | None = None) -> None:
    count = count_trainable_parameters(module, exclude_prefix=exclude_prefix)
    if count > MAX_TRAINABLE_PARAMETERS:
        raise ValueError(f"Trainable component limit exceeded: {count:,} > 10,000,000")
