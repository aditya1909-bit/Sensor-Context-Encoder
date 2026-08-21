from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sensor_context_encoder.models import (
    EXPECTED_CONTEXT_PARAMETERS,
    EXPECTED_DIRECT_PARAMETERS,
    DirectClassifier,
    FrozenContextClassifier,
    SensorEncoder,
    count_trainable_parameters,
)


class DummyTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "\n\nActivity:":
            return [5, 6]
        return [3, 7 + sum(text.encode("utf-8")) % 9]


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=960)
        self.embeddings = nn.Embedding(16, 960)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


def make_context_model() -> FrozenContextClassifier:
    return FrozenContextClassifier(
        device=torch.device("cpu"),
        backbone=DummyBackbone(),
        tokenizer=DummyTokenizer(),
        backbone_dtype=torch.float32,
    )


def test_encoder_and_direct_shapes_and_parameter_counts() -> None:
    inputs = torch.randn(2, 128, 9)
    encoder = SensorEncoder()
    assert encoder(inputs).shape == (2, 256)
    direct = DirectClassifier()
    assert direct(inputs).shape == (2, 6)
    assert count_trainable_parameters(direct) == EXPECTED_DIRECT_PARAMETERS


def test_context_inserts_exactly_one_sensor_embedding() -> None:
    model = make_context_model()
    projected = torch.randn(2, 960)
    composed = model.compose_input_embeddings(projected)
    prefix_length = model.prefix_embeddings.shape[1]
    suffix_length = model.suffix_embeddings.shape[1]
    assert composed.shape == (2, prefix_length + 1 + suffix_length, 960)
    torch.testing.assert_close(composed[:, prefix_length, :], projected)
    expected_logits = model.head(composed.cumsum(dim=1)[:, -1, :])
    torch.testing.assert_close(model.forward_from_projected(projected), expected_logits)
    assert count_trainable_parameters(model) == EXPECTED_CONTEXT_PARAMETERS


def test_gradients_flow_only_to_trainable_context_components() -> None:
    model = make_context_model()
    model.train()
    logits = model(torch.randn(2, 128, 9))
    assert logits.shape == (2, 6)
    logits.sum().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    for module in (model.encoder, model.projector, model.head):
        assert any(parameter.grad is not None for parameter in module.parameters())
    assert model.backbone.training is False


def test_context_checkpoint_excludes_backbone() -> None:
    state = make_context_model().trainable_state_dict()
    assert set(state) == {"encoder", "projector", "head"}
    assert all("backbone" not in name for group in state.values() for name in group)
