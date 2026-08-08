from __future__ import annotations

from types import SimpleNamespace

import torch

from mfq._vendor.tpq.engine import _sample_top_p
from mfq._vendor.tpq.openai_api import (
    ChatCompletionRequest,
    _options_from_openai,
    _reasoning_options,
)


def _request(**updates: object) -> ChatCompletionRequest:
    payload: dict[str, object] = {
        "model": "mfq",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(updates)
    return ChatCompletionRequest.model_validate(payload)


def test_webui_request_extensions_are_validated_and_mapped() -> None:
    request = _request(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        reasoning_format="auto",
        chat_template_kwargs={
            "enable_thinking": True,
            "reasoning_effort": "medium",
        },
    )
    service = SimpleNamespace(default_reasoning="chat")
    options = _options_from_openai(service, request)
    assert options.top_k == 20
    assert options.thinking_mode == "thinking"
    assert options.reasoning_effort == "medium"


def test_top_level_reasoning_fields_override_template_kwargs() -> None:
    request = _request(
        reasoning_effort="low",
        enable_thinking=False,
        chat_template_kwargs={
            "enable_thinking": True,
            "reasoning_effort": "max",
        },
    )
    assert _reasoning_options(
        SimpleNamespace(default_reasoning="high"), request
    ) == ("chat", None)


def test_kimi_low_and_medium_reasoning_efforts_reach_adapter_options() -> None:
    for effort in ("low", "medium"):
        request = _request(reasoning_effort=effort)
        assert _reasoning_options(
            SimpleNamespace(default_reasoning="chat"), request
        ) == ("thinking", effort)


def test_top_k_one_samples_only_the_highest_logit() -> None:
    torch.manual_seed(1)
    logits = torch.tensor([0.0, 4.0, 3.0], dtype=torch.float32)
    for _ in range(8):
        assert _sample_top_p(
            logits,
            temp=1.0,
            top_p=1.0,
            top_k=1,
        ) == 1
