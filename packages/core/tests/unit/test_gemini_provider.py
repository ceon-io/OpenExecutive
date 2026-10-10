"""Google Gemini as a direct LLM provider.

``GeminiProvider`` is the generic OpenAI-compatible engine pointed at the
Gemini API's OpenAI endpoint. These tests pin what is Gemini-specific: the
bearer auth and URL, the feature spec every Gemini id gets (thinking kept,
Anthropic-only fields stripped), the rewrite of the translator's OpenRouter
``reasoning`` object into Gemini's ``reasoning_effort`` / thinking budget,
``tool_choice: any`` → ``required``, and that no OpenRouter-only field
(``usage``, ``openrouter:web_search``) ever reaches Google.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openexecutive.providers.feature_gate import PDF_OMITTED_NOTE
from openexecutive.providers.gemini_provider import (
    GEMINI_FEATURE_SPEC,
    GEMINI_OPENAI_BASE_URL,
    GeminiProvider,
    gemini_reasoning_fields,
)


def _ok_response() -> MagicMock:
    fake = MagicMock()
    fake.json.return_value = {
        "id": "x",
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    }
    fake.raise_for_status = MagicMock()
    return fake


def _run_create(provider: GeminiProvider, **kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def _fake_post(url: str, **post_kwargs: Any) -> Any:
        captured["url"] = url
        captured["headers"] = post_kwargs.get("headers", {})
        captured["json"] = post_kwargs.get("json", {})
        return _ok_response()

    provider._client.post = AsyncMock(side_effect=_fake_post)  # type: ignore[method-assign]
    asyncio.run(
        provider.messages_create(
            model=kwargs.pop("model", "gemini-3.8-flash"),
            max_tokens=kwargs.pop("max_tokens", 8),
            messages=kwargs.pop("messages", [{"role": "user", "content": "hi"}]),
            **kwargs,
        )
    )
    return captured


def test_defaults_to_the_gemini_openai_endpoint_with_bearer_auth() -> None:
    provider = GeminiProvider(api_key="AIza-test")
    assert str(provider._client.base_url).rstrip("/") == GEMINI_OPENAI_BASE_URL
    captured = _run_create(provider)
    assert captured["url"] == "/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer AIza-test"
    assert captured["json"]["model"] == "gemini-3.8-flash"


def test_every_gemini_id_gets_the_gemini_spec() -> None:
    """A newly released id needs no table entry: the resolver covers all."""
    provider = GeminiProvider(api_key="k")
    assert provider._resolve("gemini-3.8-flash") == ("gemini-3.8-flash", GEMINI_FEATURE_SPEC)
    assert provider._resolve("gemini-4.0-pro-preview") == (
        "gemini-4.0-pro-preview",
        GEMINI_FEATURE_SPEC,
    )


def test_anthropic_only_fields_never_reach_google() -> None:
    """cache_control, the Anthropic web-search server tool and the OpenRouter
    accounting field are all absent from the wire body."""
    captured = _run_create(
        GeminiProvider(api_key="k"),
        system=[{"type": "text", "text": "persona", "cache_control": {"type": "ephemeral"}}],
        tools=[
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {
                "name": "lookup",
                "description": "d",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            },
        ],
    )
    body = captured["json"]
    assert "usage" not in body
    assert "cache_control" not in str(body)
    assert [t["type"] for t in body["tools"]] == ["function"]
    assert body["messages"][0] == {"role": "system", "content": "persona"}


def test_adaptive_thinking_becomes_reasoning_effort() -> None:
    captured = _run_create(
        GeminiProvider(api_key="k"),
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )
    body = captured["json"]
    assert body["reasoning_effort"] == "medium"
    assert "reasoning" not in body and "thinking" not in body


def test_no_thinking_request_sends_no_reasoning_field() -> None:
    body = _run_create(GeminiProvider(api_key="k"))["json"]
    assert "reasoning_effort" not in body and "reasoning" not in body


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ({"effort": "low"}, {"reasoning_effort": "low"}),
        ({"effort": "high"}, {"reasoning_effort": "high"}),
        # OpenRouter's wider ladder folds onto Gemini's three levels.
        ({"effort": "xhigh"}, {"reasoning_effort": "high"}),
        ({"effort": "max"}, {"reasoning_effort": "high"}),
        ({"effort": "minimal"}, {"reasoning_effort": "low"}),
        ({"effort": "none"}, {"reasoning_effort": "low"}),
        # A legacy token budget goes through Google's thinking_config extension.
        ({"max_tokens": 2048}, {"google": {"thinking_config": {"thinking_budget": 2048}}}),
        # Explicitly off, empty, or malformed: the model's own default.
        ({"enabled": False}, {}),
        ({}, {}),
        ({"effort": "bogus"}, {}),
        ({"max_tokens": True}, {}),
        (None, {}),
        ("high", {}),
    ],
)
def test_gemini_reasoning_fields(reasoning: Any, expected: dict[str, Any]) -> None:
    assert gemini_reasoning_fields(reasoning) == expected


def test_forced_any_tool_choice_becomes_openai_required() -> None:
    captured = _run_create(
        GeminiProvider(api_key="k"),
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "any"},
    )
    assert captured["json"]["tool_choice"] == "required"


def test_named_tool_choice_keeps_the_openai_function_form() -> None:
    captured = _run_create(
        GeminiProvider(api_key="k"),
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "emit"},
    )
    assert captured["json"]["tool_choice"] == {"type": "function", "function": {"name": "emit"}}


def test_pdf_document_becomes_a_note_not_a_file_part() -> None:
    captured = _run_create(
        GeminiProvider(api_key="k"),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0x",
                        },
                    },
                    {"type": "text", "text": "summarise"},
                ],
            }
        ],
    )
    user = captured["json"]["messages"][-1]
    parts = user["content"] if isinstance(user["content"], list) else [user["content"]]
    assert not any(isinstance(p, dict) and p.get("type") == "file" for p in parts)
    assert PDF_OMITTED_NOTE in str(user)


def test_stream_body_carries_the_same_rewrites() -> None:
    provider = GeminiProvider(api_key="k")
    stream = provider.messages_stream(
        model="gemini-3.8-flash",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "any"},
    )
    body = stream._body
    assert body["stream"] is True
    assert body["reasoning_effort"] == "high"
    assert body["tool_choice"] == "required"
    assert "reasoning" not in body and "usage" not in body
