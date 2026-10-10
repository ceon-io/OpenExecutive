"""Registry routing for Gemini ids.

``GEMINI_ENABLED`` adds a fourth backend next to Anthropic, OpenRouter and
local: a Gemini id routes to the Gemini API directly when it is on, under its
``google/`` slug through OpenRouter when only OpenRouter is on, and to an
actionable 400 when neither is.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from openexecutive.providers import (  # noqa: E402
    ANTHROPIC_DIRECT_MODELS,
    GEMINI_DIRECT_MODELS,
    allowed_models,
    get_provider,
    model_options_for,
)
from openexecutive.providers import registry as registry_mod  # noqa: E402
from openexecutive.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from openexecutive.providers.gemini_provider import (  # noqa: E402
    GEMINI_FEATURE_SPEC,
    GeminiProvider,
)
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from openexecutive.providers.openrouter_provider import OpenRouterProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gemini: bool = True,
    gemini_key: str | None = "AIza-test",
    gemini_models: list[str] | None = None,
    anthropic_key: str | None = "sk-test",
    openrouter: bool = False,
    local_models: list[str] | None = None,
) -> Any:
    settings = SimpleNamespace(
        anthropic_api_key=anthropic_key,
        openrouter_enabled=openrouter,
        openrouter_api_key="sk-or-v1-xxx" if openrouter else None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_app_title="Open Executive",
        openrouter_referer=None,
        openrouter_timeout_s=180.0,
        gemini_enabled=gemini,
        gemini_api_key=gemini_key,
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        gemini_models=list(GEMINI_DIRECT_MODELS) if gemini_models is None else gemini_models,
        gemini_timeout_s=180.0,
        local_models_enabled=bool(local_models),
        local_base_url="http://localhost:11434/v1" if local_models else None,
        local_models=local_models or [],
        local_api_key=None,
        local_timeout_s=300.0,
        local_include_usage_accounting=False,
        local_reasoning_effort=None,
        local_pdf_input=False,
    )
    monkeypatch.setattr("openexecutive.providers.registry.get_settings", lambda: settings)
    registry_mod._reset_for_tests()
    return settings


def test_gemini_id_routes_to_gemini_provider_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch)
    provider = get_provider("gemini-3.8-flash")
    assert isinstance(provider, GeminiProvider)
    # One singleton for every Gemini id, including one not in GEMINI_MODELS.
    assert get_provider("gemini-3.5-flash-lite") is provider
    assert get_provider("gemini-4.0-pro-preview") is provider


def test_gemini_provider_is_distinct_from_the_other_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, openrouter=True, local_models=["llama3.3"])
    gemini = get_provider("gemini-3.8-flash")
    assert isinstance(gemini, GeminiProvider)
    claude = get_provider("claude-sonnet-5-5")
    assert isinstance(claude, OpenRouterProvider) and claude is not gemini
    local = get_provider("llama3.3")
    assert isinstance(local, OpenAICompatibleProvider) and local is not gemini


def test_gemini_direct_wins_over_openrouter_for_bare_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both on: a bare Gemini id is billed to the Google key; the ``google/``
    OpenRouter slug for the same model still goes through OpenRouter."""
    _stub(monkeypatch, openrouter=True)
    assert isinstance(get_provider("gemini-3.8-flash"), GeminiProvider)
    assert isinstance(get_provider("google/gemini-3.8-flash"), OpenRouterProvider)


def test_listed_gemini_model_routes_direct_even_if_oddly_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, gemini_models=["my-tuned-gemini"])
    assert isinstance(get_provider("my-tuned-gemini"), GeminiProvider)


def test_local_slug_wins_over_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, local_models=["gemini-3.8-flash"])
    assert isinstance(get_provider("gemini-3.8-flash"), OpenAICompatibleProvider)
    assert not isinstance(get_provider("gemini-3.8-flash"), GeminiProvider)


def test_gemini_off_with_openrouter_on_routes_under_google_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, gemini=False, gemini_key=None, openrouter=True)
    provider = get_provider("gemini-3.8-flash")
    assert isinstance(provider, OpenRouterProvider)
    slug, spec = provider._resolve("gemini-3.8-flash")
    assert slug == "google/gemini-3.8-flash"
    assert spec.supports_cache_control is False


def test_gemini_off_with_openrouter_off_raises_actionable_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, gemini=False, gemini_key=None)
    with pytest.raises(HTTPException) as exc_info:
        get_provider("gemini-3.8-flash")
    assert exc_info.value.status_code == 400
    assert "GEMINI_ENABLED" in exc_info.value.detail


def test_claude_still_routes_direct_with_gemini_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch)
    assert isinstance(get_provider("claude-sonnet-5-5"), AnthropicProvider)


def test_gemini_enabled_without_key_raises_400(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, gemini_key=None)
    with pytest.raises(HTTPException) as exc_info:
        get_provider("gemini-3.8-flash")
    assert exc_info.value.status_code == 400
    assert "GEMINI_API_KEY" in exc_info.value.detail


def test_gemini_provider_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _stub(monkeypatch)
    settings.gemini_base_url = "https://gateway.example/openai"
    settings.gemini_timeout_s = 42.0
    provider = get_provider("gemini-3.8-flash")
    assert isinstance(provider, GeminiProvider)
    assert str(provider._client.base_url).rstrip("/") == "https://gateway.example/openai"
    assert provider._client.timeout.read == 42.0
    assert provider._resolve("gemini-3.8-flash") == ("gemini-3.8-flash", GEMINI_FEATURE_SPEC)


# --------------------------------------------------------------------------
# allowed_models / model_options_for
# --------------------------------------------------------------------------


def test_allowed_models_lists_gemini_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, gemini=False, gemini_key=None)
    assert allowed_models() == ANTHROPIC_DIRECT_MODELS
    _stub(monkeypatch)
    assert allowed_models() == ANTHROPIC_DIRECT_MODELS + GEMINI_DIRECT_MODELS


def test_gemini_only_deployment_offers_only_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, anthropic_key=None)
    assert allowed_models() == GEMINI_DIRECT_MODELS


def test_allowed_models_uses_the_configured_gemini_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, anthropic_key=None, gemini_models=["gemini-3.8-pro", "gemini-3.8-flash"])
    assert allowed_models() == ["gemini-3.8-pro", "gemini-3.8-flash"]


def test_model_options_group_gemini_under_google_with_gemini_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch)
    options = {o["id"]: o for o in model_options_for(None)}
    flash = options["gemini-3.8-flash"]
    assert flash["provider"] == "google"
    assert flash["provider_label"] == "Google"
    assert flash["route"] == "gemini"
    assert flash["label"] == "Gemini 3.8 Flash"
    assert options["gemini-3.5-flash-lite"]["label"] == "Gemini 3.5 Flash Lite"
    assert options["claude-sonnet-5-5"]["route"] == "direct"


def test_model_options_openrouter_gemini_slug_keeps_openrouter_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, openrouter=True)
    options = {o["id"]: o for o in model_options_for(None)}
    assert options["gemini-3.8-flash"]["route"] == "gemini"
    via = options["google/gemini-3.8-flash"]
    assert via["route"] == "openrouter"
    assert via["provider"] == "google"
    assert via["label"] == "Gemini 3.8 Flash"


def test_model_options_ids_match_allowlist_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, openrouter=True, local_models=["llama3.3"])
    assert [o["id"] for o in model_options_for(None)] == allowed_models()


def test_pdf_input_unsupported_on_gemini_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, openrouter=True)
    assert registry_mod.pdf_input_supported("gemini-3.8-flash") is False
    assert registry_mod.pdf_input_supported("google/gemini-3.8-flash") is True
    assert registry_mod.pdf_input_supported("claude-sonnet-5-5") is True
