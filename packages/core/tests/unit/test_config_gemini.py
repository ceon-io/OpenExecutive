"""Config-layer tests for the Gemini provider settings.

Built with ``_env_file=None`` so the repo-root ``.env`` can't leak values in.
"""
from __future__ import annotations

import pytest

from openexecutive.config import Settings

_PROVIDER_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_ENABLED",
    "OPENROUTER_API_KEY",
    "LOCAL_MODELS_ENABLED",
    "LOCAL_BASE_URL",
    "LOCAL_MODELS",
    "GEMINI_ENABLED",
    "GEMINI_API_KEY",
    "GEMINI_MODELS",
    "GEMINI_BASE_URL",
    "DEFAULT_MODEL",
    "DEEP_REASONING_MODEL",
    "ROUTING_MODEL",
    "RESEARCH_MODEL",
)


def _build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("EXEC_EMAIL_ADDRESS", "ceo.test@example.com")
    for key in _PROVIDER_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_gemini_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build(monkeypatch, ANTHROPIC_API_KEY="sk-test")
    assert s.gemini_enabled is False
    assert s.gemini_api_key is None
    assert s.gemini_models == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert s.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    # Claude stays the default when Gemini is off.
    assert s.default_model == "claude-sonnet-5-5"


def test_gemini_enabled_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        _build(monkeypatch, ANTHROPIC_API_KEY="sk-test", GEMINI_ENABLED="true")


def test_gemini_models_csv_parses_and_empty_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-test",
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
        GEMINI_MODELS="gemini-3.8-flash, gemini-3.8-pro ,",
    )
    assert s.gemini_models == ["gemini-3.8-flash", "gemini-3.8-pro"]
    s = _build(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-test",
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
        GEMINI_MODELS=" , ",
    )
    assert s.gemini_models == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


def test_gemini_alone_satisfies_provider_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The headline: boot with no Anthropic key, only a Gemini key."""
    s = _build(monkeypatch, GEMINI_ENABLED="true", GEMINI_API_KEY="AIza")
    assert s.anthropic_api_key is None
    assert s.gemini_enabled is True


def test_no_provider_error_names_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="GEMINI_ENABLED"):
        _build(monkeypatch)


def test_gemini_only_install_runs_every_model_setting_on_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing able to serve a Claude id, the unset model settings
    become the first GEMINI_MODELS entry so the whole app runs on Gemini."""
    s = _build(monkeypatch, GEMINI_ENABLED="true", GEMINI_API_KEY="AIza")
    assert s.default_model == "gemini-3.8-flash"
    assert s.deep_reasoning_model == "gemini-3.8-flash"
    assert s.routing_model == "gemini-3.8-flash"
    assert s.research_model == "gemini-3.8-flash"


def test_gemini_only_install_honours_the_first_listed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build(
        monkeypatch,
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
        GEMINI_MODELS="gemini-3.8-pro,gemini-3.8-flash",
    )
    assert s.default_model == "gemini-3.8-pro"


def test_gemini_only_install_keeps_explicit_model_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build(
        monkeypatch,
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
        ROUTING_MODEL="gemini-3.5-flash-lite",
    )
    assert s.routing_model == "gemini-3.5-flash-lite"
    assert s.default_model == "gemini-3.8-flash"


def test_anthropic_key_keeps_claude_defaults_with_gemini_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both providers configured: Claude stays primary unless the operator
    points a model setting at Gemini."""
    s = _build(monkeypatch, ANTHROPIC_API_KEY="sk-test", GEMINI_ENABLED="true", GEMINI_API_KEY="AIza")
    assert s.default_model == "claude-sonnet-5-5"
    assert s.routing_model == "claude-haiku-5-5"
    s = _build(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-test",
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
        DEFAULT_MODEL="gemini-3.8-flash",
    )
    assert s.default_model == "gemini-3.8-flash"


def test_openrouter_on_keeps_claude_defaults_with_gemini_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter can serve the Claude defaults, so they are not replaced."""
    s = _build(
        monkeypatch,
        OPENROUTER_ENABLED="true",
        OPENROUTER_API_KEY="sk-or",
        GEMINI_ENABLED="true",
        GEMINI_API_KEY="AIza",
    )
    assert s.default_model == "claude-sonnet-5-5"
