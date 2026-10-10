"""LLM provider abstraction.

All Anthropic-shaped LLM calls go through ``get_provider(model)`` rather
than instantiating ``anthropic.AsyncAnthropic`` directly. The backends are
Anthropic itself, Google's Gemini API (``GEMINI_ENABLED``), OpenRouter and a
local OpenAI-compatible server; the layer exists so any of them can serve a
call site's Anthropic-shaped request, chosen per call by model.
"""
from openexecutive.providers.provider import LLMProvider
from openexecutive.providers.registry import (
    ANTHROPIC_DIRECT_MODELS,
    GEMINI_DIRECT_MODELS,
    OPENROUTER_MODELS,
    allowed_models,
    allowed_models_for,
    get_provider,
    model_options_for,
    model_supports_deep_reasoning,
)

__all__ = [
    "ANTHROPIC_DIRECT_MODELS",
    "GEMINI_DIRECT_MODELS",
    "LLMProvider",
    "OPENROUTER_MODELS",
    "allowed_models",
    "allowed_models_for",
    "get_provider",
    "model_options_for",
    "model_supports_deep_reasoning",
]
