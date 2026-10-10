"""Google Gemini backend — ``OpenAICompatibleProvider`` on the Gemini API.

The Gemini API exposes an OpenAI-compatible ``/chat/completions`` endpoint
(``https://generativelanguage.googleapis.com/v1beta/openai``), so the whole
request/response/stream machinery is the generic provider's. What is
Gemini-specific lives in ``_extend_body``:

* Deep reasoning. The translator emits OpenRouter's ``reasoning`` object for
  a request that kept its Anthropic ``thinking`` fields. Gemini does not know
  that field; its OpenAI layer takes a top-level ``reasoning_effort``
  (``low`` / ``medium`` / ``high``) and, for a token budget, the Google
  extension ``google.thinking_config.thinking_budget``. The object is
  rewritten into whichever of those the request asked for.
* ``tool_choice``. The translator passes Anthropic's ``any`` through for
  OpenRouter, which accepts it; the OpenAI form Gemini implements is
  ``required``.

Everything else Anthropic-only is stripped before translation by the feature
spec the registry attaches to Gemini ids (no ``cache_control``, no Anthropic
web-search server tool, no PDF ``document`` block — a PDF becomes a note and
``knowledge.pdf_reader`` OCRs it locally). The OpenRouter-only ``usage``
accounting field is never sent (the base class default).
"""
from __future__ import annotations

from typing import Any

from openexecutive.providers.feature_gate import FeatureSpec
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# What a Gemini id may receive. Thinking stays so the Council's deep-reasoning
# toggle reaches the model as ``reasoning_effort``; cache_control is an
# Anthropic field, the Anthropic web-search server tool has no Gemini-API
# equivalent on this path, and the OpenAI ``file`` part a PDF would become is
# not accepted by Gemini's OpenAI layer.
GEMINI_FEATURE_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=True,
    supports_web_search=False,
    supports_tool_use=True,
    supports_pdf_input=False,
)

# Gemini's accepted ``reasoning_effort`` values. OpenRouter's wider set
# (``none`` / ``minimal`` / ``xhigh`` / ``max``) is folded onto the nearest
# one rather than sent, since an unknown value is an HTTP 400.
_GEMINI_EFFORTS = frozenset({"low", "medium", "high"})
_EFFORT_FOLD = {"none": "low", "minimal": "low", "xhigh": "high", "max": "high"}


def gemini_reasoning_fields(reasoning: Any) -> dict[str, Any]:
    """The Gemini request fields for an OpenRouter-shaped ``reasoning`` object.

    ``{"effort": E}`` → ``{"reasoning_effort": E'}`` with ``E`` folded onto
    low / medium / high; ``{"max_tokens": N}`` → the Google extension
    ``{"google": {"thinking_config": {"thinking_budget": N}}}``;
    ``{"enabled": False}``, an empty object or anything malformed → nothing,
    so the model runs on its own default.
    """
    if not isinstance(reasoning, dict) or reasoning.get("enabled") is False:
        return {}
    effort = reasoning.get("effort")
    if isinstance(effort, str):
        folded = _EFFORT_FOLD.get(effort, effort)
        if folded in _GEMINI_EFFORTS:
            return {"reasoning_effort": folded}
        return {}
    budget = reasoning.get("max_tokens")
    if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
        return {"google": {"thinking_config": {"thinking_budget": budget}}}
    return {}


class GeminiProvider(OpenAICompatibleProvider):
    """LLMProvider implementation backed by the Gemini API directly."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = GEMINI_OPENAI_BASE_URL,
        timeout_s: float = 180.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            # Every id that reaches this provider is a Gemini id (the
            # registry routes nothing else here), so one spec covers them all
            # and a newly released model needs no table entry.
            model_resolver=lambda model: (model, GEMINI_FEATURE_SPEC),
        )

    def _extend_body(self, slug: str, body: dict[str, Any]) -> None:
        reasoning = body.pop("reasoning", None)
        body.update(gemini_reasoning_fields(reasoning))
        if body.get("tool_choice") == "any":
            body["tool_choice"] = "required"
