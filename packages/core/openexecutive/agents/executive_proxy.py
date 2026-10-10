"""Thin BaseAgent subclass so the Executive can participate in the Agent Council.

The ExecutiveProxy is NOT registered in SPECIALIST_REGISTRY and is never used for
routing. Its only purpose is to give the agents API route a uniform interface for
building AgentMeta / AgentDetail and to surface effective_*() override resolution
for the Executive's prompt and model.

The test-box endpoint calls proxy.analyze() directly — this makes a single-turn
completion, which is the correct "does this prompt produce sensible output?" check.
"""
from __future__ import annotations

from openexecutive.agents.base import BaseAgent
from openexecutive.config import get_settings


class ExecutiveProxy(BaseAgent):
    name = "executive"
    domain = "orchestration"
    visibility = "core"
    use_deep_reasoning = False

    @property
    def model(self) -> str:  # type: ignore[override]
        # What the chat Executive actually runs on (executive.py reads the
        # same setting), so the Council shows the real default and a
        # Gemini-only install's test box does not call a Claude id.
        return get_settings().default_model

    def get_system_prompt(self) -> str:
        # The built-in persona for this install's workspace mode (team / solo).
        from openexecutive.memory.workspace_settings import get_workspace
        from openexecutive.prompts.executive_persona import default_persona

        return default_persona(get_workspace().mode)
