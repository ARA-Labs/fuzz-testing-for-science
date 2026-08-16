"""Harness registry. BRIDGE_HARNESS env picks one; see base.py to add more."""

from .agy import Agy
from .claude_code import ClaudeCode
from .codex import Codex
from .grok import Grok
from .kimi_code import KimiCode

REGISTRY = {cls.name: cls for cls in (ClaudeCode, Codex, KimiCode, Agy, Grok)}


def get_harness(name, model, effort=""):
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown harness '{name}'; known: {sorted(REGISTRY)}") from None
    return cls(model, effort)
