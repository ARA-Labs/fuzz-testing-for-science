"""Claude Code adapter — the original bridge behavior, plus optional
provider override for non-Anthropic backends served over the
Anthropic-compatible API (e.g. GLM via api.z.ai).

CLI contract (`claude -p --output-format json`): single JSON object on stdout
with keys session_id, result, total_cost_usd, usage, num_turns, is_error.
Resume via --resume <id>; system prompt injected once at session start via
--append-system-prompt. Session transcripts live under
~/.claude/projects/<escaped-cwd>/<session_id>.jsonl (collect_archive.sh).

Provider override: BRIDGE_ENV_FILE points at a chmod-600 KEY=VALUE file
(ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN) applied to the AGENT subprocess
env ONLY — the judge path (bridge_client._oneshot_call) must stay on the real
Anthropic API and additionally strips these vars. With an override active,
total_cost_usd from the CLI is meaningless (CC's own price table) — treat
cost as null downstream and use the provider's billing.
"""

import json
import os
import subprocess

from .base import Harness


def _agent_env():
    path = os.environ.get("BRIDGE_ENV_FILE")
    if not path:
        return None
    env = dict(os.environ)
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


class ClaudeCode(Harness):
    name = "claude-code"
    instructions_file = "CLAUDE.md"

    ALLOWED_TOOLS = ["Skill", "Read", "Write", "Edit", "Grep", "Glob", "LS",
                     "TodoWrite",
                     # agents MUST be able to execute their own analysis
                     # code: without this every archived script is
                     # untested code (dp-gravity fit_round1.py lesson,
                     # 2026-07-18 — 6 python3 calls denied in-session)
                     "Bash(python3:*)", "Bash(python:*)"]

    def turn(self, prompt, system, session_id, cwd, timeout):
        cmd = ["claude", "-p", "--output-format", "json",
               "--model", self.model or "sonnet",
               "--permission-mode", "acceptEdits",
               "--allowedTools"] + self.ALLOWED_TOOLS
        if self.effort:
            cmd += ["--effort", self.effort]
        if session_id:
            cmd += ["--resume", session_id]
        elif system:
            cmd += ["--append-system-prompt", str(system)]
        env = _agent_env()
        res = subprocess.run(cmd, input=prompt, capture_output=True,
                             text=True, timeout=timeout, cwd=cwd, env=env)
        if not res.stdout.strip():
            raise RuntimeError(
                f"empty claude output; stderr: {res.stderr[-500:]}")
        data = json.loads(res.stdout)
        return {
            "text": data.get("result") or "",
            "session_id": data.get("session_id"),
            # with a provider override the CLI's cost table is wrong — null it
            "cost_usd": None if env else data.get("total_cost_usd"),
            "usage": data.get("usage"),
            "num_turns": data.get("num_turns"),
            "is_error": bool(data.get("is_error")),
        }
