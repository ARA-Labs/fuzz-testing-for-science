"""xAI Grok CLI adapter — VERIFIED against grok 0.2.102 on this machine
(2026-07-17).

CLI contract (single-JSON style, very close to claude-code):
  - headless: `grok -p "<prompt>" --output-format json` -> one JSON object
    on stdout: {text, sessionId, stopReason, usage{input_tokens,
    output_tokens, reasoning_tokens, cache_read_input_tokens}, num_turns,
    modelUsage}. No cost (subscription via cli-chat-proxy.grok.com).
  - resume: `-r <sessionId>` (verified: same sessionId across turns)
  - model: `-m grok-4.5`; effort: `--reasoning-effort high|medium|low`
    (grok-4.5 defaults to high; pass explicitly anyway)
  - permissions: MUST be `--permission-mode auto` — headless with
    acceptEdits auto-DENIES the `write` tool ("User cancelled the
    execution"), verified live; `auto` writes fine.
  - system prompt: `--rules <text>` appends to the system prompt
    (claude's --append-system-prompt equivalent); injected on first turn.
  - reads workspace AGENTS.md natively (verified with codeword test).
  - transcripts: ~/.grok/sessions/<url-encoded-cwd>/<sessionId>/
    (chat_history.jsonl, events.jsonl, system_prompt.txt, ...);
    collect_archive.sh copies the whole session dir.
"""

import json
import subprocess

from .base import Harness


class Grok(Harness):
    name = "grok"
    instructions_file = "AGENTS.md"

    def turn(self, prompt, system, session_id, cwd, timeout):
        cmd = ["grok", "-p", prompt, "--output-format", "json",
               "--permission-mode", "auto"]
        if self.model:
            cmd += ["-m", self.model]
        if self.effort:
            cmd += ["--reasoning-effort", self.effort]
        if session_id:
            cmd += ["-r", session_id]
        elif system:
            cmd += ["--rules", str(system)]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=cwd,
                             stdin=subprocess.DEVNULL)
        if not res.stdout.strip():
            raise RuntimeError(
                f"empty grok output; stderr: {res.stderr[-500:]}")
        data = json.loads(res.stdout)
        return {
            "text": data.get("text") or "",
            "session_id": data.get("sessionId") or session_id,
            "cost_usd": None,  # grok reports tokens only
            "usage": data.get("usage"),
            "num_turns": data.get("num_turns"),
            "is_error": data.get("stopReason") not in (None, "EndTurn"),
        }
