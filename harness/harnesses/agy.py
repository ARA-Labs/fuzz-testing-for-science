"""agy (Google Antigravity CLI) adapter — VERIFIED against agy 1.1.2 on this
machine (2026-07-15).

CLI contract:
  - non-interactive: `agy --print "<prompt>"` -> final reply as plain text on
    stdout (no JSON mode)
  - resume:  `--conversation <uuid>` (verified: recalls prior turns)
  - workspace: agy uses its own scratch dir unless the cwd is mounted with
    `--add-dir <cwd>` (verified: without it, files land in
    ~/.gemini/antigravity-cli/scratch/!)
  - permissions: `--mode accept-edits` auto-approves file edits (mirrors the
    claude-code bridge's acceptEdits)
  - `--print-timeout` defaults to 5m — must be raised to the bridge turn
    timeout or long physics turns get cut off
  - model: full label from `agy models`, e.g. "Gemini 3.1 Pro (High)",
    "Claude Opus 4.6 (Thinking)". Reasoning effort is part of the label, so
    BRIDGE_EFFORT is ignored here.
  - conversation id: not printed; parsed from the per-run --log-file
    ("Created conversation <uuid>")
  - no usage/cost reporting (subscription quota) -> cost None; "usage" is a
    DETERMINISTIC ESTIMATE, never a measurement: the brain transcript
    (~/.gemini/antigravity-cli/brain/<uuid>/.system_generated/logs/
    transcript_full.jsonl, uuid == conversation id) carries no usage records
    anywhere (arc-wm verified 2026-07-08), so we port arc-wm's
    count_tokens.py algorithm incrementally per turn: one API call per
    PLANNER_RESPONSE record; est output = that record's text; est input =
    context ADDED since the previous call (the re-sent prefix is a cache
    read, which counting would inflate ~100x vs other harnesses' non-cached
    input). chars/4; keys are prefixed est_ and carry a "method" note so the
    estimate can never be mistaken for measured usage. For a post-hoc
    recount with the real Gemini tokenizer run bridge/count_tokens.py on the
    brain transcript with GEMINI_API_KEY set.
  - transcripts: ~/.gemini/antigravity-cli/conversations/<uuid>.db (sqlite;
    collect_archive.sh copies them) + the brain transcript above
  - reads AGENTS.md in the workspace (verified)
  - no system-prompt flag -> system is prepended to the FIRST prompt
"""

import json
import os
import re
import subprocess
import tempfile

from .base import Harness


class Agy(Harness):
    name = "agy"
    instructions_file = "AGENTS.md"

    def __init__(self, model, effort=""):
        super().__init__(model, effort)
        self._brain_offset = 0
        self._ctx_chars = 0        # cumulative context chars seen so far
        self._last_call_ctx = 0    # _ctx_chars at the previous PLANNER_RESPONSE

    def turn(self, prompt, system, session_id, cwd, timeout):
        if system and not session_id:
            prompt = f"[system instructions]\n{system}\n\n{prompt}"

        fd, logf = tempfile.mkstemp(prefix="agy-turn-", suffix=".log")
        os.close(fd)
        try:
            cmd = ["agy", "--print", prompt, "--mode", "accept-edits",
                   "--add-dir", cwd, "--log-file", logf,
                   "--print-timeout", f"{timeout}s"]
            if self.model:
                cmd += ["--model", self.model]
            if session_id:
                cmd += ["--conversation", session_id]
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout + 60, cwd=cwd)
            text = res.stdout.strip()
            sid = session_id
            if not sid:
                try:
                    with open(logf, errors="replace") as f:
                        m = re.search(r"Created conversation ([0-9a-f-]{36})",
                                      f.read())
                    sid = m.group(1) if m else None
                except OSError:
                    pass
            if not text:
                raise RuntimeError(
                    f"empty agy output (exit {res.returncode}); "
                    f"stderr: {res.stderr[-500:]}")
            return {
                "text": text,
                "session_id": sid,
                "cost_usd": None,   # agy reports neither dollars nor tokens
                "usage": self._est_usage(sid),
                "num_turns": None,
                "is_error": res.returncode != 0,
            }
        finally:
            try:
                os.unlink(logf)
            except OSError:
                pass

    def _est_usage(self, sid):
        """Per-turn token ESTIMATE from the brain transcript (see module
        docstring — agy reports no real usage). Best-effort: None if the
        transcript isn't found."""
        if not sid:
            return None
        path = os.path.expanduser(
            f"~/.gemini/antigravity-cli/brain/{sid}"
            "/.system_generated/logs/transcript_full.jsonl")
        try:
            f = open(path, errors="replace")
        except OSError:
            return None
        in_chars = out_chars = calls = 0
        with f:
            f.seek(self._brain_offset)
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (isinstance(d, dict) and "step_index" in d
                        and "source" in d and "status" in d):
                    continue
                step = 0
                if isinstance(d.get("content"), str):
                    step += len(d["content"])
                if isinstance(d.get("thinking"), str):
                    step += len(d["thinking"])
                if isinstance(d.get("tool_calls"), list):
                    step += len(json.dumps(d["tool_calls"]))
                if d.get("type") == "PLANNER_RESPONSE":
                    in_chars += self._ctx_chars - self._last_call_ctx
                    self._last_call_ctx = self._ctx_chars
                    out_chars += step
                    calls += 1
                self._ctx_chars += step
            self._brain_offset = f.tell()
        if not (calls or in_chars or out_chars):
            return None
        return {
            "est_input_tokens_uncached": in_chars // 4,
            "est_output_tokens": out_chars // 4,
            "api_calls": calls,
            "method": "chars/4 estimate from brain transcript "
                      "(agy reports no usage; arc-wm algorithm)",
        }
