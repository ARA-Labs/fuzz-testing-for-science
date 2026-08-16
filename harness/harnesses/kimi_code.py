"""Kimi Code (Moonshot kimi CLI) adapter — VERIFIED against kimi-code 0.24.1
on this machine (2026-07-15).

CLI contract:
  - non-interactive: `kimi -p "<prompt>" --output-format stream-json` ->
    JSONL on stdout: {"role":"assistant","content":...} messages,
    {"role":"tool",...} results, and a final meta event
    {"type":"session.resume_hint","session_id":"session_<uuid>"}
  - print mode auto-approves tool calls (verified: wrote a file with no
    permission flag; -y/--auto are rejected in combination with -p)
  - resume: `-S <session_id>` (verified: recalls prior turns; replays only
    the NEW turn's events)
  - model: alias from ~/.kimi-code/config.toml (default kimi-code/kimi-for-
    coding). No effort flag -> BRIDGE_EFFORT ignored.
  - token usage: not in the stream; read incrementally from the session's
    wire.jsonl ("usage.record" events) under
    ~/.kimi-code/sessions/wd_*/<session_id>/agents/*/wire.jsonl
  - cost: not reported (subscription) -> cost_usd None
  - transcripts: the session dir above (collect_archive.sh copies it)
  - reads AGENTS.md in the workspace (verified)
  - no system-prompt flag -> system is prepended to the FIRST prompt
"""

import glob
import json
import os
import subprocess

from .base import Harness


class KimiCode(Harness):
    name = "kimi-code"
    instructions_file = "AGENTS.md"

    def __init__(self, model, effort=""):
        super().__init__(model, effort)
        self._wire_path = None
        self._wire_offset = 0

    def turn(self, prompt, system, session_id, cwd, timeout):
        if system and not session_id:
            prompt = f"[system instructions]\n{system}\n\n{prompt}"

        cmd = ["kimi", "-p", prompt, "--output-format", "stream-json"]
        if self.model:
            cmd += ["-m", self.model]
        if session_id:
            cmd += ["-S", session_id]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=cwd)

        events = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not events:
            raise RuntimeError(
                f"no stream-json events from kimi (exit {res.returncode}); "
                f"stderr: {res.stderr[-500:]}")

        sid = next((e.get("session_id") for e in reversed(events)
                    if e.get("type") == "session.resume_hint"
                    and e.get("session_id")), None) or session_id
        # join all assistant messages of the turn: final tags may be emitted
        # before a last short wrap-up message
        assistant = [str(e.get("content")) for e in events
                     if e.get("role") == "assistant" and e.get("content")]
        return {
            "text": "\n\n".join(assistant),
            "session_id": sid,
            "cost_usd": None,  # subscription; tokens only
            "usage": self._turn_usage(sid),
            "num_turns": None,
            "is_error": res.returncode != 0,
        }

    def _turn_usage(self, sid):
        """Sum token counters from wire.jsonl lines appended since the last
        turn. Best-effort: returns None if the session dir isn't found."""
        try:
            if self._wire_path is None:
                if not sid:
                    return None
                hits = glob.glob(os.path.expanduser(
                    f"~/.kimi-code/sessions/*/{sid}/agents/*/wire.jsonl"))
                if not hits:
                    return None
                self._wire_path = hits[0]
            usage = {}
            with open(self._wire_path) as f:
                f.seek(self._wire_offset)
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    u = d.get("usage")
                    if isinstance(u, dict):
                        for k, v in u.items():
                            if isinstance(v, (int, float)):
                                usage[k] = usage.get(k, 0) + v
                self._wire_offset = f.tell()
            return usage or None
        except OSError:
            return None
