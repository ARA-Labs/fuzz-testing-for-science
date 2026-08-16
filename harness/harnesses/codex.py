"""OpenAI Codex CLI adapter — VERIFIED against codex-cli 0.144.4 on this
machine (2026-07-15).

CLI contract:
  - `codex exec --json <prompt>` streams JSONL events on stdout:
    {"type":"thread.started","thread_id":...}, item.completed
    (agent_message/file_change/...), {"type":"turn.completed","usage":{...}}
  - resume:  `codex exec resume <SESSION_ID> <prompt>` (exec is
    non-interactive, no approval prompts)
  - `--output-last-message <file>` writes the final assistant message
  - reasoning effort via `-c model_reasoning_effort="high"`
  - no --append-system-prompt equivalent: the system prompt is prepended to
    the FIRST user message; standing rules belong in AGENTS.md in the
    workspace (launch_sweep.sh puts them there)
  - cost: codex reports token usage only -> cost_usd is None; compute $ in
    post-processing from usage if needed
  - usage in turn.completed is CUMULATIVE for the whole session (verified:
    strictly increasing across resumed turns, incl. output_tokens), so this
    adapter differences it against the previous turn and returns per-turn
    deltas in "usage" (the raw session total rides along as
    "usage_cumulative")
  - transcripts on disk: ~/.codex/sessions/YYYY/MM/DD/rollout-*-<id>.jsonl
    (collect_archive.sh finds them by session id)

Sandbox ($BRIDGE_CODEX_SANDBOX, default workspace-write): on THIS Azure VM
codex's bundled bwrap sandbox cannot start (kernel.apparmor_restrict_
unprivileged_userns=1 -> "bwrap: loopback: Failed RTM_NEWADDR") and every
write silently fails, so either run with
BRIDGE_CODEX_SANDBOX=danger-full-access or fix the sysctl:
  sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0
launch_sweep.sh preflights this and refuses to start a broken sweep.
"""

import json
import os
import subprocess
import tempfile

from .base import Harness

_ID_KEYS = ("thread_id", "session_id", "conversation_id", "rollout_id")


def _find_key(obj, keys):
    """Depth-first search a parsed event for the first string value under
    any of `keys` (event shapes vary across codex versions)."""
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
        for v in obj.values():
            hit = _find_key(v, keys)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key(v, keys)
            if hit:
                return hit
    return None


class Codex(Harness):
    name = "codex"
    instructions_file = "AGENTS.md"

    def __init__(self, model, effort=""):
        super().__init__(model, effort)
        self.sandbox = os.environ.get("BRIDGE_CODEX_SANDBOX",
                                      "workspace-write")
        self._cum_usage = {}  # session_id -> last cumulative usage seen

    def turn(self, prompt, system, session_id, cwd, timeout):
        if system and not session_id:
            prompt = f"[system instructions]\n{system}\n\n{prompt}"

        fd, last_msg = tempfile.mkstemp(prefix="codex-last-", suffix=".txt")
        os.close(fd)
        try:
            # `exec resume` accepts fewer flags than `exec` (no --sandbox/-C),
            # so the sandbox is set via -c config override, valid for both
            flags = ["--json", "--skip-git-repo-check",
                     "--output-last-message", last_msg,
                     "-c", f'sandbox_mode="{self.sandbox}"']
            if self.model:  # empty -> CLI's configured default model
                flags += ["-m", self.model]
            if self.effort:
                flags += ["-c", f'model_reasoning_effort="{self.effort}"']
            if session_id:
                cmd = ["codex", "exec", "resume", session_id] + flags + [prompt]
            else:
                cmd = ["codex", "exec", "-C", cwd] + flags + [prompt]
            # DEVNULL: with a piped stdin codex appends it as a <stdin> block
            # and blocks/errors waiting for EOF
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout, cwd=cwd,
                                 stdin=subprocess.DEVNULL)

            events = []
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if not events and res.returncode != 0:
                raise RuntimeError(
                    f"codex exited {res.returncode} with no JSON events; "
                    f"stderr: {res.stderr[-500:]}")

            sid = None
            usage = None
            agent_msgs = []
            for ev in events:
                sid = sid or _find_key(ev, _ID_KEYS)
                u = ev.get("usage")
                if not isinstance(u, dict) and isinstance(ev.get("msg"), dict):
                    u = ev["msg"].get("usage")
                if isinstance(u, dict) and any(
                        isinstance(v, (int, float)) for v in u.values()):
                    usage = u  # keep the last one (cumulative session totals)
                item = ev.get("item") or ev.get("msg") or {}
                if isinstance(item, dict) and item.get("type") in (
                        "agent_message", "assistant_message"):
                    t = item.get("text") or item.get("message")
                    if t:
                        agent_msgs.append(str(t))

            text = ""
            try:
                with open(last_msg) as f:
                    text = f.read().strip()
            except OSError:
                pass
            if not text and agent_msgs:
                text = agent_msgs[-1]
            if not text and res.returncode != 0:
                raise RuntimeError(
                    f"codex exited {res.returncode} with no final message; "
                    f"stderr: {res.stderr[-500:]}")

            usage_cum, usage_delta = usage, None
            if isinstance(usage, dict):
                key = sid or session_id or "_"
                prev = self._cum_usage.get(key) or {}
                usage_delta = {
                    k: (max(0, v - prev.get(k, 0))
                        if isinstance(v, (int, float)) else v)
                    for k, v in usage.items()}
                self._cum_usage[key] = usage

            return {
                "text": text,
                "session_id": sid or session_id,
                "cost_usd": None,  # codex reports tokens, not dollars
                "usage": usage_delta,
                "usage_cumulative": usage_cum,
                "num_turns": None,
                "is_error": res.returncode != 0,
            }
        finally:
            try:
                os.unlink(last_msg)
            except OSError:
                pass
