"""Bridge: routes scienceagent.llm_client.complete() through a coding-agent
CLI harness (Claude Code / Codex / Kimi Code / agy — see harnesses/).

Two paths:
  - model == "cc-session": one persistent harness session in the agent-work
    dir (the measured agent). Which CLI runs it is chosen by $BRIDGE_HARNESS.
    Only NEW user messages since the last call are forwarded — the session
    keeps its own history.
  - any other model (e.g. the explanation judge): stateless one-shot
    `claude -p` with no tools. The judge ALWAYS runs through Claude Code
    regardless of the measured harness, so explanation scores stay
    comparable across harnesses.

Per session turn: git snapshot of agent-work (ara/ growth history) + a ledger
line with usage/cost in $BRIDGE_RUN_DIR/bridge_ledger.jsonl (now also
recording "harness"; collect_archive.sh keys transcript collection off it).

Env (new BRIDGE_* names; old CC_* names still honored):
  BRIDGE_HARNESS                     claude-code (default) | codex | kimi-code | agy
  BRIDGE_MODEL      / CC_MODEL       model string passed to the harness CLI
  BRIDGE_EFFORT     / CC_EFFORT      reasoning effort, if the CLI supports it
  BRIDGE_AGENT_WORK / CC_AGENT_WORK  the measured workspace (default ~/agent-work)
  BRIDGE_RUN_DIR                     ledger/result dir
  BRIDGE_TURN_TIMEOUT / CC_TURN_TIMEOUT  seconds per bridge turn (default 1800)
"""

import json
import os
import subprocess
import tempfile
import time

import instrument
from harnesses import get_harness


def _env(*names, default=""):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


HARNESS_NAME = _env("BRIDGE_HARNESS", default="claude-code")
# empty model = each CLI's own default (claude adapter defaults to sonnet)
MODEL = _env("BRIDGE_MODEL", "CC_MODEL")
EFFORT = _env("BRIDGE_EFFORT", "CC_EFFORT")
AGENT_WORK = os.path.expanduser(
    _env("BRIDGE_AGENT_WORK", "CC_AGENT_WORK", default="~/agent-work"))
RUN_DIR = os.path.expanduser(
    _env("BRIDGE_RUN_DIR", default="~/bench/bridge/runs/default"))
TURN_TIMEOUT_S = int(_env("BRIDGE_TURN_TIMEOUT", "CC_TURN_TIMEOUT",
                          default="1800"))

_state = {"harness": None, "session_id": None, "fwd_len": 0, "round": 0}


def _ledger(entry):
    os.makedirs(RUN_DIR, exist_ok=True)
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(RUN_DIR, "bridge_ledger.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _snapshot(round_num):
    subprocess.run(["git", "-C", AGENT_WORK, "add", "-A"], capture_output=True)
    subprocess.run(
        ["git", "-C", AGENT_WORK, "-c", "user.email=bridge@local", "-c",
         "user.name=bridge", "commit", "-q", "--allow-empty", "-m",
         f"round {round_num}"],
        capture_output=True,
    )


def _session_call(messages, system):
    if _state["harness"] is None:
        _state["harness"] = get_harness(HARNESS_NAME, MODEL, EFFORT)
    h = _state["harness"]

    new = messages[_state["fwd_len"]:]
    parts = [m["content"] for m in new if m.get("role") == "user"]
    prompt = "\n\n".join(str(p) for p in parts) or "(continue)"

    # B1 arm: append the oracle-free instrument panel ($BRIDGE_INSTRUMENT)
    # AFTER the experiment output — recency position, so it is actually read.
    panel, readings = instrument.maybe_panel(messages)
    if panel:
        prompt = prompt + "\n\n" + panel

    _state["round"] += 1
    t0 = time.time()
    try:
        r = h.turn(prompt, system, _state["session_id"],
                   cwd=AGENT_WORK, timeout=TURN_TIMEOUT_S)
    except Exception as e:  # graceful: official loop re-prompts on tag-less replies
        _ledger({"round": _state["round"], "harness": h.name,
                 "error": str(e)[:800]})
        return f"ERROR: bridge turn failed ({e}). Please continue from your last state."
    _state["session_id"] = r.get("session_id") or _state["session_id"]

    # Submission gate ($BRIDGE_SUBMISSION_GATE): a <final_law> that cannot
    # reproduce the agent's OWN collected data bounces back ONCE per run;
    # anything submitted after the bounce is accepted (the gate forces
    # engagement with the anomaly, it does not block indefinitely).
    if os.environ.get("BRIDGE_SUBMISSION_GATE") and not _state.get("gate_bounced"):
        try:
            sigma = float(os.environ.get("BRIDGE_NOISE_STD") or 0) or None
            g = instrument.gate_check(messages, r.get("text") or "", sigma)
            if g:
                _ledger({"round": _state["round"], "gate": g})
            if g and g["action"] == "bounce":
                _state["gate_bounced"] = True
                # keep the rejected turn's spend on the books
                _ledger({"round": _state["round"], "gate_rejected_turn": {
                    "cost_usd": r.get("cost_usd"), "usage": r.get("usage"),
                    "num_turns": r.get("num_turns")}})
                r = h.turn(instrument.gate_bounce_message(g), system,
                           _state["session_id"], cwd=AGENT_WORK,
                           timeout=TURN_TIMEOUT_S)
                _state["session_id"] = r.get("session_id") or _state["session_id"]
        except Exception as e:
            _ledger({"round": _state["round"], "gate_error": str(e)[:300]})
    # official agent appends our reply as one assistant message before the next call
    _state["fwd_len"] = len(messages) + 1
    _snapshot(_state["round"])
    entry = {
        "round": _state["round"],
        "harness": h.name,
        "model": MODEL,
        "session_id": _state["session_id"],
        "duration_s": round(time.time() - t0, 1),
        "num_turns": r.get("num_turns"),
        "cost_usd": r.get("cost_usd"),
        "usage": r.get("usage"),
        "is_error": r.get("is_error"),
        "prompt_chars": len(prompt),
        "reply_chars": len(r.get("text") or ""),
    }
    if readings is not None:
        entry["instrument"] = readings
    # codex reports cumulative session totals; its adapter converts "usage"
    # to per-turn deltas and passes the raw total through for the record
    if r.get("usage_cumulative") is not None:
        entry["usage_cumulative"] = r["usage_cumulative"]
    _ledger(entry)
    if r.get("is_error"):
        return f"ERROR: agent session errored: {str(r.get('text'))[:500]}"
    return r.get("text") or ""


def _oneshot_call(model, messages, system):
    """Judge path: always Claude Code, all tools disabled, throwaway neutral
    cwd — independent of the measured harness so scores stay comparable, and
    nothing in the judge's environment (the cwd path used to contain the
    arm/world tag) identifies whose output is being graded.

    The model id is passed through VERBATIM — no alias mapping. (Until
    2026-07-15 "claude-opus-4-6" was collapsed to the CLI alias "opus", which
    actually resolved to claude-opus-4-8; all fable/codex-sol scores were
    graded by opus-4-8, consistently across arms.)"""
    parts = ([f"[system]\n{system}"] if system else []) + [
        str(m["content"]) for m in messages]
    cwd = tempfile.mkdtemp(prefix="dp-judge-")
    # judge must hit the REAL Anthropic API even when the agent runs on a
    # provider override (BRIDGE_ENV_FILE / ANTHROPIC_* in the parent env)
    judge_env = {k: v for k, v in os.environ.items()
                 if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                              "ANTHROPIC_API_KEY")}
    res = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--model", model,
         "--tools", ""],
        input="\n\n".join(parts), capture_output=True, text=True,
        timeout=600, cwd=cwd, env=judge_env)
    if not res.stdout.strip():
        raise RuntimeError(f"empty claude output; stderr: {res.stderr[-500:]}")
    data = json.loads(res.stdout)
    _ledger({"oneshot_model": model, "judge_cwd": cwd,
             "model_used": sorted((data.get("modelUsage") or {}).keys()),
             "cost_usd": data.get("total_cost_usd")})
    if data.get("is_error"):
        raise RuntimeError(f"one-shot call errored: {data.get('result')}")
    return data.get("result") or ""


def patched_complete(model, messages, system=None, max_tokens=None, **kwargs):
    if model == "cc-session":
        return _session_call(messages, system)
    return _oneshot_call(model, messages, system)
