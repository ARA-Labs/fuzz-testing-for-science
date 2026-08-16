"""Harness adapter interface.

A "harness" is a coding-agent CLI that can run a PERSISTENT session (the
measured agent keeps working in one workspace across many bridge turns) and
be resumed by a session id.

To add one: subclass Harness in a new module here, implement turn(), and
register the class in harnesses/__init__.py. turn() must return a dict:

    text        str        assistant's final reply for this turn (fed back
                           into the official discovery loop)
    session_id  str|None   id to resume the same session on the next turn
    cost_usd    float|None turn cost if the CLI reports it (else None; the
                           ledger/meta rollup tolerates missing cost)
    usage       dict|None  flat token counters if reported
    num_turns   int|None   internal agent turns if reported
    is_error    bool

Raise on transport failures (timeout, empty output, unparseable JSON) — the
bridge catches, writes an error ledger line, and tells the discovery loop to
continue from its last state.
"""


class Harness:
    name = "base"
    # filename the CLI reads for standing workspace rules (CLAUDE.md,
    # AGENTS.md, ...) — used by launch_sweep.sh when copying the scaffold
    instructions_file = "AGENTS.md"

    def __init__(self, model, effort=""):
        self.model = model
        self.effort = effort

    def turn(self, prompt, system, session_id, cwd, timeout):
        """Run one agent turn in `cwd`.

        session_id None => start a fresh session and inject `system` at
        session start (however the CLI supports that); otherwise resume.
        """
        raise NotImplementedError
