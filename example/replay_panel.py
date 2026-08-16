#!/usr/bin/env python3
"""Replay the instrument panel over an archived run — no API keys needed.

Reconstructs, round by round, the exact <instrument_panel> text the bridge
injects into the agent's context, from nothing but the run's episode.json.
On the final round it also runs the submission gate against the submitted law.

Usage:
    python example/replay_panel.py data/three_species/with_instruments
    python example/replay_panel.py data/three_species/without_instruments

The envelope reading works with the standard library alone. The surprise
reading and the gate additionally need numpy plus the DiscoverPhysics package
(for compiling the agent's candidate laws); point DISCOVERPHYSICS_ROOT at a
checkout of https://github.com/SampsonML/DiscoverPhysics to enable them.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
dp = os.path.expanduser(os.environ.get("DISCOVERPHYSICS_ROOT", "~/bench/DiscoverPhysics"))
sys.path.insert(0, os.path.join(dp, "ScienceAgent"))

import instrument  # noqa: E402


def messages_up_to(episode, round_no):
    """Rebuild the official conversation prefix the bridge would have seen."""
    msgs = [{"role": "user", "content": episode.get("mission", "")}]
    for r in episode["rounds"]:
        if r["round"] > round_no:
            break
        msgs.append({"role": "assistant", "content": r.get("llm_reply") or ""})
        if r.get("experiment_output") is not None:
            msgs.append({"role": "user", "content": "<experiment_output>\n"
                         + json.dumps(r["experiment_output"]) + "\n</experiment_output>"})
        if r.get("mse_fit_output") is not None:
            msgs.append({"role": "user", "content": "<mse_fit_output>\n"
                         + json.dumps(r["mse_fit_output"]) + "\n</mse_fit_output>"})
    return msgs


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/three_species/with_instruments")
    episode = json.loads((run_dir / "episode.json").read_text())
    os.environ["BRIDGE_INSTRUMENT"] = "1"
    os.environ["BRIDGE_NOISE_STD"] = str(episode["noise_std"])
    os.environ["BRIDGE_WORLD"] = episode["world"]

    print(f"world={episode['world']}  sigma={episode['noise_std']}  rounds={len(episode['rounds'])}\n")

    final_law = None
    for r in episode["rounds"]:
        panel, readings = instrument.maybe_panel(messages_up_to(episode, r["round"] - 1))
        print(f"════ round {r['round']} ({r['action']}) — panel injected before this round:")
        print(panel if panel else "(no readings yet — nothing collected)")
        print()
        if r.get("final_law"):
            final_law = r["final_law"]

    if final_law:
        msgs = messages_up_to(episode, len(episode["rounds"]))
        g = instrument.gate_check(msgs, f"<final_law>\n{final_law}\n</final_law>",
                                  episode["noise_std"])
        print("════ submission gate on the final law:")
        if g is None:
            print("(gate could not evaluate — DiscoverPhysics not installed?)")
        else:
            print(json.dumps(g, indent=2))
            if g["action"] == "bounce":
                print("\nBounce message the agent would receive:\n")
                print(instrument.gate_bounce_message(g))


if __name__ == "__main__":
    main()
