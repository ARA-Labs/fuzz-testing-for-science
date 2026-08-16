"""Entry point: run the official DiscoverPhysics discovery loop with a
coding-agent harness bridge patched in (Claude Code / Codex / ... — pick
with BRIDGE_HARNESS; see bridge_client.py for all env vars).

Usage (from ~/bench, venv python):
    BRIDGE_RUN_DIR=~/bench/bridge/runs/<name> BRIDGE_HARNESS=claude-code \
    BRIDGE_MODEL=sonnet \
    venv/bin/python bridge/run_cc.py --world gravity --model cc-session \
        --max-rounds 10 --store_output <run_dir>/episode

Old CC_MODEL/CC_AGENT_WORK env names still work. All argv after the script
name is forwarded verbatim to run_discovery.py. NOTE: launch with
cwd=~/bench/DiscoverPhysics (agent.py resolves prompt templates from cwd).
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_client  # noqa: E402

import scienceagent.llm_client as llm_client  # noqa: E402

llm_client.complete = bridge_client.patched_complete

DP_ROOT = os.path.expanduser(
    os.environ.get("DISCOVERPHYSICS_ROOT", "~/bench/DiscoverPhysics"))
RUN_DISCOVERY = os.path.join(DP_ROOT, "ScienceAgent", "run_discovery.py")

sys.argv = [RUN_DISCOVERY] + sys.argv[1:]
runpy.run_path(RUN_DISCOVERY, run_name="__main__")
