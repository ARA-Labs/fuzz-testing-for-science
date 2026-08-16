#!/usr/bin/env bash
# Generic DiscoverPhysics sweep launcher — one tmux session per world, any
# harness (claude-code / codex / kimi-code / agy; see bridge/harnesses/).
# Supersedes launch_fable_sweep.sh (kept for reference; same conventions).
#
# Usage:
#   bash bridge/launch_sweep.sh --harness codex --model gpt-5-codex --tag codex-g5
#   bash bridge/launch_sweep.sh --harness claude-code --model fable --tag fable   # = old fable sweep
#
# Options (defaults mirror the fable production sweep):
#   --harness NAME     claude-code | codex | kimi-code | agy   (default claude-code)
#   --model MODEL      model string passed to the harness CLI  (required)
#   --tag TAG          agent tag: names run dirs, workspaces, tmux, archive
#                      (default: <harness>-<model>)
#   --worlds "a b c"   subset of worlds (default: all 11)
#   --effort E         reasoning effort, if the CLI supports it
#   --template DIR     scaffold source (default ~/agent-work: CLAUDE.md + .claude/)
#   --max-rounds N     default 16
#   --noise-frac F     default 0.075  (sigma = F*sqrt(Var_world))
#   --noise-seed N     default 0
#   --stagger S        seconds between session starts (default 30)
#
# Layout:
#   workspace  ~/<tag>-bench/<world>/agent-work
#   run dir    ~/bench/bridge/runs/<tag>-<world>   (skipped if result.json exists)
#   tmux       dp-<tag>-<world>
#
# Scaffold per harness: claude-code gets CLAUDE.md + .claude/ verbatim; other
# harnesses get the same rules as AGENTS.md plus .claude/skills/*/SKILL.md
# flattened into skills/ (no Skill tool outside Claude Code — check the rules
# text still makes sense for the target harness before a paid sweep!).
set -euo pipefail

BENCH="${BENCH:-$HOME/bench}"
DP_ROOT="${DISCOVERPHYSICS_ROOT:-$BENCH/DiscoverPhysics}"
HARNESS=claude-code
MODEL=""
TAG=""
EFFORT=""
TEMPLATE=""
MAX_ROUNDS=16
NOISE_FRAC=0.075
NOISE_SEED=0
STAGGER=30
WORLDS_STR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --harness)    HARNESS=$2; shift 2;;
    --model)      MODEL=$2; shift 2;;
    --tag)        TAG=$2; shift 2;;
    --worlds)     WORLDS_STR=$2; shift 2;;
    --effort)     EFFORT=$2; shift 2;;
    --template)   TEMPLATE=$2; shift 2;;
    --max-rounds) MAX_ROUNDS=$2; shift 2;;
    --noise-frac) NOISE_FRAC=$2; shift 2;;
    --noise-seed) NOISE_SEED=$2; shift 2;;
    --stagger)    STAGGER=$2; shift 2;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 1; }
TAG=${TAG:-"$HARNESS-$MODEL"}
SWEEP_ROOT="$HOME/$TAG-bench"

# scaffold template: harness-adapted dir if one exists (skill-invocation
# wording differs per CLI — Claude has a Skill tool, the others read
# skills/<name>/SKILL.md), else the plain claude-code template
if [ -z "$TEMPLATE" ]; then
  if [ -d "$HOME/agent-work-$HARNESS" ]; then
    TEMPLATE="$HOME/agent-work-$HARNESS"
  else
    TEMPLATE="$HOME/agent-work"
  fi
fi
echo "template: $TEMPLATE"

# harness CLIs live outside the default PATH on this box
export PATH="$HOME/.local/bin:$HOME/.kimi-code/bin:$PATH"

# codex's bundled bwrap sandbox cannot start when unprivileged user
# namespaces are restricted (this Azure VM!) — writes then FAIL SILENTLY and
# the whole run is garbage, so refuse to launch.
if [ "$HARNESS" = codex ] && \
   [ "${BRIDGE_CODEX_SANDBOX:-workspace-write}" != danger-full-access ] && \
   [ "$(sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>/dev/null)" = 1 ]; then
  echo "ERROR: codex sandbox (bwrap) is broken on this machine" >&2
  echo "  (kernel.apparmor_restrict_unprivileged_userns=1). Either:" >&2
  echo "    export BRIDGE_CODEX_SANDBOX=danger-full-access   # no sandbox (isolated VM)" >&2
  echo "    sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0" >&2
  exit 1
fi

# fail fast on skeleton adapters / unknown harness names
"$BENCH/venv/bin/python" - "$HARNESS" "$BENCH/bridge" <<'EOF'
import sys
sys.path.insert(0, sys.argv[2])
from harnesses import get_harness
get_harness(sys.argv[1], "probe")
EOF

# Var(GT_world) copied from scripts/run_benchmark.py _WORLD_VARS
declare -A WORLD_VARS=(
  [gravity]=4.283
  [yukawa]=5.677
  [fractional]=5.721
  [dark_matter]=63.303
  [three_species]=28.717
  [ether]=21.259
  [hubble]=41.189
  [coulomb_easy]=4.244
  [extra_dimensions]=4.248
  [circle]=6.596
  [oscillator]=6.332
)
if [ -n "$WORLDS_STR" ]; then
  read -r -a WORLDS <<<"$WORLDS_STR"
else
  WORLDS=(gravity yukawa fractional dark_matter three_species ether hubble
          coulomb_easy extra_dimensions circle oscillator)
fi

i=0
for w in "${WORLDS[@]}"; do
  [ -n "${WORLD_VARS[$w]:-}" ] || { echo "unknown world: $w" >&2; exit 1; }
  SESS="dp-$TAG-$w"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "skip $w: tmux session $SESS already exists"
    continue
  fi

  AW="$SWEEP_ROOT/$w/agent-work"
  RUN="$BENCH/bridge/runs/$TAG-$w"
  if [ -e "$RUN/result.json" ]; then
    echo "skip $w: $RUN/result.json already exists"
    continue
  fi
  mkdir -p "$AW" "$RUN"

  if [ "$HARNESS" = claude-code ]; then
    cp "$TEMPLATE/CLAUDE.md" "$AW/"
    cp -r "$TEMPLATE/.claude" "$AW/"
  else
    cp "$TEMPLATE/CLAUDE.md" "$AW/AGENTS.md"
    # copy full skill dirs (SKILL.md + references/) — flattening SKILL.md alone
    # loses the references/ contracts the skills depend on
    if compgen -G "$TEMPLATE/.claude/skills/*/SKILL.md" >/dev/null; then
      mkdir -p "$AW/skills"
      for d in "$TEMPLATE"/.claude/skills/*/; do
        cp -r "$d" "$AW/skills/$(basename "$d")"
      done
    fi
  fi
  if [ ! -d "$AW/.git" ]; then
    git -C "$AW" init -q
    git -C "$AW" add -A
    git -C "$AW" -c user.email=bridge@local -c user.name=bridge \
      commit -q -m "scaffold + unit rules"
  fi

  sigma=$(python3 -c "import math; print(round($NOISE_FRAC*math.sqrt(${WORLD_VARS[$w]}), 6))")
  delay=$((i * STAGGER))

  # NOTE: cwd must be the DiscoverPhysics repo root — agent.py resolves the
  # prompt-template paths relative to repo-root-or-cwd (see 2026-07-14 notes).
  tmux new-session -d -s "$SESS" \
    "sleep $delay; export PATH=\"\$HOME/.local/bin:\$HOME/.kimi-code/bin:\$PATH\"; \
     cd '$DP_ROOT' && \
     BRIDGE_RUN_DIR='$RUN' BRIDGE_AGENT_WORK='$AW' \
     BRIDGE_HARNESS='$HARNESS' BRIDGE_MODEL='$MODEL' BRIDGE_EFFORT='$EFFORT' \
     BRIDGE_INSTRUMENT='${BRIDGE_INSTRUMENT:-}' \
     BRIDGE_SUBMISSION_GATE='${BRIDGE_SUBMISSION_GATE:-}' \
     BRIDGE_NOISE_STD=$sigma BRIDGE_WORLD=$w \
     BRIDGE_CODEX_SANDBOX='${BRIDGE_CODEX_SANDBOX:-workspace-write}' \
     BRIDGE_ENV_FILE='${BRIDGE_ENV_FILE:-}' \
     BRIDGE_TURN_TIMEOUT='${BRIDGE_TURN_TIMEOUT:-1800}' \
     '$BENCH/venv/bin/python' '$BENCH/bridge/run_cc.py' \
       --world $w --model cc-session \
       --max-rounds $MAX_ROUNDS \
       --noise-std $sigma --noise-seed $NOISE_SEED \
       --store_output '$RUN/episode' \
       --output '$RUN/result.json' \
       2>&1 | tee '$RUN/run.log'; \
     echo \"== $SESS finished (exit \$?) ==\"; sleep infinity"

  echo "launched $SESS  sigma=$sigma  start_delay=${delay}s  run_dir=$RUN"
  i=$((i + 1))
done

echo
echo "all sessions:"
tmux ls
