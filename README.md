# 🧪 Fuzz Testing for Science
### Give your AI scientist the fuzzer's feedback loop

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.09855-b31b1b.svg)](https://arxiv.org/abs/2608.09855)
[![Blog](https://img.shields.io/badge/Essay-The%20Goal%20of%20Science%20Is%20Not%20to%20Win-5d7045.svg)](https://www.agenticresearch.sh/blog/the-goal-of-science-is-not-to-win)
[![Demo](https://img.shields.io/badge/Demo-video-purple.svg)](media/demo.mp4)
[![Benchmark](https://img.shields.io/badge/Benchmark-DiscoverPhysics-e8a878.svg)](https://arxiv.org/abs/2605.26087)

> **Autonomous research agents already generate experiments faster than anyone can validate them — and the only feedback they get is the final score.**
> Between submissions they fly blind: they treat every failure as an optimization problem, refit ever-fancier equations on the same stale evidence, and grind the same corner of the world until the clock runs out. A greybox fuzzer solved this exact problem decades ago — almost no run finds a bug, but *every* run returns a dense, cheap signal that steers the next input. This repository wires that feedback loop into a live physics-discovery agent, and shows it flipping a failed discovery into a perfect score.

<p align="center"><img src="media/verdict.png" width="760" alt="Verdict: explanation 0.4 FAIL without instruments vs 1.0 PASS with; trajectory error 90x more accurate"></p>

---

## The pain this solves

Today's auto-research loop is **generate-and-rank**: scale the generator, run everything, keep whatever scores best. Three things quietly break:

1. **Sparse feedback.** The agent finds out whether it was pointed in a useful direction only at the final, expensive evaluation. Until then it is *sampling*, not searching.
2. **Optimizing the answer instead of the evidence.** With only a fit metric in view, every failure looks like a curve-fitting problem. Our control run revised its law **twelve times** — exotic fields, screened forces, Gaussian bumps — while launching probe after probe into territory it had already mapped.
3. **Nothing protects the verdict.** If the agent grades itself on its own signals, it overfits them exactly the way the field overfits public benchmarks. Guidance and judgment have to be different instruments.

## What this repo adds

Three oracle-free instruments — computed **only from the agent's own experiments and candidates**. The simulator's hidden state never crosses the wall, and the judge never sees the gauges.

**1 — Surprise** (σ-calibrated self-consistency): *does your law explain your own data, down to the noise floor?*

```
<instrument_panel>
- SURPRISE: your latest candidate law vs all 240 data points collected so far:
  median residual = 0.87 sigma, max = 5003786 sigma. Reading guide: ~1 sigma means
  the theory explains your data down to the noise floor; >2 sigma is a systematic
  misfit — that is not noise.
```

**2 — Envelope** (exploration ledger): *where have your experiments actually been — and where have they never been?*

```
- ENVELOPE explored so far (24 cases): initial radius 2.0–8.0; initial speed: 79% zero;
  time horizon <= 10. This is NOT a limit — it is the region your experiments have
  covered. Your law is only tested INSIDE it; outside it, it is unconstrained.
```

**3 — Submission gate** (protected validation, in miniature): a final law that cannot reproduce the agent's own observations bounces back once, with the exact offending configuration — revise, run the discriminating experiment, or defend the anomaly in writing.

```
<submission_gate>
Your <final_law> has NOT been accepted yet. Scored against ALL the data you collected
this run: median 0.88 sigma, max = 7093 sigma. Worst offender: experiment
{p1: 20, pos2: [40, 0]} at t=50. A law that cannot reproduce your own observations
is not yet a discovery.
```

The division of labor is the paper's central claim: **guidance signals steer; the protected verdict (held-out trajectories + a judge the agent never meets) still decides.**

## Does it work?

One world of [DiscoverPhysics](https://arxiv.org/abs/2605.26087): `three_species`, where 30 background particles secretly form three species with couplings **+1 / +3 / −2**. One agent: GPT-5.6-Sol, bare workspace, same budget, same noise. The only difference between the two runs in [`data/three_species/`](data/three_species/) is the instruments.

| | without instruments | with instruments |
|---|---|---|
| explanation (judge) | 0.4 — **FAIL** | **1.0 — PASS** |
| trajectory error (norm.) | 0.082 | **0.0009** (90× better) |
| what it found | "a single shared field; every particle responds identically" | **all three species, strengths +1 / +3 / −2** |

The pivot is on the record. Reacting to the panel in round 4, the winning agent wrote:

> *"The NaN-contaminated cases make **the aggregate residual diagnostic** unusable, while
> the valid-case fit is **far above the noise scale**; I will therefore test whether the
> missing structure is **species-dependent source and response charge**, rather than collect
> more redundant trajectories."*

"The aggregate residual diagnostic" is the instrument panel, named by the agent itself.
Eleven rounds later it submitted the exact law and scored 1.0.

▶ **[media/demo.mp4](media/demo.mp4)** — a walkthrough: the world in motion, both runs
step by step, every number and quote from these archived runs.
The companion essay: [*The Goal of Science Is Not to Win*](https://www.agenticresearch.sh/blog/the-goal-of-science-is-not-to-win).

---

## How to use it

**Replay the demo data (no keys, no setup):**

```bash
python example/replay_panel.py data/three_species/with_instruments
```

This prints the exact panel the agent received before every round, and the gate's
verdict on the submitted law. For the surprise line and the gate you additionally need
`numpy` and a checkout of [DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics)
(`export DISCOVERPHYSICS_ROOT=/path/to/DiscoverPhysics`); the envelope line runs on the
standard library alone.

**Run your own instrumented episode** (needs the DiscoverPhysics repo and a coding-agent
CLI such as `claude` or `codex` on PATH, with its API access configured):

```bash
export DISCOVERPHYSICS_ROOT=/path/to/DiscoverPhysics
BRIDGE_INSTRUMENT=1 BRIDGE_SUBMISSION_GATE=1 \
  bash harness/launch_sweep.sh \
    --harness codex --model gpt-5.6-sol --effort high \
    --tag my-instrumented --worlds "three_species" --template /path/to/blank-workspace
```

Drop the two `BRIDGE_*` variables and you have the control arm — that is the entire
experimental difference between the two runs in `data/`.

### How the agent is wired

The bare agent is deliberately boring: a coding-agent CLI in an empty workspace,
driven by DiscoverPhysics' official experiment protocol. `bridge_client.py` forwards
each round's messages to the CLI and does exactly two extra things when instrumented:

1. after the experiment results, it appends the `<instrument_panel>` computed by
   `instrument.py` from the conversation so far (plus one sentence requiring the agent
   to state what the readings imply — information that is never read is not feedback);
2. when the reply contains a `<final_law>`, it scores that law against every logged
   observation and bounces it once if the residuals say the law contradicts the
   agent's own data.

### Repository layout

```
harness/
  instrument.py        the panel + the gate (the paper's loop, ~400 lines, zero oracle)
  bridge_client.py     routes DiscoverPhysics' agent loop through a coding-agent CLI
                       (Claude Code / Codex / ...) and injects the panel each round
  harnesses/           per-CLI adapters
  run_cc.py            entry point: official DiscoverPhysics loop + bridge patched in
  launch_sweep.sh      one-command launcher (tmux session per world)
example/
  replay_panel.py      replay the panel round-by-round from an archived run — no API keys
analysis/
  replay_signals.py    recompute surprise + held-out curves for any archived run
  breadth.py           per-knob exploration-envelope statistics
data/three_species/
  without_instruments/ the bare run   (episode.json, result.json, ledger, log)
  with_instruments/    the winning run (same files; the panel text it saw is
                       reproducible from episode.json via example/replay_panel.py)
media/                 demo video + stills
```

## Citation

```bibtex
@article{he2026agentic,
  title   = {Agentic Auto-Research is Fuzz Testing},
  author  = {He, Yifeng and Wang, Jicheng and Zhao, Yinzhe and Liu, Jiachen and Chen, Hao},
  journal = {arXiv preprint arXiv:2608.09855},
  year    = {2026}
}
```

The benchmark: *DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific
Thinking* (arXiv:2605.26087). Archived trajectories for the wider replay study are
public under the [AgentNativeResearchLab](https://huggingface.co/AgentNativeResearchLab)
Hugging Face org.

## License

MIT.

---

<p align="center">
  <a href="https://www.agenticresearch.sh">
    <img src="https://raw.githubusercontent.com/ARA-Labs/brand/main/out/ara-lab-lockup-horizontal-600.png" width="220" alt="ARA Lab">
  </a>
  <br/>
  <sub>Built at <a href="https://www.agenticresearch.sh">ARA Lab</a> — infrastructure for rigorous, trustworthy AI scientists.</sub>
</p>
