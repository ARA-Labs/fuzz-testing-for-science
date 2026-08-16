#!/usr/bin/env python3
"""Exploration-breadth (envelope) stats per archived run — signal #2.

Reads each run's experiment_input history from episode.json and reports the
span each interface knob was actually varied over. Purely a bookkeeping of
the agent's own actions; no world knowledge involved.

Usage: python3 breadth.py [arm ...]     (default: fable codex-sol opus48)
"""

import json
import math
import re
import sys
from pathlib import Path

WORLD_ORDER = [
    "gravity", "yukawa", "fractional", "oscillator", "extra_dimensions",
    "coulomb_easy", "circle", "three_species", "dark_matter", "ether", "hubble",
]


def _norm(v):
    return math.hypot(float(v[0]), float(v[1]))


def breadth(run_dir):
    ep = json.loads((Path(run_dir) / "episode.json").read_text())
    radii, speeds, ratios, masses, tmax = [], [], [], [], []
    ncases = 0
    for r in ep["rounds"]:
        if r["action"] != "experiment" or not r.get("experiment_input"):
            continue
        for c in r["experiment_input"]:
            ncases += 1
            if "pos2" in c:
                radii.append(_norm(c["pos2"]))
                speeds.append(_norm(c["velocity2"]))
                ratios.append(float(c["p1"]) / float(c["p2"]))
            elif "ring_radius" in c:
                radii.append(float(c["ring_radius"]))
                speeds.append(abs(float(c.get("initial_tangential_velocity", 0))))
            elif "probe_positions" in c:
                radii.extend(_norm(p) for p in c["probe_positions"])
                speeds.extend(_norm(v) for v in c["probe_velocities"])
                if "probe_masses" in c:
                    masses.extend(float(m) for m in c["probe_masses"])
            if "measurement_times" in c:
                tmax.append(max(c["measurement_times"]))

    def logspan(xs):
        xs = [x for x in xs if x > 1e-9]
        return 0.0 if len(xs) < 2 else math.log10(max(xs) / min(xs))

    return {
        "world": ep["world"],
        "n_cases": ncases,
        "r_min": min(radii) if radii else None,
        "r_max": max(radii) if radii else None,
        "r_logspan": round(logspan(radii), 2),
        "v_nonzero_frac": (
            round(sum(1 for s in speeds if s > 1e-6) / len(speeds), 2)
            if speeds else None
        ),
        "v_max": round(max(speeds), 2) if speeds else None,
        "ratio_logspan": round(logspan(ratios), 2) if ratios else None,
        "mass_logspan": round(logspan(masses), 2) if masses else None,
        "t_logspan": round(logspan(tmax), 2),
    }


def parse_scoreboard(p):
    rows = {}
    for line in Path(p).read_text().splitlines():
        m = re.match(
            r"\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.e-]+)\s*\|\s*([\d.]+)\s*"
            r"\|\s*(PASS|FAIL)\s*\|", line)
        if m:
            rows[m.group(1)] = m.group(5)
    return rows


def main():
    arms = sys.argv[1:] or ["fable", "codex-sol", "opus48"]
    archive = Path.home() / "dp-archive"
    print(f"{'world':<17} {'arm':<10} {'verdict':<8} {'r range':<16} "
          f"{'r_span':<7} {'v%!=0':<6} {'v_max':<6} {'p-ratio':<8} {'t_span':<6} n")
    for w in WORLD_ORDER:
        for arm in arms:
            d = archive / arm / w
            if not (d / "episode.json").exists():
                continue
            sb = archive / arm / "SCOREBOARD.md"
            v = parse_scoreboard(sb).get(w, "?") if sb.exists() else "?"
            b = breadth(d)
            rr = (f"{b['r_min']:.2g}-{b['r_max']:.3g}"
                  if b["r_min"] is not None else "-")
            print(f"{w:<17} {arm:<10} {v:<8} {rr:<16} {b['r_logspan']:<7} "
                  f"{b['v_nonzero_frac']!s:<6} {b['v_max']!s:<6} "
                  f"{b['ratio_logspan']!s:<8} {b['t_logspan']:<6} {b['n_cases']}")


if __name__ == "__main__":
    main()
