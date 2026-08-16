#!/usr/bin/env python3
"""
Replay a DiscoverPhysics archive run and compute epistemic-progress signals.

For each archived run (episode.json) this script reconstructs, per round:
  1. surprise  — sigma-normalized prediction error of the agent's then-current
                 candidate law on the experiments of that round
                 (candidate = null "no force" model before the first fit)
  2. progress  — for each candidate law, surprise on ALL data observed up to
                 that point (an online-computable intermediate signal)
  3. protected — held-out error of each candidate on the official evaluator
                 probe cases against the noise-free simulator (never visible
                 to the agent). Final candidates go through the official
                 evaluator class verbatim (incl. evaluator-side param refit),
                 so the last point replicates the official score bit-exact.

Usage:  python3 replay_signals.py <run_dir> [--out out.json]
        run_dir e.g. ~/dp-archive/fable/gravity
"""

import argparse
import functools
import os
import json
import math
import sys
from pathlib import Path

import numpy as np

SA_ROOT = Path(
    os.environ.get("DISCOVERPHYSICS_ROOT", str(Path.home() / "bench/DiscoverPhysics"))
) / "ScienceAgent"
sys.path.insert(0, str(SA_ROOT))

from scienceagent import evaluator as EV  # noqa: E402
from scienceagent.evaluator import (  # noqa: E402
    _compile_law,
    _extract_training_trajectories,
    _wrap_with_timeout,
    clean_law_source,
)
from scienceagent.worlds import get_world  # noqa: E402

NULL_LAW_2P = """
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    # Null model: no forces, straight-line inertial motion.
    return (
        [pos2[0] + velocity2[0] * duration, pos2[1] + velocity2[1] * duration],
        [velocity2[0], velocity2[1]],
    )
"""

NULL_LAW_MULTI = """
def discovered_law(positions, velocities, duration, masses=None):
    # Null model: no forces, straight-line inertial motion for every particle.
    return [
        [p[0] + v[0] * duration, p[1] + v[1] * duration]
        for p, v in zip(positions, velocities)
    ]
"""

# ---------------------------------------------------------------------------
# world protocol adapters
# ---------------------------------------------------------------------------

TWO_PARTICLE_WORLDS = {
    "gravity", "yukawa", "fractional", "oscillator",
    "extra_dimensions", "coulomb_easy",
}

MULTI = {
    # world: (official evaluator class name, test-case list name,
    #         passes training_trajectories, uses masses kwarg)
    "circle": ("CircleEvaluator", "_CIRCLE_TEST_CASES", True, False),
    "three_species": ("ThreeSpeciesEvaluator", "_THREE_SPECIES_TEST_CASES", False, False),
    "dark_matter": ("DarkMatterEvaluator", "_DARK_MATTER_TEST_CASES", False, False),
    "ether": ("EtherEvaluator", "_ETHER_TEST_CASES", True, True),
    "hubble": ("HubbleEvaluator", "_HUBBLE_TEST_CASES", True, True),
}


def make_law(source, params):
    """Compile a candidate; returns (law, None) or (None, error_str)."""
    try:
        fn = _wrap_with_timeout(_compile_law(clean_law_source(source)))
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if params:
        fn = functools.partial(fn, **params)
    return fn, None


def multi_init(world, executor, case, bg_source):
    """Reconstruct the full initial state for a multi-particle law call.

    bg_source: the executor output dict this case corresponds to (experiment
    observation or noise-free ground truth) — used for background layouts the
    agent does not choose. Returns (law_kwargs_without_duration, score_slice).
    """
    if world == "circle":
        ring_radius = float(case.get("ring_radius", 5.0))
        v_tang = float(case.get("initial_tangential_velocity", 0.0))
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring_pos = np.column_stack(
            [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
        )
        pos = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        vel = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()
        return {"positions": pos, "velocities": vel}, slice(0, 11)

    probe_pos = np.asarray(case["probe_positions"], dtype=float)
    probe_vel = np.asarray(case["probe_velocities"], dtype=float)

    if world == "three_species":
        bg = np.asarray(bg_source["background_initial_positions"], dtype=float)
        pos = np.vstack([bg, probe_pos]).tolist()
        vel = np.vstack([np.zeros((executor.N_BACKGROUND, 2)), probe_vel]).tolist()
        return {"positions": pos, "velocities": vel}, slice(0, 35)

    if world == "dark_matter":
        bg = np.asarray(bg_source["background_initial_positions"], dtype=float)
        sign = float(case.get("visible_velocity_sign", 1.0))
        vis_vel = sign * np.asarray(executor._visible_velocities)
        pos = np.vstack([bg, probe_pos]).tolist()
        vel = np.vstack([vis_vel, probe_vel]).tolist()
        n_vis = executor.N_VISIBLE
        return {"positions": pos, "velocities": vel}, slice(n_vis, n_vis + executor.N_PROBES)

    # ether / hubble
    bg_pos = np.asarray(executor._bg_positions_rel)
    bg_vel = np.asarray(executor._bg_velocities)
    bg_mass = np.asarray(executor._bg_masses)
    probe_mass = np.asarray(
        case.get("probe_masses", [executor.DEFAULT_PROBE_MASS] * executor.N_PROBES),
        dtype=float,
    )
    pos = np.vstack([bg_pos, probe_pos]).tolist()
    vel = np.vstack([bg_vel, probe_vel]).tolist()
    masses = np.concatenate([bg_mass, probe_mass]).tolist()
    return {"positions": pos, "velocities": vel, "masses": masses}, slice(21, 26)


def predict_2p(law, case):
    """Predicted pos2 at each measurement time (None where the law fails)."""
    preds = []
    for t in case["measurement_times"]:
        try:
            p2_out, _ = law(
                pos1=[0.0, 0.0],
                pos2=case["pos2"],
                p1=case["p1"],
                p2=case["p2"],
                velocity2=case["velocity2"],
                duration=t,
            )
            preds.append([float(p2_out[0]), float(p2_out[1])])
        except Exception:
            preds.append(None)
    return preds


def sq_errors(world, executor, law, case, observed, uses_masses):
    """Per-(particle,time) squared L2 errors of `law` vs `observed` positions,
    restricted to the world's scoring slice. Returns list of floats
    (skips timesteps where the law fails)."""
    errs = []
    if world in TWO_PARTICLE_WORLDS:
        preds = predict_2p(law, case)
        for pred, o in zip(preds, observed["pos2"]):
            if pred is None:
                continue
            errs.append((pred[0] - o[0]) ** 2 + (pred[1] - o[1]) ** 2)
        return errs

    kwargs, sl = multi_init(world, executor, case, observed)
    if not uses_masses:
        kwargs.pop("masses", None)
    obs_pos = np.asarray(observed["positions"], dtype=float)  # (T, N, 2)
    for j, t in enumerate(case["measurement_times"]):
        try:
            out = np.asarray(law(duration=float(t), **kwargs), dtype=float)
            diff = out[sl] - obs_pos[j, sl]
            errs.extend(np.sum(diff * diff, axis=-1).tolist())
        except Exception:
            continue
    return errs


def surprise_stats(world, executor, law, experiments, sigma, uses_masses):
    """chi2 per point = err^2/(2 sigma^2); expectation 1 for an exact law."""
    chi2 = []
    for case, obs in experiments:
        for e in sq_errors(world, executor, law, case, obs, uses_masses):
            chi2.append(e / (2.0 * sigma * sigma))
    if not chi2:
        return None
    a = np.asarray(chi2)
    return {
        "n_points": int(a.size),
        "surprise_median": float(math.sqrt(np.median(a))),
        "surprise_mean": float(math.sqrt(np.mean(a))),
        "surprise_max": float(math.sqrt(np.max(a))),
    }


def protected_error(world, executor, law, test_cases, uses_masses):
    """Mean squared L2 error on the official held-out probes, noise-free."""
    with executor.noise_disabled():
        gts = executor.run(test_cases)
    errs = []
    for case, gt in zip(test_cases, gts):
        if world in TWO_PARTICLE_WORLDS:
            preds = predict_2p(law, case)
            gt_pos2 = np.asarray(gt["pos2"])
            for pred, g in zip(preds, gt_pos2):
                if pred is None:
                    return None
                errs.append((pred[0] - g[0]) ** 2 + (pred[1] - g[1]) ** 2)
        else:
            got = sq_errors(world, executor, law, case, gt, uses_masses)
            want = len(case["measurement_times"])
            sl_n = None  # sq_errors already sliced; count check below
            if not got:
                return None
            errs.extend(got)
    return float(np.mean(errs)) if errs else None


def official_final_score(world, executor, source, rounds):
    """Run the untouched official evaluator on a final law."""
    name, cases_name, wants_training, _ = MULTI.get(
        world, ("Evaluator", "_DEFAULT_TEST_CASES", True, False)
    )
    ev_cls = getattr(EV, name)
    kwargs = {"verbose": False}
    if wants_training:
        kwargs["training_trajectories"] = _extract_training_trajectories(rounds)
    try:
        res = ev_cls(executor).evaluate(source, **kwargs)
        mpe = res.get("mean_pos_error")
        if mpe is None or not math.isfinite(mpe):
            return None
        return mpe
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ep = json.loads((args.run_dir / "episode.json").read_text())
    world = ep["world"]
    sigma = ep["noise_std"]
    res_path = args.run_dir / "result.json"
    engine = "field"
    if res_path.exists():
        engine = json.loads(res_path.read_text()).get("engine", "field")
    executor = get_world(world, engine=engine, noise_std=0.0)["executor"]

    uses_masses = MULTI.get(world, (None, None, None, False))[3]
    test_cases = getattr(EV, MULTI.get(world, (None, "_DEFAULT_TEST_CASES"))[1]
                         if world in MULTI else "_DEFAULT_TEST_CASES")
    null_src = NULL_LAW_2P if world in TWO_PARTICLE_WORLDS else NULL_LAW_MULTI

    # --- walk rounds: collect experiments and the candidate-law timeline ---
    candidates = [{"label": "null", "round": 0, "source": null_src, "params": {}}]
    skipped = []
    observed = []          # [(case, obs), ...] in round order
    observed_rounds = []   # matching round index per (case, obs)
    per_round_surprise = []

    for r in ep["rounds"]:
        k = r["round"]
        if r["action"] == "experiment" and r.get("experiment_output"):
            batch = [
                (i, o)
                for i, o in zip(r["experiment_input"], r["experiment_output"])
                if isinstance(o, dict)
            ]
            cur = candidates[-1]
            law, _ = make_law(cur["source"], cur["params"])
            stats = (
                surprise_stats(world, executor, law, batch, sigma, uses_masses)
                if law else None
            )
            per_round_surprise.append(
                {"round": k, "candidate": cur["label"], **(stats or {"n_points": 0})}
            )
            observed.extend(batch)
            observed_rounds.extend([k] * len(batch))
        elif r["action"] == "experiment":
            per_round_surprise.append(
                {"round": k, "candidate": candidates[-1]["label"],
                 "n_points": 0, "note": "experiment failed"}
            )
        elif r["action"] == "mse_fit" and r.get("mse_fit_input"):
            out = r.get("mse_fit_output") or {}
            params = out.get("fitted_params") or {}
            label = f"fit_r{k}" + ("" if not out.get("error") else "_initparams")
            cand = {"label": label, "round": k, "source": r["mse_fit_input"],
                    "params": params,
                    "agent_seen_loss": out.get("loss_after"),
                    "fit_error": out.get("error")}
            law, cerr = make_law(cand["source"], cand["params"])
            if cerr:
                skipped.append({"label": label, "round": k, "compile_error": cerr})
            else:
                candidates.append(cand)
        elif r["action"] == "final_law" and r.get("final_law"):
            cand = {"label": f"final_r{k}", "round": k,
                    "source": r["final_law"], "params": {}}
            law, cerr = make_law(cand["source"], cand["params"])
            if cerr:
                skipped.append({"label": cand["label"], "round": k,
                                "compile_error": cerr})
            else:
                candidates.append(cand)

    # --- progress: each candidate vs all data observed by its round,
    #     plus its protected held-out error ---
    progress = []
    for c in candidates:
        if c["round"] == 0:
            upto = observed
        else:
            upto = [
                pair for rk, pair in zip(observed_rounds, observed)
                if rk <= c["round"]
            ]
        law, _ = make_law(c["source"], c["params"])
        stats = (
            surprise_stats(world, executor, law, upto, sigma, uses_masses)
            if (law and upto) else None
        )
        if c["label"].startswith("final"):
            prot = official_final_score(world, executor, c["source"], ep["rounds"])
        else:
            prot = (
                protected_error(world, executor, law, test_cases, uses_masses)
                if law else None
            )
        progress.append(
            {"candidate": c["label"], "round": c["round"],
             "agent_seen_loss": c.get("agent_seen_loss"),
             "surprise_on_observed": stats,
             "protected_mean_pos_error": prot}
        )

    expl = None
    meta_path = args.run_dir / "meta.json"
    if meta_path.exists():
        expl = json.loads(meta_path.read_text()).get("explanation_score")

    fin = ep["evaluation"].get("mean_pos_error") if ep.get("evaluation") else None
    result = {
        "run_dir": str(args.run_dir),
        "world": world,
        "noise_std": sigma,
        "final_eval_mean_pos_error": fin,
        "mse_criterion_passed": ep["evaluation"].get("passed") if ep.get("evaluation") else None,
        "explanation_score": expl,
        "per_round_surprise": per_round_surprise,
        "progress": progress,
        "skipped_candidates": skipped,
    }
    out = args.out or (Path(__file__).parent / f"signals_{world}.json")
    out.write_text(json.dumps(result, indent=2))

    fin_s = "None" if fin is None else f"{fin:.3g}"
    print(f"world={world}  sigma={sigma:.4f}"
          f"  mse_pass={result['mse_criterion_passed']}"
          f"  explanation={expl}  final mean_pos_error={fin_s}\n")
    print("Per-round surprise (candidate active at that round vs that round's data):")
    for row in per_round_surprise:
        s = row.get("surprise_median")
        mx = row.get("surprise_max")
        print(f"  round {row['round']:>2}  cand={row['candidate']:<12}"
              f"  median={'      -' if s is None else f'{s:7.2f}'}sigma"
              f"  max={'      -' if mx is None else f'{mx:8.2f}'}  n={row['n_points']}")
    print("\nCandidate progress (vs all data seen so far | protected probes):")
    for row in progress:
        st = row["surprise_on_observed"]
        med = "      -" if not st else f"{st['surprise_median']:7.2f}"
        prot = row["protected_mean_pos_error"]
        prot_s = "   FAILED" if prot is None else f"{prot:9.3g}"
        seen = row["agent_seen_loss"]
        seen_s = "     -" if seen is None else f"{seen:6.2f}"
        print(f"  {row['candidate']:<18} round={row['round']:>2}"
              f"  surprise_med={med}sigma  agent_seen_loss={seen_s}"
              f"  protected_mpe={prot_s}")
    if skipped:
        print("\nSkipped (uncompilable) candidates:")
        for s in skipped:
            print(f"  round {s['round']}: {s['label']} — {s['compile_error'][:100]}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
