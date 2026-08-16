"""Oracle-free instrument panel for the B1 arm.

Computes two per-round readings from the official protocol messages ALONE —
the same information the agent has already seen — plus the observation-noise
sigma (instrument precision, $BRIDGE_NOISE_STD):

  SURPRISE  sigma-calibrated residual of the agent's latest fitted candidate
            against all data it has collected (noise floor = 1 sigma).
  ENVELOPE  span each experiment knob has actually been varied over.

No ground-truth simulator, no held-out probes, no judge access. The panel is
prepended to the next round's prompt by bridge_client when
$BRIDGE_INSTRUMENT is truthy.

Message formats parsed (defined by scienceagent/agent.py):
  assistant: <run_experiment>[...json cases...]</run_experiment>
             <run_mse_fit>law source</run_mse_fit>
  user:      <experiment_output>[...json results...]</experiment_output>
             <mse_fit_output>{...fit result json...}</mse_fit_output>
"""

import functools
import json
import math
import os
import re

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?|```")


def _tag(text, tag):
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(rf"<{tag}>(.*?)</{tag}>", _FENCE_RE.sub("", text), re.DOTALL)
    return m.group(1) if m else None


def parse_history(messages):
    """Walk the official conversation; return (experiments, fits).

    experiments: list of (input_case_dict, output_dict) pairs
    fits: list of {source, params, loss, error} in submission order
    """
    experiments, fits = [], []
    pending = None
    for m in messages:
        content = str(m.get("content", ""))
        role = m.get("role")
        if role == "assistant":
            b = _tag(content, "run_experiment")
            if b is not None:
                try:
                    pending = json.loads(b)
                except Exception:
                    pending = None
            f = _tag(content, "run_mse_fit")
            if f is not None:
                fits.append({"source": f.strip(), "params": {},
                             "loss": None, "error": "no_output_yet"})
        elif role == "user":
            o = _tag(content, "experiment_output")
            if o is not None and pending is not None:
                try:
                    outs = json.loads(o)
                    for i, out in zip(pending, outs):
                        if isinstance(i, dict) and isinstance(out, dict):
                            experiments.append((i, out))
                except Exception:
                    pass
                pending = None
            fo = _tag(content, "mse_fit_output")
            if fo is not None and fits:
                try:
                    r = json.loads(fo)
                    fits[-1].update(
                        params=r.get("fitted_params") or {},
                        loss=r.get("loss_after"),
                        error=r.get("error"),
                    )
                except Exception:
                    fits[-1]["error"] = "unparseable_output"
    return experiments, fits


# --------------------------------------------------------------------------
# surprise
# --------------------------------------------------------------------------

def _compile(source, params):
    from scienceagent.evaluator import (
        _compile_law, _wrap_with_timeout, clean_law_source,
    )
    fn = _wrap_with_timeout(_compile_law(clean_law_source(source)))
    return functools.partial(fn, **params) if params else fn


def _predict_errors(law, case, out):
    """Squared L2 residuals per (scored particle, time) for one experiment."""
    import numpy as np
    errs = []
    times = case.get("measurement_times") or out.get("measurement_times") or []
    if "pos2" in case:  # two-particle protocol
        obs = out.get("pos2")
        if obs is None:
            return errs
        for t, o in zip(times, obs):
            try:
                res = law(pos1=[0.0, 0.0], pos2=case["pos2"],
                          p1=case["p1"], p2=case["p2"],
                          velocity2=case["velocity2"], duration=float(t))
                # STRICT protocol convention only: (final_pos2, final_vel2),
                # each a 2-vector. Anything else counts as a failed
                # prediction — the panel and mse_fit both tell the agent to
                # fix its format; no lenient parsing here.
                p2, _v2 = res
                if not (hasattr(p2, "__len__") and len(p2) >= 2):
                    continue
                errs.append((float(p2[0]) - o[0]) ** 2 + (float(p2[1]) - o[1]) ** 2)
            except Exception:
                continue
        return errs

    obs_pos = out.get("positions")
    if obs_pos is None:
        return errs
    obs_pos = np.asarray(obs_pos, dtype=float)

    if "ring_radius" in case:  # circle protocol (11 particles)
        rr = float(case.get("ring_radius", 5.0))
        vt = float(case.get("initial_tangential_velocity", 0.0))
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        pos = np.vstack([[[0.0, 0.0]],
                         np.column_stack([rr * np.cos(ang), rr * np.sin(ang)])])
        vel = np.vstack([[[0.0, 0.0]],
                         np.column_stack([-vt * np.sin(ang), vt * np.cos(ang)])])
        kwargs = {"positions": pos.tolist(), "velocities": vel.tolist()}
        sl = slice(0, obs_pos.shape[1])
    elif "probe_positions" in case:
        bg = out.get("background_initial_positions")
        if bg is None:
            return errs
        bg = np.asarray(bg, dtype=float)
        probe_pos = np.asarray(case["probe_positions"], dtype=float)
        probe_vel = np.asarray(case["probe_velocities"], dtype=float)
        bg_vel = out.get("background_initial_velocities")
        if bg_vel is None:
            # three_species: background starts at rest (stated in the agent's
            # own world instructions). dark_matter: visible-particle initial
            # velocities are NOT agent-known — no fair prediction possible.
            if os.environ.get("BRIDGE_WORLD") == "three_species":
                bg_vel = [[0.0, 0.0]] * len(bg)
            else:
                return errs
        bg_vel = np.asarray(bg_vel, dtype=float)
        kwargs = {
            "positions": np.vstack([bg, probe_pos]).tolist(),
            "velocities": np.vstack([bg_vel, probe_vel]).tolist(),
        }
        pm = out.get("particle_masses")
        if pm is not None:
            kwargs["masses"] = [float(x) for x in pm]
        n_bg = len(bg)
        sl = slice(n_bg, n_bg + len(probe_pos))  # score probes only
    else:
        return errs

    for j, t in enumerate(times):
        if j >= obs_pos.shape[0]:
            break
        try:
            pred = np.asarray(law(duration=float(t), **kwargs), dtype=float)
            diff = pred[sl] - obs_pos[j, sl]
            errs.extend(np.sum(diff * diff, axis=-1).tolist())
        except Exception:
            continue
    return errs


def surprise_reading(experiments, fits, sigma):
    """Median/max residual (in sigma units) of the newest candidate vs all
    collected data. Returns None when not computable."""
    if not fits or not experiments or not sigma:
        return None
    usable = [f for f in fits if not f["error"]] or fits
    cand = usable[-1]
    try:
        law = _compile(cand["source"], cand["params"])
    except Exception:
        return None
    import numpy as np
    chi2 = []
    n_cases = 0
    for case, out in experiments:
        n_cases += 1
        for e in _predict_errors(law, case, out):
            chi2.append(e / (2.0 * sigma * sigma))
    if not chi2:
        # candidate exists but produced zero usable predictions — say so
        # loudly instead of a silent n/a (instrumentation must not hide
        # its own failures)
        return {
            "candidate_index": len(fits),
            "fitted": not cand["error"],
            "n_points": 0,
            "predict_failed": True,
            "n_cases_attempted": n_cases,
        }
    a = np.asarray(chi2)
    return {
        "candidate_index": len(fits),
        "fitted": not cand["error"],
        "n_points": int(a.size),
        "median_sigma": round(float(math.sqrt(np.median(a))), 2),
        "max_sigma": round(float(math.sqrt(np.max(a))), 2),
    }


# --------------------------------------------------------------------------
# envelope
# --------------------------------------------------------------------------

def _norm(v):
    return math.hypot(float(v[0]), float(v[1]))


def envelope_reading(experiments):
    if not experiments:
        return None
    radii, speeds, ratios, masses, tmax = [], [], [], [], []
    for case, _ in experiments:
        if "pos2" in case:
            radii.append(_norm(case["pos2"]))
            speeds.append(_norm(case["velocity2"]))
            try:
                ratios.append(float(case["p1"]) / float(case["p2"]))
            except Exception:
                pass
        elif "ring_radius" in case:
            radii.append(float(case["ring_radius"]))
            speeds.append(abs(float(case.get("initial_tangential_velocity", 0))))
        elif "probe_positions" in case:
            radii.extend(_norm(p) for p in case["probe_positions"])
            speeds.extend(_norm(v) for v in case["probe_velocities"])
            for mv in case.get("probe_masses", []):
                masses.append(float(mv))
        ts = case.get("measurement_times")
        if ts:
            tmax.append(max(ts))
    env = {"n_cases": len(experiments)}
    if radii:
        env["r"] = (round(min(radii), 3), round(max(radii), 3))
    if speeds:
        nz = [s for s in speeds if s > 1e-6]
        env["speed_zero_frac"] = round(1 - len(nz) / len(speeds), 2)
        env["speed_max"] = round(max(speeds), 3)
    if ratios:
        env["p1_over_p2"] = (round(min(ratios), 3), round(max(ratios), 3))
    if masses:
        env["mass"] = (round(min(masses), 3), round(max(masses), 3))
    if tmax:
        env["time_horizon_max"] = round(max(tmax), 3)
    return env


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------

def build_panel(messages, sigma):
    """Returns (panel_text or None, readings_dict for the ledger)."""
    experiments, fits = parse_history(messages)
    sur = surprise_reading(experiments, fits, sigma)
    env = envelope_reading(experiments)
    readings = {"n_experiments": len(experiments), "n_fits": len(fits),
                "surprise": sur, "envelope": env}
    if not sur and not env:
        return None, readings

    lines = [
        "<instrument_panel>",
        "Auto-computed from YOUR OWN experiments and candidate laws only — "
        "no ground-truth access. Observation noise sigma = "
        f"{sigma:g} (instrument precision).",
    ]
    if sur and sur.get("predict_failed"):
        lines.append(
            "- SURPRISE: could not evaluate your latest candidate on your own "
            f"data (every call across {sur['n_cases_attempted']} experiment "
            "cases failed). Check that it executes cleanly and returns "
            "(final_pos2, final_velocity2) — the same convention as "
            "<final_law>."
        )
    elif sur:
        fit_note = "" if sur["fitted"] else " (last fit errored; init params used)"
        lines.append(
            f"- SURPRISE: your latest candidate law{fit_note} vs all "
            f"{sur['n_points']} data points collected so far: median residual "
            f"= {sur['median_sigma']} sigma, max = {sur['max_sigma']} sigma. "
            "Reading guide: ~1 sigma means the theory explains your data down "
            "to the noise floor; >2 sigma is a systematic misfit — that is "
            "not noise."
        )
    elif fits:
        lines.append(
            "- SURPRISE: not computable for the latest candidate "
            "(law failed to run on your collected data)."
        )
    else:
        lines.append(
            "- SURPRISE: no candidate law submitted to <run_mse_fit> yet — "
            "no reading."
        )
    if env:
        parts = []
        if "r" in env:
            parts.append(f"initial radius {env['r'][0]}–{env['r'][1]}")
        if "speed_max" in env:
            parts.append(
                f"initial speed: {int(env['speed_zero_frac']*100)}% zero, "
                f"max {env['speed_max']}")
        if "p1_over_p2" in env:
            parts.append(
                f"p1/p2 ratio {env['p1_over_p2'][0]}–{env['p1_over_p2'][1]}")
        if "mass" in env:
            parts.append(f"probe mass {env['mass'][0]}–{env['mass'][1]}")
        if "time_horizon_max" in env:
            parts.append(f"time horizon <= {env['time_horizon_max']}")
        lines.append(
            f"- ENVELOPE explored so far ({env['n_cases']} experiment cases): "
            + "; ".join(parts) + ". This is NOT a limit — it is simply the "
            "region your experiments have covered. Your law is only tested "
            "INSIDE it; outside it, it is unconstrained by any of your data."
        )
    lines.append(
        "Before choosing your next action, state in one sentence what these "
        "readings imply for your plan — or why they change nothing."
    )
    lines.append("</instrument_panel>")
    return "\n".join(lines), readings


def gate_check(messages, reply_text, sigma):
    """Submission gate: if `reply_text` contains a <final_law>, score it
    against ALL data the agent collected this run. Returns None when no
    final_law is present or the gate cannot evaluate; otherwise a dict
    {action: 'pass'|'bounce', median_sigma, max_sigma, worst_case, worst_t}.

    Content-free by construction: it only points at the agent's OWN data
    points that the submitted law fails to reproduce.
    """
    law_src = _tag(reply_text, "final_law")
    if law_src is None or not sigma:
        return None
    experiments, _fits = parse_history(messages)
    if not experiments:
        return None
    try:
        law = _compile(law_src.strip(), {})
    except Exception:
        return None  # official validator handles uncompilable submissions
    import numpy as np
    chi2 = []
    worst = (0.0, None, None)  # (sigma, case, t)
    for case, out in experiments:
        times = case.get("measurement_times") or out.get("measurement_times") or []
        errs = _predict_errors(law, case, out)
        for e, t in zip(errs, times):
            s = math.sqrt(e / (2.0 * sigma * sigma))
            chi2.append(s * s)
            if s > worst[0]:
                worst = (s, case, t)
    if not chi2:
        return None
    a = np.asarray(chi2)
    med = float(math.sqrt(np.median(a)))
    mx = float(math.sqrt(np.max(a)))
    med_thr = float(os.environ.get("BRIDGE_GATE_MED", "2.0"))
    max_thr = float(os.environ.get("BRIDGE_GATE_MAX", "20.0"))
    action = "bounce" if (med > med_thr or mx > max_thr) else "pass"
    wc = dict(worst[1]) if isinstance(worst[1], dict) else None
    if wc:
        wc.pop("measurement_times", None)
    return {"action": action, "median_sigma": round(med, 2),
            "max_sigma": round(mx, 2), "worst_case": wc,
            "worst_t": worst[2]}


def gate_bounce_message(g):
    return (
        "<submission_gate>\n"
        "Your <final_law> has NOT been accepted yet. Scored against ALL the "
        "data you collected this run, it leaves unexplained residuals: "
        f"median {g['median_sigma']} sigma, max {g['max_sigma']} sigma "
        "(noise floor = 1 sigma). Worst offender: experiment "
        f"{json.dumps(g['worst_case'])} at t={g['worst_t']}, residual "
        f"{g['max_sigma']} sigma.\n"
        "A law that cannot reproduce your own observations is not yet a "
        "discovery. Choose one:\n"
        "(1) revise the law and submit again;\n"
        "(2) run further experiments to resolve the anomaly first;\n"
        "(3) if you are confident the residual is a measurement or "
        "integration artifact, resubmit the SAME <final_law> together with a "
        "one-paragraph justification — a resubmission after this notice will "
        "be accepted.\n"
        "</submission_gate>"
    )


def maybe_panel(messages):
    """Env-driven entry point for bridge_client. Never raises."""
    if not os.environ.get("BRIDGE_INSTRUMENT"):
        return None, None
    try:
        sigma = float(os.environ.get("BRIDGE_NOISE_STD") or 0) or None
        return build_panel(messages, sigma)
    except Exception as e:  # instrumentation must never kill a run
        return None, {"instrument_error": str(e)[:300]}
