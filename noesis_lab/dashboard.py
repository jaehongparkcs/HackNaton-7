"""Presentation helpers for the dashboard: pure functions over stored payloads (no statistics).

Kept out of app.py so the wording rules (which session opens first, how a sealed bundle's
limitation is phrased, which change gets credit for a gain) are testable without Streamlit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import ExperimentConfig

# ----------------------------------------------------------------------------- sessions
SESSION_FIRST = ("explore2", "golden")       # the two exhibits: hypothesis engine, core loop
SESSION_NOTES = {
    "explore2": "hypothesis engine (exhibit)",
    "golden": "core loop (exhibit)",
    "explore1": "superseded (scout schema bug, rule v1)",
    "mps-mock": "pipeline checks: not results",
    "smoke": "pipeline checks: not results",
    "pytest-smoke": "pipeline checks: not results",
    "rehearse": "rehearsal profile: not results",
}


def session_name(key: str) -> str:
    """'results/explore2/notebook.sqlite' -> 'explore2'; 'work/replay-golden/…' -> 'replay-golden'."""
    return Path(key).parent.name


def _base(name: str) -> str:
    return name.removeprefix("replay-")


def order_notebooks(keys: list[str]) -> list[str]:
    """explore2, then golden, then everything else alphabetically; a bundle before its replay."""
    def rank(k: str) -> tuple:
        name = session_name(k)
        base = _base(name)
        first = SESSION_FIRST.index(base) if base in SESSION_FIRST else len(SESSION_FIRST)
        return (first, base, name.startswith("replay-"), k)
    return sorted(keys, key=rank)


def notebook_label(key: str) -> str:
    name = session_name(key)
    note = SESSION_NOTES.get(_base(name))
    if not note and _base(name).startswith(("smoke", "rehearse")):
        note = "pipeline checks: not results"
    tag = " · replay" if name.startswith("replay-") else ""
    return f"{_base(name)}{tag}" + (f" — {note}" if note else "") + f"   ({key})"


# ----------------------------------------------------------------------------- corpus limitations
def degraded_note(reason: str, sealed: bool) -> str:
    """A search limitation. Before recording it is a to-do; in a sealed bundle it is a fact about
    how the bundle was recorded, so the instruction to fix it is dropped."""
    if not sealed:
        return f"Search degraded: {reason}"
    text = reason.replace(" Fix before recording.", "").replace("; fix before recording", "").rstrip()
    return f"Recorded with this limitation: {text}"


# ----------------------------------------------------------------------------- credit on an incumbent
def _fmt(delta: dict[str, Any]) -> str:
    return " + ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in sorted(delta.items()))


def incumbent_delta(h: dict, baseline: dict | None) -> dict[str, Any]:
    """Fields the hypothesis's config differs from the original baseline in, minus its own change:
    the incumbent it was tested on ({} when it was tested on the baseline itself)."""
    if not baseline or not h.get("candidate_config"):
        return {}
    full = ExperimentConfig(**baseline).diff(ExperimentConfig(**h["candidate_config"]))
    own = set(h.get("config_delta") or {})
    return {k: v for k, v in full.items() if k not in own}


def credit(h: dict, analyses: dict[str, dict], baseline: dict | None) -> dict[str, Any] | None:
    """For a hypothesis tested on top of an incumbent: its own effect (vs incumbent, the analysis
    the rule used) and the cumulative effect (vs the original baseline, which INCLUDES the
    incumbent's gain). None when there is no incumbent or no headline analysis."""
    own = analyses.get(h.get("latest_analysis_id", ""))
    cum = analyses.get(h.get("headline_analysis_id", ""))
    inc = incumbent_delta(h, baseline)
    if not (own and cum and inc):
        return None
    change, on = _fmt(h.get("config_delta") or {}), _fmt(inc)
    branch = own["branch"].replace("_", " ")
    text = (f"{change} on top of {on}: {branch} vs incumbent ({own['improvement_in_noise_sd']:+.2f} SD). "
            f"vs original {cum['improvement_in_noise_sd']:+.2f} SD, which includes {on}.")
    return {"incumbent": on, "own": own, "cumulative": cum, "text": text}


def analysis_role(h: dict, analysis_id: str, baseline: dict | None) -> str:
    """How to title one of a hypothesis's analyses. Empty for a hypothesis tested on the baseline."""
    inc = incumbent_delta(h, baseline)
    if not inc:
        return ""
    if analysis_id == h.get("headline_analysis_id"):
        return (f"vs original baseline (cumulative: includes the incumbent's gain, {_fmt(inc)}; "
                "informational, not this change's effect)")
    return f"vs incumbent {_fmt(inc)} (this change's own effect; the rule used this)"


# ----------------------------------------------------------------------------- Lion
LION_CAVEAT = ("Exploration only; our hyperparameter scaling, not a test of Lion. Every Lion run scored "
               "~1.64–1.70 vs ~1.53 baseline (single seed, never confirmed); the likely cause is our lr ÷ 5 "
               "scaling under a 2,000-step cosine budget, not Lion itself.")


def is_lion(h: dict) -> bool:
    delta = h.get("config_delta") or {}
    return delta.get("optimizer") == "lion" or "optimizer=lion" in (h.get("candidate_key") or "")
