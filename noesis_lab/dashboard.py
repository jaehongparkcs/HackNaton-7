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


# ----------------------------------------------------------------------------- timeline
STALL_SECONDS = 300          # a silent stretch longer than this is flagged (a sleeping Mac, a stalled call)


def _parse(ts: str):
    import datetime as dt
    t = dt.datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def timeline_rows(events: list[dict], decisions: list[dict], runs: list[dict]) -> list[dict[str, Any]]:
    """Everything with a wall-clock time, in time order, with the silent gap before each item.
    Events and decisions carry `ts` (bundles recorded before FINAL_FIXES A6 do not); runs carry
    `started_at`."""
    items = [(e["ts"], "LLM call", f"{e['event_id']} · {e.get('role', '')}") for e in events if e.get("ts")]
    items += [(d["ts"], "decision", f"{d['decision_id']} · {d['kind']}") for d in decisions if d.get("ts")]
    items += [(r["started_at"], "run", f"{r['run_id']} · seed {r['seed']} · {r.get('train_seconds', 0):.0f} s")
              for r in runs if r.get("started_at")]
    rows, prev = [], None
    for ts, kind, what in sorted(items, key=lambda x: (_parse(x[0]), x[1], x[2])):
        t = _parse(ts)
        gap = (t - prev).total_seconds() if prev else 0.0
        rows.append({"time (UTC)": t.strftime("%H:%M:%S"), "kind": kind, "what": what,
                     "gap before (s)": round(gap, 1), "stall": "⚠ stall" if gap > STALL_SECONDS else ""})
        prev = t
    return rows


# ----------------------------------------------------------------------------- deep read
def deep_selection_rows(selection: dict) -> list[dict[str, Any]]:
    """Why each paper was read: hypothesis → paper, proximity and its relations."""
    rows = []
    for t in selection.get("targets", []):
        for p in t["papers"]:
            rows.append({"hypothesis": t["key"], "why queued": t["reason"], "paper": p["paper_id"],
                         "title": p["title"][:70], "proximity": p["proximity"],
                         "relations": ", ".join(f"{r['relation']} ({r['ref']}, {r['weight']:.2f})" for r in p["relations"]),
                         "read": "yes" if p["read"] else "no (cap)"})
    return rows


def _v(x: Any) -> str:
    return "not in queue" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def deep_update_rows(decisions: list[dict]) -> list[dict[str, Any]]:
    """Every outcome the full text changed (deep_read_update decisions), with the quote it rests on."""
    rows = []
    for d in decisions:
        if d["kind"] != "deep_read_update":
            continue
        w = d["what"]
        if w == "coverage":
            row = {"what": "coverage", "subject": f"{d['claim_id']} (arXiv:{d['paper_id']})",
                   "change": f"{d['before']} → {d['after']}",
                   "quote": f"{d['section']}: “{d['quote']}” ({d['parameter_count']} parameters)"}
        elif w == "recipe":
            src = d.get("sources", {})
            quote = next((v["quote"] for v in src.values()), "")
            row = {"what": "recipe", "subject": d["method"],
                   "change": f"lr × {d['lr_mult']:g}, weight decay × {d['wd_mult']:g}",
                   "quote": f"arXiv:{d['paper_id']} ({d.get('version') or '?'}): “{quote}”"}
        elif w == "stated_gap":
            row = {"what": "stated gap", "subject": d["candidate_key"], "change": f"new gap, score {d['gap_score']:.3f}"
                   + ("" if d.get("concrete") else " (no concrete condition: × 0.5)"),
                   "quote": f"arXiv:{d['paper_id']}: “{d['limitation']}”"}
        else:
            passages = d.get("passages") or {}
            row = {"what": w.replace("_", " "), "subject": d["candidate_key"],
                   "change": f"{_v(d['before'])} → {_v(d['after'])}",
                   "quote": "; ".join(f"{c}: “{q}”" for c, q in passages.items())
                   or ("full-text findings: " + ", ".join(d.get("findings", [])) if d.get("findings") else "")}
        rows.append({"decision": d["decision_id"], **row})
    return rows


def revision_lines(rev: dict) -> list[str]:
    """r0 (abstract only) → r1 (with the full text) for one hypothesis, as plain lines."""
    r0, r1 = rev.get("r0") or {}, rev.get("r1") or {}
    out = []
    for k, label in (("gap_score", "gap score"), ("status", "literature status"),
                     ("predicted_direction", "predicted direction"), ("coverage", "coverage in our setting")):
        if r0.get(k) != r1.get(k):
            out.append(f"{label}: {_v(r0.get(k))} → {_v(r1.get(k))}")
    new = sorted(set(r1.get("claim_ids", [])) - set(r0.get("claim_ids", [])))
    if new:
        out.append("new claims on the path: " + ", ".join(new))
    return out
