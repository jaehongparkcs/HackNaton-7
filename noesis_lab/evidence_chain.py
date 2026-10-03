"""Evidence chain for the dashboard: paper → claim → verdict → hypothesis → config → runs →
analysis → decision → next decision, as a Graphviz DOT string.

Pure formatting over rows already stored in the notebook (no statistics here). Solid nodes were
produced by code (or curated by a human); dashed nodes carry LLM output.
"""
from __future__ import annotations

import textwrap
from typing import Any


def _q(text: str, width: int = 34, lines: int = 4) -> str:
    """Escape and wrap text for a DOT label."""
    wrapped = textwrap.wrap(str(text), width=width) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1].rstrip(". ") + "…"
    return "\\n".join(w.replace("\\", "\\\\").replace('"', "'") for w in wrapped)


def _node(nid: str, label: str, *, llm: bool = False, fill: str | None = None) -> str:
    style = "dashed,filled" if llm else "solid,filled"
    color = fill or ("#f4efe6" if llm else "#e8f0fb")
    return f'  {nid} [label="{label}", shape=box, style="{style}", fillcolor="{color}", fontsize=10];'


def chain_dot(data: dict[str, Any], hypothesis_id: str) -> str:
    """`data` is the dashboard's loaded notebook (hyps, claims, analyses, decisions, evidence)."""
    hyp = next(h for h in data["hyps"] if h["hypothesis_id"] == hypothesis_id)
    decisions = data["decisions"]
    out = ["digraph G {", "  rankdir=LR; nodesep=0.25; ranksep=0.45;",
           '  node [fontname="Helvetica"]; edge [color="#8a8a85"];']
    edges: list[tuple[str, str]] = []

    pa = hyp.get("prior_art") or {}
    prev = None
    if pa.get("claim_id"):
        claim = data["claims"].get(pa["claim_id"], {})
        out.append(_node("paper", _q(f"arXiv:{pa.get('paper_id')}  {pa.get('paper_title') or ''}")))
        cov = {True: "covers our setting: YES", False: "covers our setting: NO"}.get(
            claim.get("covers_our_setting"), "coverage: n/a")
        out.append(_node("claim", _q(f"{pa['claim_id']}  {cov} (human-curated)\n“{pa.get('passage') or ''}”", 36, 6)))
        edges += [("paper", "claim")]
        prev = "claim"
    if pa:
        cmp = "same comparison" if pa.get("same_comparison") else "no same comparison"
        out.append(_node("verdict", _q(f"verdict: {pa['verdict']}\n(LLM judged: {cmp}; code derived the verdict)"),
                         llm=True))
        if prev:
            edges.append((prev, "verdict"))
        prev = "verdict"
    out.append(_node("hyp", _q(f"[{hyp.get('origin', 'FIXTURE')}] {hyp.get('statement', '')}", 36, 5), llm=True))
    if prev:
        edges.append((prev, "hyp"))
    prev = "hyp"

    if hyp["status"] == "rejected_prior_art":
        d = next((x for x in decisions if x["kind"] == "prior_art_rejection"
                  and x["hypothesis_id"] == hypothesis_id), None)
        if d:
            out.append(_node("dec0", _q(f"decision {d['decision_id']}: rejected as prior art"), fill="#fde8e8"))
            out.append(_node("avoided", f"{d['runs_avoided']} runs avoided", fill="#e4f4e4"))
            edges += [(prev, "dec0"), ("dec0", "avoided")]
    else:
        if hyp.get("config_delta"):
            out.append(_node("cfg", _q(f"config delta {hyp['config_delta']}\nhash {hyp.get('config_hash')}")))
            edges.append((prev, "cfg"))
            prev = "cfg"
        for a in [a for a in data["analyses"] if a["hypothesis_id"] == hypothesis_id]:
            aid = a["analysis_id"]
            out.append(_node(f"runs_{aid}", f"{len(a['pairs'])} paired seeds ({2 * len(a['pairs'])} runs)"))
            out.append(_node(aid, _q(f"analysis {aid}\nbranch: {a['branch']}\n"
                                      f"{a['improvement_in_noise_sd']:+.2f}× noise SD")))
            edges += [(prev, f"runs_{aid}"), (f"runs_{aid}", aid)]
            last = aid
            for d in [x for x in decisions if x.get("analysis_id") == aid
                      and x["kind"] in ("screening_result", "extra_seeds_result", "next_action")]:
                did = d["decision_id"]
                title = d["kind"].replace("_", " ")
                if d["kind"] == "next_action":
                    title += f": {d['next_action']['action'].replace('_', ' ')}"
                out.append(_node(did, _q(f"{did} {title}"), fill="#e6f4ea"))
                edges.append((aid if d["kind"] != "next_action" else last, did))
                if d["kind"] != "next_action":
                    last = did
                else:
                    for f in [x for x in decisions if x["kind"] == "followup_candidate"
                              and x.get("triggered_by") == did]:
                        fid = f["decision_id"]
                        out.append(_node(fid, _q(f"next: {f['candidate_key']} ({f['status']})"), fill="#e6f4ea"))
                        edges.append((did, fid))
                    for f in [x for x in decisions if x["kind"] in ("literature_check", "extra_seeds_result")
                              and x.get("triggered_by") == did]:
                        fid = f["decision_id"]
                        out.append(_node(fid, _q(f"{fid} {f['kind'].replace('_', ' ')}"), fill="#e6f4ea",
                                         llm=f["kind"] == "literature_check"))
                        edges.append((did, fid))
    out += [f"  {a} -> {b};" for a, b in edges]
    out.append("}")
    return "\n".join(out)


def evidence_grid(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per candidate: paper claim, coverage, verdict, measured result and relation."""
    rows = []
    for h in data["hyps"]:
        pa = h.get("prior_art") or {}
        ana = next((a for a in reversed(data["analyses"]) if a["hypothesis_id"] == h["hypothesis_id"]), None)
        ev = [e for e in data["evidence"] if ana and e["analysis_id"] == ana["analysis_id"]]
        rows.append({
            "origin": h.get("origin", "FIXTURE"),
            "candidate": h.get("title", h["fixture_id"]),
            "paper claim": pa.get("claim_id") or "—",
            "covers our setting (human-curated)": {True: "yes", False: "no"}.get(pa.get("covers_our_setting"), "—"),
            "verdict": pa.get("verdict", "—"),
            "measured": f"{ana['branch']} ({ana['improvement_in_noise_sd']:+.2f}× SD)" if ana else h["status"],
            "relation": ", ".join(f"{e['claim_id']}: {e['relation']}" for e in ev) or "—",
        })
    return rows
