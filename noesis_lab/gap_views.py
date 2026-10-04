"""Dashboard views of the hypothesis engine, as Graphviz DOT strings and table rows. Pure
formatting over stored gaps, graph and directions: nothing here scores, ranks or decides."""
from __future__ import annotations

import re
import textwrap
from typing import Any

BADGE = {"coverage": "transfer", "abc": "ABC", "link": "combination", "contradiction": "resolution",
         "bridge": "bridge"}
STATUS_STYLE = {      # direction gap status → (fill, border style)
    "covered": ("#d9d9d6", "solid"), "partial": ("#f6d58a", "solid"), "contested": ("#c9a7e8", "solid"),
    "open": ("#a8dba8", "solid"), "unexplored": ("#ffffff", "solid")}
SIGN_COLOR = {"+": "#2e9e4f", "-": "#d64545", "0": "#8a8a85"}
BRANCH_COLOR = {"promising": "#2e9e4f", "harmful": "#d64545", "no_improvement": "#8a8a85"}
_PALETTE = ["#dbe9fb", "#fde3d3", "#e2f3e0", "#efe1f7", "#fdf3c7", "#d8f1f1", "#f7dce6", "#e8e8e8"]


def _q(text: str, width: int = 30, lines: int = 4) -> str:
    wrapped = textwrap.wrap(str(text), width=width) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1].rstrip(". ") + "…"
    return "\\n".join(w.replace("\\", "\\\\").replace('"', "'") for w in wrapped)


def _id(name: str) -> str:
    return "n_" + re.sub(r"\W", "_", name)


def gap_rows(gaps: list[dict]) -> list[dict[str, Any]]:
    """The ranked gap list with its score components (structural hints, not probabilities)."""
    return [{"rank": i + 1, "type": BADGE.get(g["gap_type"], g["gap_type"]), "label": g["novelty_type"],
             "config change": g["delta_key"].replace("+", " + "), "predicts": g.get("predicted_direction", "") or "—",
             "gap score": g["score"],
             "plausibility": g["plausibility"], "1 − coverage": round(1 - g["coverage"], 4),
             "testability": g["testability"], "evidence quality": g["evidence_quality"],
             "claims on path": len(g["claim_ids"]), "gap_id": g["gap_id"]} for i, g in enumerate(gaps)]


DIRECTION_WORD = {"+": "lower validation loss (+)", "-": "higher validation loss (−)", "0": "no worse (0)",
                  "": "none (no directional prediction for this gap)"}


def gap_type_summary(gaps: list[dict], graph: dict) -> list[dict[str, Any]]:
    """One row per gap family with its count and, when empty, a true statement about this corpus
    instead of an empty panel."""
    count = {t: sum(1 for g in gaps if g["gap_type"] == t) for t in BADGE}
    shared = graph.get("shared_mechanisms", {})
    n_mech = len({e["mechanism_id"] for e in graph.get("edges", []) if e["mechanism_id"]})
    rows = [
        ("Transfer (coverage gap)", count["coverage"],
         "No structural support found: every runnable method with claims is already covered in our setting, or has no claims."),
        ("Transfer by mechanism (ABC closure)", count["abc"],
         f"No structural support found: no runnable method is linked, by a claim reporting a benefit, to a mechanism "
         f"that another claim ties to better quality ({n_mech} mechanism nodes in this corpus)."),
        ("Combination (link prediction + bridges)", count["link"] + count["bridge"],
         f"No structural support found: in this corpus no two combinable methods share a mechanism or were tested in "
         f"the same paper ({len(shared)} mechanism node(s) are linked to 2 or more runnable methods)."),
        ("Resolution (contradiction)", count["contradiction"],
         "No structural support found: no runnable method has claims with opposite signs in one setting bucket."),
    ]
    return [{"family": name, "gaps": n, "note": "" if n else note} for name, n, note in rows]


def gap_card_dot(gap: dict, claims: dict[str, dict]) -> str:
    """The explanation path of one gap, with the verbatim quote on every edge."""
    out = ["digraph G {", "  rankdir=LR; nodesep=0.3;", '  node [fontname="Helvetica", fontsize=10, shape=box, style="filled", fillcolor="#e8f0fb"];',
           '  edge [fontname="Helvetica", fontsize=8, color="#8a8a85"];']
    nodes: dict[str, str] = {}
    for p in gap["path"]:
        for end in (p["from"], p["to"]):
            if end not in nodes:
                nodes[end] = _id(end)
                fill = "#f4efe6" if end.startswith("mechanism") else ("#e2f3e0" if end.startswith("quality") else "#e8f0fb")
                out.append(f'  {nodes[end]} [label="{_q(end, 26)}", fillcolor="{fill}"];')
        c = claims.get(p["claim_id"], {})
        quote = f"[{p['claim_id']} · {c.get('tier', '?')}] “{c.get('source_span', '')}”"
        out.append(f'  {nodes[p["from"]]} -> {nodes[p["to"]]} [label="{_q(quote, 34, 5)}"];')
    target = _id("gap_target")
    out.append(f'  {target} [label="{_q("GAP: " + gap["delta_key"].replace("+", " + ") + " in our setting — not found", 26)}", '
               'style="dashed,filled", fillcolor="#fff4e5"];')
    for m in gap["delta_key"].split("+"):
        if m in nodes:
            out.append(f'  {nodes[m]} -> {target} [style=dashed, color="#eb6834", label="score {gap["score"]:.2f}"];')
    out.append("}")
    return "\n".join(out)


def overview_dot(graph: dict, gaps: list[dict], results: list[dict], max_gaps: int = 12) -> str:
    """Methods and mechanisms as nodes (communities colored); claim edges signed (green +, red −,
    grey 0); our measured results as thick edges; the top gaps as dashed edges with their score."""
    comm = graph.get("communities", {})
    out = ["graph G {", "  layout=neato; overlap=false; splines=true;",
           '  node [fontname="Helvetica", fontsize=10, style="filled"];', '  edge [fontname="Helvetica", fontsize=8];',
           '  quality [label="model quality\\n(validation loss)", shape=doubleoctagon, fillcolor="#ffffff"];']
    for m in graph["methods"]:
        out.append(f'  {_id(m)} [label="{m}", shape=box, fillcolor="{_PALETTE[comm.get("M:" + m, 0) % len(_PALETTE)]}"];')
    used = {e["mechanism_id"] for e in graph["edges"] if e["mechanism_id"]}
    for k, mech in graph["mechanisms"].items():
        if k in used:
            out.append(f'  {_id(k)} [label="{_q(mech["label"], 22, 3)}", shape=ellipse, '
                       f'fillcolor="{_PALETTE[comm.get("K:" + k, 0) % len(_PALETTE)]}"];')
    signed: dict[tuple[str, str], int] = {}
    mech_edges: dict[tuple[str, str], int] = {}
    mech_out: dict[tuple[str, str], int] = {}
    for e in graph["edges"]:
        if e["method"] and e["sign"]:
            signed[(e["method"], e["sign"])] = signed.get((e["method"], e["sign"]), 0) + 1
        if e["method"] and e["mechanism_id"]:
            mech_edges[(e["method"], e["mechanism_id"])] = mech_edges.get((e["method"], e["mechanism_id"]), 0) + 1
        if e["mechanism_id"] and e["sign"]:
            mech_out[(e["mechanism_id"], e["sign"])] = mech_out.get((e["mechanism_id"], e["sign"]), 0) + 1
    for (m, sign), n in sorted(signed.items()):
        out.append(f'  {_id(m)} -- quality [color="{SIGN_COLOR[sign]}", label="{sign} ×{n}"];')
    for (m, k), n in sorted(mech_edges.items()):
        out.append(f'  {_id(m)} -- {_id(k)} [color="#8a8a85", label="×{n}"];')
    for (k, sign), n in sorted(mech_out.items()):
        out.append(f'  {_id(k)} -- quality [color="{SIGN_COLOR[sign]}", style=dotted, label="{sign} ×{n}"];')
    for r in results:                                  # our own measurements: thick edges
        for m in r["delta_key"].split("+"):
            out.append(f'  {_id(m)} -- quality [penwidth=4, color="{BRANCH_COLOR.get(r["branch"], "#8a8a85")}", '
                       f'label="ours: {r["branch"]} ({r["sd"]:+.2f}× SD)'
                       + (f' · {r["outcome"].upper()}' if r.get("outcome") in ("hit", "miss") else "") + '"];')
    seen = set()
    for g in gaps:
        if g["delta_key"] in seen or g["score"] <= 0 or len(seen) >= max_gaps:
            continue
        seen.add(g["delta_key"])
        ms = g["delta_key"].split("+")
        a, b = (_id(ms[0]), _id(ms[1])) if len(ms) == 2 else (_id(ms[0]), "quality")
        out.append(f'  {a} -- {b} [style=dashed, color="#eb6834", fontcolor="#eb6834", label="gap {g["score"]:.2f}"];')
    out.append("}")
    return "\n".join(out)


def direction_map_dot(directions: list[dict]) -> str:
    """Directions as large nodes colored by gap status, each connected to its papers; a paper found
    by two directions connects them."""
    out = ["graph G {", "  layout=neato; overlap=false;", '  node [fontname="Helvetica", fontsize=10];']
    for d in directions:
        fill, _ = STATUS_STYLE.get(d["status"], ("#ffffff", "solid"))
        style = "dashed,filled" if d["outside_testbed"] else "filled"
        extra = " · outside testbed" if d["outside_testbed"] else ""
        out.append(f'  {d["direction_id"]} [label="{_q(d["title"], 20, 3)}\\n[{d["status"]}{extra}]\\n'
                   f'{d["n_papers"]} papers · {d["n_claims"]} claims", shape=box, style="{style}", fillcolor="{fill}", '
                   'penwidth=2];')
    papers = sorted({p for d in directions for p in d["paper_ids"]})
    for p in papers:
        out.append(f'  {_id(p)} [label="", shape=circle, width=0.12, style=filled, fillcolor="#8a8a85", tooltip="arXiv:{p}"];')
    for d in directions:
        for p in d["paper_ids"]:
            out.append(f'  {d["direction_id"]} -- {_id(p)} [color="#c8c8c4"];')
    out.append("}")
    return "\n".join(out)


def direction_view_dot(direction: dict, claims: dict[str, dict], papers: dict[str, dict],
                       hyps: list[dict], analyses: dict[str, dict], max_claims: int = 12) -> str:
    """One direction's subgraph: papers (tier, date, review status) → claims (verbatim quote) →
    building blocks → our hypotheses (novelty type) → our results."""
    out = ["digraph G {", "  rankdir=LR; nodesep=0.2;", '  node [fontname="Helvetica", fontsize=9, shape=box, style="filled"];',
           '  edge [color="#8a8a85"];',
           f'  dir [label="{_q(direction["title"], 22)}\\n[{direction["status"]}]", fillcolor="#fff4e5", penwidth=2];']
    blocks = set(direction["building_blocks"])
    shown = direction["claim_ids"][:max_claims]
    for pid in sorted({claims[c]["paper_id"] for c in shown} | set(direction["paper_ids"][:max_claims])):
        p = papers.get(pid, {})
        review = {True: "peer-reviewed", False: "preprint"}.get(p.get("peer_reviewed"), "review status unknown")
        review = "curated" if p.get("source") == "curated" else review
        out.append(f'  {_id(pid)} [label="{_q(p.get("title", pid), 26, 3)}\\n{p.get("tier", "?")} · {p.get("published", "")} · {review}", '
                   'fillcolor="#e8f0fb"];')
        out.append(f"  dir -> {_id(pid)};")
    for cid in shown:
        c = claims[cid]
        out.append(f'  {_id(cid)} [label="{_q(f"[{cid}] “" + c["source_span"] + "”", 34, 5)}", fillcolor="#f4f4f1"];')
        out.append(f"  {_id(c['paper_id'])} -> {_id(cid)};")
        ch = c.get("config_change")
        if ch:
            (f, v), = ch.items()
            if f in blocks:
                out.append(f'  {_id("blk_" + f)} [label="building block: {f}", fillcolor="#e2f3e0"];')
                out.append(f"  {_id(cid)} -> {_id('blk_' + f)};")
    for h in hyps:
        fields = set((h.get("config_delta") or {}).keys()) or {k.split("=")[0] for k in (h.get("candidate_key") or "").split("+") if k}
        if not fields & blocks:
            continue
        hid = _id(h["hypothesis_id"])
        badge = h.get("novelty_type") or h.get("origin", "")
        out.append(f'  {hid} [label="{_q(h.get("candidate_key") or h.get("title", ""), 24)}\\n[{badge}] {h["status"]}", '
                   'style="dashed,filled", fillcolor="#f4efe6"];')
        for f in sorted(fields & blocks):
            out.append(f'  {_id("blk_" + f)} [label="building block: {f}", fillcolor="#e2f3e0"];')
            out.append(f"  {_id('blk_' + f)} -> {hid};")
        a = analyses.get(h.get("latest_analysis_id", ""))
        if a:
            out.append(f'  {_id(a["analysis_id"])} [label="our result: {a["branch"]}\\n{a["improvement_in_noise_sd"]:+.2f}× noise SD", '
                       f'fillcolor="{BRANCH_COLOR.get(a["branch"], "#8a8a85")}", fontcolor="white"];')
            out.append(f"  {hid} -> {_id(a['analysis_id'])};")
    out.append("}")
    return "\n".join(out)


def our_results(hyps: list[dict], analyses: dict[str, dict], baseline: dict | None = None) -> list[dict]:
    """Measured results as (delta key, branch, SDs, prediction outcome) for the overview map's
    thick edges and the gap card. The analysis is the one the rule used: vs the incumbent the
    change was tested on (named in `incumbent`), never the cumulative vs-original headline."""
    from .dashboard import _fmt, incumbent_delta
    from .stats import delta_key, prediction_outcome
    out = []
    for h in hyps:
        a = analyses.get(h.get("latest_analysis_id", ""))
        if a and h.get("config_delta"):
            pred = (h.get("gap") or {}).get("predicted_direction", "") if h.get("kind") == "gap" else ""
            out.append({"delta_key": delta_key(h["config_delta"]), "branch": a["branch"],
                        "sd": a["improvement_in_noise_sd"], "seeds": len(a["pairs"]),
                        "outcome": prediction_outcome(pred, a["branch"]) if h.get("kind") == "gap" else "",
                        "incumbent": _fmt(incumbent_delta(h, baseline))})
    return out


def exploration_rows(decisions: list[dict]) -> list[dict[str, Any]]:
    """Single-seed exploration results. Labeled EXPLORATION — NOT A RESULT."""
    finalists = {(d["cycle"], k) for d in decisions if d["kind"] == "finalists_selected" for k in d["finalists"]}
    return [{"cycle": d["cycle"], "candidate": d["key"], "archive cell": d["cell"], "seed": d["seed"],
             "single-seed × noise SD": d["improvement_in_noise_sd"], "finalist": "yes" if (d["cycle"], d["key"]) in finalists else "",
             "label": d["label"]} for d in decisions if d["kind"] == "exploration_result"]


def shrinkage_rows(decisions: list[dict]) -> list[dict[str, Any]]:
    """Per finalist: the single-seed estimate, the paired estimate and their difference."""
    promoted = {d["candidate_key"] for d in decisions if d["kind"] == "promotion"}
    return [{"cycle": d["cycle"], "candidate": d["candidate_key"], "single-seed × SD": d["single_seed_sd"],
             "paired × SD": d["paired_sd"], "shrinkage (single − paired)": d["shrinkage_sd"],
             "paired branch": d["branch"], "promoted": "yes" if d["candidate_key"] in promoted else ""}
            for d in decisions if d["kind"] == "confirmation"]


OUTCOMES = ("hit", "miss", "null", "no_prediction")


def prediction_rows(hyps: list[dict], analyses: dict[str, dict], baseline: dict | None = None) -> list[dict[str, Any]]:
    """One row per screened gap hypothesis: what its gap predicted and what was measured (against
    the config it was tested on: the baseline, or an incumbent)."""
    from .dashboard import _fmt, incumbent_delta
    from .stats import prediction_outcome
    rows = []
    for h in hyps:
        gap, a = h.get("gap"), analyses.get(h.get("latest_analysis_id", ""))
        if h.get("kind") != "gap" or not gap or not a:
            continue
        pred = gap.get("predicted_direction", "")
        rows.append({"hypothesis": h.get("candidate_key", ""), "gap type": BADGE.get(gap["gap_type"], gap["gap_type"]),
                     "label": h.get("novelty_type", ""), "predicted": pred or "—",
                     "measured vs": ("incumbent " + inc) if (inc := _fmt(incumbent_delta(h, baseline))) else "baseline",
                     "measured": a["branch"],
                     "× noise SD": a["improvement_in_noise_sd"], "paired seeds": len(a["pairs"]),
                     "outcome": prediction_outcome(pred, a["branch"])})
    return rows


def prediction_counts(rows: list[dict], by: str) -> list[dict[str, Any]]:
    """Hit / miss / null counts grouped by `by` ("gap type" or "label"). Counts, not percentages:
    the numbers are small."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        row = out.setdefault(r[by], {by: r[by], **{o: 0 for o in OUTCOMES}})
        row[r["outcome"]] += 1
    return [out[k] for k in sorted(out)]
