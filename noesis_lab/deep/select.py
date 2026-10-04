"""Which papers to read in full: the ones closest to the queued hypotheses in the gap graph
(DEEP_READ §1). Pure code, deterministic, no LLM."""
from __future__ import annotations

import re
from collections.abc import Sequence

from .. import gaps
from ..schemas import Claim, DirectionRecord, QueueItem, RetrievedPaper
from ..search.coverage import is_derived

BASE = {"path": 1.0, "gate": 1.0, "method": 0.7, "mechanism": 0.5, "direction": 0.3}
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
SELECTION_RULE = (
    "For each of the top N queued hypotheses (by gap_score) and every contradiction gap, each paper gets\n"
    "proximity = Σ over its relations to the hypothesis of base × tier_weight × setting_similarity, with base:\n"
    "  1.0 a claim on the hypothesis's explanation path; 1.0 a claim the prior-art gate listed (re-runs only);\n"
    "  0.7 a claim on the same method node (any setting); 0.5 a claim on a mechanism node of the hypothesis's\n"
    "  methods; 0.3 found by a scout direction on the hypothesis's building blocks (paper tier weight).\n"
    "tier_weight: T1 1.0, T2 0.8, T3 0.6, T4 0. Top 3 papers per hypothesis, deduplicated, at most 20 in total,\n"
    "taken in hypothesis order. Ties: more citations, then arXiv id. Only arXiv papers can be read (arXiv HTML / ar5iv)."
)


def _targets(queue: Sequence[QueueItem], found: Sequence[gaps.Gap], top_n: int) -> list[dict]:
    out, seen = [], set()
    for q in list(queue)[:top_n]:
        out.append({"key": q.key, "delta": dict(q.delta), "gap_ids": list(q.gap_ids), "gap_score": q.gap_score,
                    "reason": "queued"})
        seen.add(q.key)
    for x in gaps.rank_gaps(found):
        if x.gap_type == "contradiction" and x.delta_key not in seen:
            out.append({"key": x.delta_key, "delta": dict(x.delta), "gap_ids": [x.gap_id], "gap_score": x.score,
                        "reason": "contradiction"})
            seen.add(x.delta_key)
    return out


def select_papers(queue: Sequence[QueueItem], found: Sequence[gaps.Gap], graph: gaps.GapGraph,
                  claims: dict[str, Claim], papers: dict[str, RetrievedPaper],
                  directions: Sequence[DirectionRecord] = (), *, top_n: int = 8, per_hypothesis: int = 3,
                  max_papers: int = 20, gate_claims: dict[str, list[str]] | None = None) -> dict:
    by_gap = {g.gap_id: g for g in found}
    lit = {cid: c for cid, c in claims.items() if not is_derived(c)}
    targets = _targets(queue, found, top_n)
    selected: list[str] = []
    rows = []
    for t in targets:
        methods = {f"{f}={v}" for f, v in t["delta"].items()}
        fields = set(t["delta"])
        rel: dict[str, list[dict]] = {}

        def add(pid: str, relation: str, weight: float, ref: str) -> None:
            if pid in papers and ARXIV_ID.match(pid) and weight > 0:
                rel.setdefault(pid, []).append({"relation": relation, "ref": ref, "weight": round(weight, 6)})

        def claim_weight(c: Claim, relation: str) -> float:
            return BASE[relation] * gaps.TIER_WEIGHT.get(c.tier, 0.0) * gaps.setting_similarity(c)

        path_ids = sorted({cid for gid in t["gap_ids"] if gid in by_gap for cid in by_gap[gid].claim_ids})
        for cid in path_ids:
            if cid in lit:
                add(lit[cid].paper_id, "path", claim_weight(lit[cid], "path"), cid)
        for cid in sorted((gate_claims or {}).get(t["key"], [])):
            if cid in lit:
                add(lit[cid].paper_id, "gate", claim_weight(lit[cid], "gate"), cid)
        mechs = {e.mechanism_id for e in graph.edges if e.method in methods and e.mechanism_id}
        for cid, c in sorted(lit.items()):
            m = gaps.method_key(c)
            if m in methods:
                add(c.paper_id, "method", claim_weight(c, "method"), cid)
            elif c.mechanism_category in mechs and c.mechanism:
                add(c.paper_id, "mechanism", claim_weight(c, "mechanism"), cid)
        for d in directions:
            if fields & set(d.building_blocks):
                for pid, p in sorted(papers.items()):
                    if d.direction_id in p.directions:
                        add(pid, "direction", BASE["direction"] * gaps.TIER_WEIGHT.get(p.tier, 0.0), d.direction_id)
        ranked = sorted(rel, key=lambda pid: (-sum(r["weight"] for r in rel[pid]), -papers[pid].cited_by, pid))
        picks = []
        for pid in ranked[:per_hypothesis]:
            prox = round(sum(r["weight"] for r in rel[pid]), 6)
            take = pid in selected or len(selected) < max_papers
            if take and pid not in selected:
                selected.append(pid)
            picks.append({"paper_id": pid, "title": papers[pid].title, "proximity": prox,
                          "relations": rel[pid], "read": take})
        rows.append({**t, "papers": picks})
    return {"rule": SELECTION_RULE, "top_n": top_n, "per_hypothesis": per_hypothesis, "max_papers": max_papers,
            "targets": rows, "selected": selected}
