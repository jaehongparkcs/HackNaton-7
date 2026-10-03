"""`make rederive` (R1): recompute every statistic from the shipped runs and demand an exact match.

Uses the same pure functions as the live session. Reads only the bundle (no LLM, no training).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import stats
from .bundle import verify_manifest
from .schemas import Analysis, Claim, ExperimentConfig, QueueItem, RunResult, canonical_json
from .store import Store


def rederive(bundle: Path) -> list[str]:
    """Returns a list of mismatches (empty = every number reproduced exactly)."""
    problems = [f"manifest: {p}" for p in verify_manifest(bundle)]
    runs = {}
    for line in (bundle / "runs.jsonl").read_text().splitlines():
        if line.strip():
            r = RunResult(**json.loads(line))
            runs[r.run_id] = r
    st = Store(bundle / "notebook.sqlite", readonly=True)
    rule = st.get_meta("rule_thresholds")
    claims = {c["claim_id"]: Claim(**c) for c in st.claims()}
    baseline = ExperimentConfig(**st.get_meta("baseline_config"))
    n_checked = 0

    def same(label: str, stored, fresh) -> None:
        nonlocal n_checked
        n_checked += 1
        if canonical_json(stored) != canonical_json(fresh):
            problems.append(f"{label}: stored != recomputed")

    for nid, stored in st.noise_floors().items():
        fresh = stats.noise_floor([runs[i] for i in stored["run_ids"]])
        same(f"noise {nid}", stored, fresh.model_dump(mode="json"))

    for stored in st.analyses():
        a = Analysis(**stored)
        cand = [runs[p.candidate_run_id] for p in a.pairs]
        base = [runs[p.baseline_run_id] for p in a.pairs]
        noise = stats.noise_floor([runs[i] for i in a.noise.run_ids])
        fresh = stats.analyze(a.hypothesis_id, cand, base, noise,
                              all_base_runs=[runs[i] for i in a.noise.run_ids],
                              promising_sd=rule["promising_sd"], harmful_sd=rule["harmful_sd"])
        same(f"analysis {a.analysis_id}", stored, fresh.model_dump(mode="json"))

    analyses = {x["analysis_id"]: Analysis(**x) for x in st.analyses()}
    for d in st.decisions():
        if d["kind"] == "next_action":
            inp = d["inputs"]
            queue = [QueueItem(**q) for q in inp["queue"]]
            fresh = stats.next_action(analyses[d["analysis_id"]], extra_seeds=inp["extra_seeds"],
                                      queue=queue)
            same(f"next_action {d['decision_id']}", d["next_action"], fresh.model_dump(mode="json"))
            fresh_q = stats.candidate_queue(
                baseline, claims.values(), inp["tested"], soft_rejected=inp.get("soft_rejected", []),
                held=inp.get("held", []), overrides=inp.get("overrides", []))
            same(f"queue {d['decision_id']}", inp["queue"], [q.model_dump(mode="json") for q in fresh_q])
    for e in st.derived_evidence():
        fresh = stats.relate_to_claim(analyses[e["analysis_id"]], claims[e["claim_id"]])
        same(f"evidence {e['evidence_id']}", e, fresh.model_dump(mode="json"))
    st.close()
    if n_checked == 0:
        problems.append("nothing was checked (empty bundle?)")
    print(f"rederive: {n_checked} derived objects recomputed from {len(runs)} stored runs")
    return problems
