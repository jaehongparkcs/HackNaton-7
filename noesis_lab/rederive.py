"""`make rederive` (R1): recompute every statistic from the shipped runs and demand an exact match.

Uses the same pure functions as the live session. Reads only the bundle (no LLM, no training).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import gaps, stats
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
    rule_version = st.get_meta("rule_version", 1)        # bundles recorded before v2 are v1, never re-judged
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
                              promising_sd=rule["promising_sd"], harmful_sd=rule["harmful_sd"],
                              rule_version=rule_version)
        same(f"analysis {a.analysis_id}", stored, fresh.model_dump(mode="json"))

    analyses = {x["analysis_id"]: Analysis(**x) for x in st.analyses()}
    stored_gaps = st.get_meta("gaps")
    if stored_gaps is not None:             # every gap score is recomputed from the frozen claims
        from .literature import Snapshot
        frozen = sorted(Snapshot.from_corpus(bundle / "corpus").claims.values(), key=lambda c: c.claim_id)
        graph, found = gaps.find_gaps(frozen)
        same("gaps", stored_gaps, [g.model_dump() for g in found])
    cycles = {0: (graph, found, frozen)} if stored_gaps is not None else {}
    for cyc in st.get_meta("gap_cycles", []):   # each cycle: frozen claims + our own results written back
        cl = [*frozen, *(Claim(**c) for c in cyc["derived"])]
        g_c, f_c = gaps.find_gaps(cl)
        same(f"gaps cycle {cyc['cycle']}", cyc["gaps"], [g.model_dump() for g in f_c])
        cycles[cyc["cycle"]] = (g_c, f_c, cl)
    for d in st.decisions():
        if d["kind"] == "next_action":
            inp = d["inputs"]
            queue = [QueueItem(**q) for q in inp["queue"]]
            fresh = stats.next_action(analyses[d["analysis_id"]], extra_seeds=inp["extra_seeds"],
                                      queue=queue, rule_version=rule_version)
            same(f"next_action {d['decision_id']}", d["next_action"], fresh.model_dump(mode="json"))
            if inp.get("mode") == "gaps":       # hypothesis engine: gaps and queue from the frozen corpus
                graph, found, frozen = cycles[inp.get("cycle", 0)]
                fresh_q = gaps.gap_queue(found, graph, frozen, inp["tested"], soft_rejected=inp["soft_rejected"],
                                         held=inp["held"], overrides=inp["overrides"], top_n=inp["top_n"])
            else:
                fresh_q = stats.candidate_queue(
                    baseline, claims.values(), inp["tested"], soft_rejected=inp.get("soft_rejected", []),
                    held=inp.get("held", []), overrides=inp.get("overrides", []))
            same(f"queue {d['decision_id']}", inp["queue"], [q.model_dump(mode="json") for q in fresh_q])
    for d in st.decisions():                # explore / confirm / promote: every number and every pick
        if d["kind"] == "exploration_result":
            fresh = stats.single_seed_estimate(runs[d["candidate_run_id"]], runs[d["baseline_run_id"]], d["noise_sd"])
            same(f"exploration {d['decision_id']}", {k: d[k] for k in fresh}, fresh)
        elif d["kind"] == "finalists_selected":
            same(f"finalists {d['decision_id']}", d["finalists"],
                 stats.select_finalists(d["explored"], d["top_n"], d["min_n"]))
        elif d["kind"] == "confirmation":
            a = analyses[d["analysis_id"]]
            same(f"confirmation {d['decision_id']}", [d["paired_sd"], d["shrinkage_sd"]],
                 [a.improvement_in_noise_sd, stats.shrinkage(d["single_seed_sd"], a.improvement_in_noise_sd)])
        elif d["kind"] == "promotion":
            same(f"promotion {d['decision_id']}", True,
                 stats.should_promote(analyses[d["analysis_id"]], len(d["run_ids"])))
    for e in st.derived_evidence():
        fresh = stats.relate_to_claim(analyses[e["analysis_id"]], claims[e["claim_id"]])
        same(f"evidence {e['evidence_id']}", e, fresh.model_dump(mode="json"))
    st.close()
    if n_checked == 0:
        problems.append("nothing was checked (empty bundle?)")
    print(f"rederive: {n_checked} derived objects recomputed from {len(runs)} stored runs")
    return problems
