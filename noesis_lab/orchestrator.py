"""Orchestrator: plain Python, one golden path (BUILD_PLAN s7).

LLMs propose and interpret. Everything that becomes a number, a label or a decision is computed
here by `stats.py` from stored runs, with a rule that was written before the first run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import stats
from .agents import Critic, Scientist
from .bundle import fresh_dir, seal
from .config import baseline_config, get_profile, load_config, path_of
from .literature import LiteratureAgent, Snapshot
from .llm import LLM, BudgetExceeded, Mode
from .mock_llm import make_mock
from .runner import LiveRunner, RecordedRunner, RunBudgetExceeded
from .schemas import Analysis, ExperimentConfig, Fixture, PriorArtVerdictKind
from .store import Store


@dataclass
class SessionOpts:
    session: str
    profile: str = "full"
    llm_mode: Mode = "live"
    force: bool = False
    # replay: serve LLM + runs from results/<session>, write to work/replay-<session>
    # otherwise: write a new bundle to results/<session>


@dataclass
class Outcome:
    fixture_id: str
    hypothesis_id: str
    status: str
    analysis: Analysis | None = None


class Session:
    def __init__(self, o: SessionOpts):
        self.o, self.cfg = o, load_config()
        self.profile = get_profile(o.profile)
        self.replay = o.llm_mode == "replay"
        root = Path(__file__).resolve().parents[1]
        self.src = root / self.cfg["paths"]["results"] / o.session
        self.out = (root / self.cfg["paths"]["work"] / f"replay-{o.session}") if self.replay else self.src
        fresh_dir(self.out, force=self.replay or o.force)
        self.store = Store(self.out / "notebook.sqlite")
        self.snap = Snapshot(path_of("papers"), path_of("claims"))
        fx = json.loads(path_of("fixtures").read_text())["fixtures"]
        self.fixtures = {f["fixture_id"]: Fixture(**f) for f in fx}
        self.t0 = time.monotonic()
        self.budget = self.cfg["budget"]

        if self.replay:
            self.llm = LLM("replay", self.cfg["llm"], recordings_path=self.src / "recordings/llm.jsonl",
                           store=self.store)
            self.runner = RecordedRunner(self.store, self.profile, self.src / "runs.jsonl")
        else:
            self.llm = LLM(o.llm_mode, self.cfg["llm"], recordings_path=self.out / "recordings/llm.jsonl",
                           store=self.store, mock_fn=make_mock(self.fixtures) if o.llm_mode == "mock" else None,
                           max_cost_usd=self.budget["max_llm_cost_usd"])
            self.runner = LiveRunner(self.store, self.profile, path_of("data"),
                                     self.cfg["paths"]["data_sha256"], self.out / "runs.jsonl",
                                     self.budget["max_runs"])
        self.lit = LiteratureAgent(self.llm, self.snap)
        self.scientist, self.critic = Scientist(self.llm, self.snap), Critic(self.llm)
        self.baseline = baseline_config(self.profile)
        self.seeds = self.cfg["seeds"]
        self.rule = self.cfg["rule"]
        self.noise: stats.NoiseFloor | None = None
        self.base_by_seed: dict = {}
        self.runs_avoided = 0

    # ------------------------------------------------------------------ helpers
    def _check_wall(self) -> None:
        if (time.monotonic() - self.t0) / 60 > self.budget["max_wall_minutes"]:
            raise BudgetExceeded("wall-clock budget exhausted")

    def _set_status(self, hid: str, fx: Fixture, status: str, **payload) -> None:
        old = {h["hypothesis_id"]: h for h in self.store.hypotheses()}.get(hid, {})
        merged = {k: v for k, v in old.items() if k not in ("hypothesis_id", "status", "fixture_id")}
        merged.update(payload)
        self.store.put_hypothesis(hid, fx.fixture_id, status, merged)

    # ------------------------------------------------------------------ step 0
    def noise_floor(self) -> None:
        runs = [self.runner.get(self.baseline, s) for s in self.seeds["baseline"]]
        failed = [r for r in runs if r.status != "ok"]
        if failed:
            raise RuntimeError(f"baseline run failed, cannot continue: {failed[0].error}")
        self.noise = stats.noise_floor(runs)
        self.base_by_seed = {r.seed: r for r in runs}
        self.store.put_noise("noise_baseline", self.noise)
        did = self.store.add_decision("noise_floor", "", {
            "baseline_config_hash": self.baseline.config_hash(), "profile": self.profile.name,
            "mean": self.noise.mean, "sd": self.noise.sd, "n": self.noise.n})
        for r in runs:
            self.store.link("decision", did, "measured_by", "run", r.run_id)

    # ------------------------------------------------------------------ one hypothesis
    def evaluate(self, fx: Fixture, seeds: list[int] | None = None) -> Outcome:
        self._check_wall()
        hid = f"hyp_{fx.fixture_id}"
        self._set_status(hid, fx, "proposed", statement=fx.statement, title=fx.title,
                         kind=fx.kind, fixture=True)
        seeds = seeds or self.seeds["paired"]

        # 1. Prior-art gate (declared LLM channel: verdict shown with the exact passage)
        pa = self.lit.check(fx)
        self.store.link("hypothesis", hid, "gated_by", "event", pa.event_id)
        if pa.claim_id:
            self.store.link("hypothesis", hid, "nearest_claim", "claim", pa.claim_id)
            self.store.link("claim", pa.claim_id, "from_paper", "paper", pa.paper_id)
        if pa.verdict == PriorArtVerdictKind.known:
            self.runs_avoided += len(seeds)
            self._set_status(hid, fx, "rejected_prior_art", prior_art=pa.model_dump(mode="json"))
            did = self.store.add_decision("prior_art_rejection", hid, {
                "runs_avoided": len(seeds), "passage": pa.passage, "paper_id": pa.paper_id,
                "label": pa.label, "claim_id": pa.claim_id})
            self.store.link("decision", did, "rests_on", "claim", pa.claim_id)
            return Outcome(fx.fixture_id, hid, "rejected_prior_art")

        # 2. Scientist -> typed config; Critic reviews before compute
        context = [f"[{c.claim_id}] {c.claim}" for c, _ in self.snap.search(fx.statement, 3)]
        prop, cfg, delta, sci_eid = self.scientist.propose(fx, self.baseline, context)
        self.store.link("hypothesis", hid, "proposed_by", "event", sci_eid)
        self._set_status(hid, fx, "proposed", prior_art=pa.model_dump(mode="json"),
                         proposal=prop.model_dump(mode="json"), config_delta=delta,
                         config_hash=cfg.config_hash(), candidate_config=cfg.model_dump())
        review, crit_eid = self.critic.review(fx, prop, self.baseline, cfg, delta)
        self.store.link("hypothesis", hid, "reviewed_by", "event", crit_eid)
        self._set_status(hid, fx, "queued", critic_pre=review.model_dump(mode="json"))
        if review.verdict == "reject":
            self._set_status(hid, fx, "rejected_critic")
            did = self.store.add_decision("critic_rejection", hid, {
                "reasons": review.reasons, "confounds": review.confounds})
            self.store.link("decision", did, "rests_on", "event", crit_eid)
            return Outcome(fx.fixture_id, hid, "rejected_critic")

        # 3. Real paired runs
        self._set_status(hid, fx, "screening")
        cand = [self.runner.get(cfg, s) for s in seeds]
        for r in cand:
            self.store.link("hypothesis", hid, "candidate_run", "run", r.run_id)
        failed = [r for r in cand if r.status != "ok"]
        if failed:
            self._set_status(hid, fx, "run_failed", errors=[r.error for r in failed])
            self.store.add_decision("run_failure", hid, {"errors": [r.error for r in failed]})
            return Outcome(fx.fixture_id, hid, "run_failed")

        # 4. Deterministic statistics, labels, derived evidence
        a = self._analyze(hid, fx, cand, seeds)
        return Outcome(fx.fixture_id, hid, f"screened_{a.branch}", a)

    def _analyze(self, hid: str, fx: Fixture, cand_runs, seeds) -> Analysis:
        assert self.noise is not None
        base = [self.base_by_seed[s] for s in seeds]
        a = stats.analyze(hid, cand_runs, base, self.noise,
                          all_base_runs=list(self.base_by_seed.values()),
                          promising_sd=self.rule["promising_sd"], harmful_sd=self.rule["harmful_sd"])
        self.store.put_analysis(a)
        self.store.link("analysis", a.analysis_id, "for", "hypothesis", hid)
        for p in a.pairs:
            self.store.link("analysis", a.analysis_id, "uses", "run", p.candidate_run_id)
            self.store.link("analysis", a.analysis_id, "uses", "run", p.baseline_run_id)
        pa = self._hyp(hid).get("prior_art", {})
        cited = fx.cited_claim_ids or ([pa["claim_id"]] if pa.get("claim_id") else [])
        for cid in cited:
            ev = stats.relate_to_claim(a, self.snap.claims[cid])
            self.store.put_evidence(ev)
            self.store.link("evidence", ev.evidence_id, "derived_from", "analysis", a.analysis_id)
            self.store.link("evidence", ev.evidence_id, "about", "claim", cid)

        branch_text = f"branch={a.branch}"
        reading, source, eid = self.critic.interpret(a, branch_text)
        status = {"promising": "screened_promising", "no_improvement": "screened_no_improvement",
                  "harmful": "screened_harmful"}[a.branch]
        self._set_status(hid, fx, status, latest_analysis_id=a.analysis_id)
        did = self.store.add_decision("screening_result", hid, {
            "analysis_id": a.analysis_id, "branch": a.branch, "label": a.label,
            "reading": reading, "reading_source": source, "reading_event": eid})
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        self.store.link("decision", did, "interpreted_in", "event", eid)
        return a

    # ------------------------------------------------------------------ pre-registered rule
    def apply_rule(self, first: Outcome, considered: list[str]) -> None:
        a = first.analysis
        assert a is not None
        na = stats.next_action(a, extra_seeds=self.seeds["extra"],
                               candidate_order=self.cfg["candidate_order"], already_considered=considered)
        did = self.store.add_decision("next_action", first.hypothesis_id, {
            "analysis_id": a.analysis_id, "next_action": na.model_dump(mode="json"),
            "inputs": {"extra_seeds": self.seeds["extra"], "candidate_order": self.cfg["candidate_order"],
                       "already_considered": list(considered)}})
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        fx = self.fixtures[first.fixture_id]

        if na.action == "extra_seeds":
            seeds = list(self.seeds["paired"]) + list(self.seeds["extra"])
            cfg = self._candidate_cfg(first.hypothesis_id)
            cand = [self.runner.get(cfg, s) for s in seeds]
            for r in cand:
                self.store.link("hypothesis", first.hypothesis_id, "candidate_run", "run", r.run_id)
            if any(r.status != "ok" for r in cand):
                self.store.add_decision("run_failure", first.hypothesis_id,
                                        {"errors": [r.error for r in cand if r.status != "ok"]})
                return
            a2 = self._analyze(first.hypothesis_id, fx, cand, seeds)
            d2 = self.store.add_decision("extra_seeds_result", first.hypothesis_id, {
                "analysis_id": a2.analysis_id, "triggered_by": did, "branch": a2.branch,
                "note": "Still a screening result. Confirmation tests are out of scope today."})
            self.store.link("decision", d2, "triggered_by", "decision", did)
            self.store.link("decision", d2, "based_on", "analysis", a2.analysis_id)
        elif na.action == "literature_check":
            q = (f"Does the snapshot already report that this change makes validation loss worse or "
                 f"gives no benefit in small-scale language models? Change: {fx.statement}")
            pa = self.lit.check(fx, question=q, role="literature_targeted")
            d2 = self.store.add_decision("literature_check", first.hypothesis_id, {
                "triggered_by": did, "verdict": pa.verdict.value, "label": pa.label,
                "claim_id": pa.claim_id, "passage": pa.passage, "rationale": pa.rationale,
                "analysis_id": a.analysis_id, "contradiction_recorded": True})
            self.store.link("decision", d2, "triggered_by", "decision", did)
            self.store.link("decision", d2, "based_on", "event", pa.event_id)
            if pa.claim_id:
                self.store.link("decision", d2, "rests_on", "claim", pa.claim_id)
        else:
            self._follow_candidates(na.detail.get("fixture_id"), did, considered)

    def _hyp(self, hid: str) -> dict:
        return {x["hypothesis_id"]: x for x in self.store.hypotheses()}[hid]

    def _candidate_cfg(self, hid: str) -> ExperimentConfig:
        return ExperimentConfig.model_validate(self._hyp(hid)["candidate_config"])

    def _follow_candidates(self, fid: str | None, trigger_did: str, considered: list[str]) -> None:
        limit = self.budget["max_followup_candidates"]
        tried = 0
        while fid and tried < limit:
            fx = self.fixtures[fid]
            considered.append(fid)
            tried += 1
            res = self.evaluate(fx)
            d2 = self.store.add_decision("followup_candidate", res.hypothesis_id, {
                "triggered_by": trigger_did, "fixture_id": fid, "status": res.status,
                "analysis_id": res.analysis.analysis_id if res.analysis else None})
            self.store.link("decision", d2, "triggered_by", "decision", trigger_did)
            if res.analysis is not None:
                self.store.link("decision", d2, "based_on", "analysis", res.analysis.analysis_id)
                return
            # gated out before compute: take the next candidate in the fixed order
            fid = next((f for f in self.cfg["candidate_order"] if f not in considered), None)

    # ------------------------------------------------------------------ whole session
    def run(self) -> dict:
        status = "complete"
        try:
            for p in self.snap.papers.values():
                self.store.put_paper(p)
            for c in self.snap.claims.values():
                self.store.put_claim(c)
            self.store.set_meta("rule_text", stats.RULE_TEXT)
            self.store.set_meta("rule_thresholds", self.rule)
            self.store.set_meta("snapshot", {"papers": self.snap.size, "claims": len(self.snap.claims),
                                             "dimensions": self.snap.dimensions(),
                                             "fetched": self.snap.fetched, "source": self.snap.source})
            self.store.set_meta("objective", "Improve validation loss of a tiny character-level "
                                "Transformer under a short fixed wall-clock training budget.")
            self.store.set_meta("profile", self.profile.model_dump())
            self.store.set_meta("baseline_config", self.baseline.model_dump())
            self.noise_floor()
            order = [f for f in self.fixtures.values() if f.kind != "followup_candidate"]
            outcomes = [self.evaluate(f) for f in order]
            considered = [o.fixture_id for o in outcomes]
            first = next((o for o in outcomes
                          if o.analysis is not None and self.fixtures[o.fixture_id].kind == "untested_setting"), None)
            if first is None:    # primary fixtures were all gated out: start from the fixed order
                for fid in self.cfg["candidate_order"]:
                    res = self.evaluate(self.fixtures[fid])
                    considered.append(fid)
                    if res.analysis is not None:
                        first = res
                        break
            if first is not None:
                self.apply_rule(first, considered)
            self.store.set_meta("runs_avoided", self.runs_avoided)
        except (BudgetExceeded, RunBudgetExceeded) as e:
            status = f"budget_exhausted: {e}"
            self.store.add_decision("budget_exhausted", "", {"reason": str(e)})
        meta = {"session": self.o.session, "profile": self.profile.name, "llm_mode": self.o.llm_mode,
                "status": status, "llm_cost_usd": round(self.llm.cost_usd, 4),
                "llm_calls": self.llm.n_calls, "runs_executed": self.runner.n_executed,
                "runs_avoided": self.runs_avoided, "model": self.cfg["llm"]["model"]}
        if self.replay:
            self.store.set_meta("replay_of", self.o.session)
            digest = self.store.state_digest()
            self.store.close()
            return {**meta, "state_digest": digest, "out": str(self.out)}
        self.store.set_meta("session", meta)
        state = seal(self.out, self.store, meta=meta)
        return {**state, "out": str(self.out)}


def run_session(o: SessionOpts) -> dict:
    return Session(o).run()
