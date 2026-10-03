"""Orchestrator: plain Python, one golden path (BUILD_PLAN s7).

LLMs propose and interpret. Everything that becomes a number, a label or a decision is computed
here by `stats.py` from stored runs, with a rule that was written before the first run.
"""
from __future__ import annotations

import json
import re
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
from .schemas import Analysis, ExperimentConfig, Fixture, PriorArtVerdictKind, QueueItem
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
    decision_id: str = ""             # the screening_result decision that the rule acts on


def generate_fixture(item: QueueItem, baseline: ExperimentConfig, snap: Snapshot) -> Fixture:
    """A hypothesis seed written by code from a queue item (templated, no LLM). The Scientist then
    writes the statement, mechanism and falsification rule inside the item's one field."""
    top = min((snap.claims[c] for c in item.supporting_claim_ids),
              key=lambda c: (stats.OUTCOME_RANK[c.expected_outcome], c.claim_id))
    base_val = getattr(baseline, item.field)
    return Fixture(
        fixture_id="gen_" + re.sub(r"\W+", "_", f"{item.field}_{item.value}"), kind="generated",
        origin="generated", candidate_key=item.key,
        title=f"{item.field} = {item.value} (generated from the literature queue)",
        statement=(f"Setting {item.field} to {item.value} instead of {base_val} ({top.method}) lowers "
                   "the validation loss of a tiny character-level Transformer language model "
                   "trained for a fixed 2,000-step budget."),
        allowed_fields=[item.field], cited_claim_ids=list(item.supporting_claim_ids),
        queue_priority=list(item.priority),
        mock={"same_comparison": True, "claim_id": top.claim_id,
              "rationale": "(mock) the supporting claim tests this change.",
              "config_changes": [{"field": item.field, "value": item.value}]})


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
        self.tested: set[str] = set()        # "field=value" keys tested or gated out this session
        self.followups = 0                   # queue candidates taken so far (bounded by the budget)
        self.extra_done: set[str] = set()    # hypotheses that already got extra seeds (never twice)
        self.exhausted = False
        self.noise_did = ""
        self.trigger = ""                    # decision that led to the hypothesis being evaluated now

    # ------------------------------------------------------------------ helpers
    def _check_wall(self) -> None:
        if (time.monotonic() - self.t0) / 60 > self.budget["max_wall_minutes"]:
            raise BudgetExceeded("wall-clock budget exhausted")

    def _chained(self, payload: dict) -> dict:
        """Every decision names the decision that led to it (`triggered_by`)."""
        return {**payload, "triggered_by": self.trigger} if self.trigger else payload

    def _link_trigger(self, did: str) -> None:
        if self.trigger:
            self.store.link("decision", did, "triggered_by", "decision", self.trigger)

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
        did = self.noise_did = self.store.add_decision("noise_floor", "", {
            "baseline_config_hash": self.baseline.config_hash(), "profile": self.profile.name,
            "mean": self.noise.mean, "sd": self.noise.sd, "n": self.noise.n})
        for r in runs:
            self.store.link("decision", did, "measured_by", "run", r.run_id)

    # ------------------------------------------------------------------ one hypothesis
    def evaluate(self, fx: Fixture, seeds: list[int] | None = None) -> Outcome:
        self._check_wall()
        hid = f"hyp_{fx.fixture_id}"
        self._set_status(hid, fx, "proposed", statement=fx.statement, title=fx.title, kind=fx.kind,
                         fixture=fx.origin == "fixture", origin=fx.origin.upper(),
                         supporting_claim_ids=list(fx.cited_claim_ids),
                         queue_priority=fx.queue_priority)
        if fx.candidate_key:
            self.tested.add(fx.candidate_key)
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
            did = self.store.add_decision("prior_art_rejection", hid, self._chained({
                "runs_avoided": len(seeds), "passage": pa.passage, "paper_id": pa.paper_id,
                "label": pa.label, "claim_id": pa.claim_id}))
            self.store.link("decision", did, "rests_on", "claim", pa.claim_id)
            self._link_trigger(did)
            return Outcome(fx.fixture_id, hid, "rejected_prior_art")

        # 2. Scientist -> typed config; Critic reviews before compute
        context = [f"[{c.claim_id}] {c.claim}" for c, _ in self.snap.search(fx.statement, 3)]
        prop, cfg, delta, sci_eid = self.scientist.propose(fx, self.baseline, context)
        self.store.link("hypothesis", hid, "proposed_by", "event", sci_eid)
        if len(delta) == 1:
            self.tested.add("{}={}".format(*next(iter(delta.items()))))
        self._set_status(hid, fx, "proposed", prior_art=pa.model_dump(mode="json"),
                         proposal=prop.model_dump(mode="json"), config_delta=delta,
                         config_hash=cfg.config_hash(), candidate_config=cfg.model_dump())
        review, crit_eid = self.critic.review(fx, prop, self.baseline, cfg, delta)
        self.store.link("hypothesis", hid, "reviewed_by", "event", crit_eid)
        self._set_status(hid, fx, "queued", critic_pre=review.model_dump(mode="json"))
        if review.verdict == "reject":
            self._set_status(hid, fx, "rejected_critic")
            did = self.store.add_decision("critic_rejection", hid, self._chained({
                "reasons": review.reasons, "confounds": review.confounds}))
            self.store.link("decision", did, "rests_on", "event", crit_eid)
            self._link_trigger(did)
            return Outcome(fx.fixture_id, hid, "rejected_critic")

        # 3. Real paired runs
        self._set_status(hid, fx, "screening")
        cand = [self.runner.get(cfg, s) for s in seeds]
        for r in cand:
            self.store.link("hypothesis", hid, "candidate_run", "run", r.run_id)
        failed = [r for r in cand if r.status != "ok"]
        if failed:
            self._set_status(hid, fx, "run_failed", errors=[r.error for r in failed])
            did = self.store.add_decision("run_failure", hid,
                                          self._chained({"errors": [r.error for r in failed]}))
            self._link_trigger(did)
            return Outcome(fx.fixture_id, hid, "run_failed")

        # 4. Deterministic statistics, labels, derived evidence
        a, did = self._analyze(hid, fx, cand, seeds, triggered_by=self.trigger)
        return Outcome(fx.fixture_id, hid, f"screened_{a.branch}", a, did)

    def _analyze(self, hid: str, fx: Fixture, cand_runs, seeds, *, kind: str = "screening_result",
                 triggered_by: str | None = None) -> tuple[Analysis, str]:
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
        payload = {"analysis_id": a.analysis_id, "branch": a.branch, "label": a.label,
                   "reading": reading, "reading_source": source, "reading_event": eid}
        if kind == "extra_seeds_result":
            payload["note"] = "Still a screening result. Confirmation tests are out of scope today."
        if triggered_by:
            payload["triggered_by"] = triggered_by
        did = self.store.add_decision(kind, hid, payload)
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        self.store.link("decision", did, "interpreted_in", "event", eid)
        if triggered_by:
            self.store.link("decision", did, "triggered_by", "decision", triggered_by)
        return a, did

    # ------------------------------------------------------------------ candidate queue
    def _queue(self) -> list[QueueItem]:
        return stats.candidate_queue(self.baseline, self.snap.claims.values(), self.tested)

    def _generate(self, item: QueueItem) -> Fixture:
        fx = generate_fixture(item, self.baseline, self.snap)
        self.fixtures[fx.fixture_id] = fx      # registered so the mock LLM can resolve it
        return fx

    # ------------------------------------------------------------------ pre-registered rule
    def _stop(self, bound: str, reason: str, triggered_by: str) -> None:
        self.exhausted = True
        self.store.add_decision("budget_exhausted", "", {"bound": bound, "reason": reason,
                                                         "triggered_by": triggered_by})

    def _decide(self, cur: Outcome) -> str:
        """Apply the pre-registered rule to one screening result and execute the action."""
        a = cur.analysis
        assert a is not None
        queue = self._queue()
        na = stats.next_action(a, extra_seeds=self.seeds["extra"], queue=queue)
        did = self.store.add_decision("next_action", cur.hypothesis_id, {
            "analysis_id": a.analysis_id, "next_action": na.model_dump(mode="json"),
            "triggered_by": cur.decision_id,
            "inputs": {"extra_seeds": self.seeds["extra"], "tested": sorted(self.tested),
                       "queue": [q.model_dump(mode="json") for q in queue]}})
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        if cur.decision_id:
            self.store.link("decision", did, "triggered_by", "decision", cur.decision_id)
        fx = self.fixtures[cur.fixture_id]

        if na.action == "extra_seeds":
            if cur.hypothesis_id in self.extra_done:
                return did
            if self.runner.n_executed + len(self.seeds["extra"]) > self.budget["max_runs"]:
                self._stop("max_runs", f"run budget {self.budget['max_runs']} would be exceeded by "
                           "the extra seeds", did)
                return did
            self.extra_done.add(cur.hypothesis_id)
            seeds = list(self.seeds["paired"]) + list(self.seeds["extra"])
            cfg = self._candidate_cfg(cur.hypothesis_id)
            cand = [self.runner.get(cfg, s) for s in seeds]
            for r in cand:
                self.store.link("hypothesis", cur.hypothesis_id, "candidate_run", "run", r.run_id)
            if any(r.status != "ok" for r in cand):
                self.store.add_decision("run_failure", cur.hypothesis_id,
                                        {"errors": [r.error for r in cand if r.status != "ok"]})
                return did
            self._analyze(cur.hypothesis_id, fx, cand, seeds, kind="extra_seeds_result", triggered_by=did)
        elif na.action == "literature_check":
            q = (f"Does the snapshot already report that this change makes validation loss worse or "
                 f"gives no benefit in small-scale language models? Change: {fx.statement}")
            pa = self.lit.check(fx, question=q, role="literature_targeted")
            d2 = self.store.add_decision("literature_check", cur.hypothesis_id, {
                "triggered_by": did, "verdict": pa.verdict.value, "label": pa.label,
                "claim_id": pa.claim_id, "passage": pa.passage, "rationale": pa.rationale,
                "covers_our_setting": pa.covers_our_setting, "coverage_note": pa.coverage_note,
                "analysis_id": a.analysis_id, "contradiction_recorded": True})
            self.store.link("decision", d2, "triggered_by", "decision", did)
            self.store.link("decision", d2, "based_on", "event", pa.event_id)
            if pa.claim_id:
                self.store.link("decision", d2, "rests_on", "claim", pa.claim_id)
        return did            # next_candidate: the chain below takes the head of the queue

    def _hyp(self, hid: str) -> dict:
        return {x["hypothesis_id"]: x for x in self.store.hypotheses()}[hid]

    def _candidate_cfg(self, hid: str) -> ExperimentConfig:
        return ExperimentConfig.model_validate(self._hyp(hid)["candidate_config"])

    def _chain(self, pending: list[Outcome]) -> None:
        """Every screening result gets the rule; after any action the lab takes the next candidate
        from the generated queue, until a bound (followups / runs) or the queue ends."""
        limit, did = self.budget["max_followup_candidates"], ""
        while not self.exhausted:
            while pending and not self.exhausted:
                did = self._decide(pending.pop(0))
            queue = self._queue()
            if self.exhausted or not queue:
                break
            if self.followups >= limit:
                self._stop("max_followup_candidates", f"{limit} follow-up candidates taken; "
                           f"{len(queue)} left in the queue", did)
                break
            if self.runner.n_executed + len(self.seeds["paired"]) > self.budget["max_runs"]:
                self._stop("max_runs", f"run budget {self.budget['max_runs']} would be exceeded by "
                           "the next candidate", did)
                break
            item = queue[0]
            fx = self._generate(item)
            self.followups += 1
            self.trigger = did
            res = self.evaluate(fx)
            d2 = self.store.add_decision("followup_candidate", res.hypothesis_id, {
                "triggered_by": did, "fixture_id": fx.fixture_id, "candidate_key": item.key,
                "supporting_claim_ids": item.supporting_claim_ids, "priority": item.priority,
                "status": res.status,
                "analysis_id": res.analysis.analysis_id if res.analysis else None})
            if did:
                self.store.link("decision", d2, "triggered_by", "decision", did)
            if res.analysis is not None:
                self.store.link("decision", d2, "based_on", "analysis", res.analysis.analysis_id)
                pending.append(res)

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
                                "Transformer under a fixed 2,000-step training budget.")
            self.store.set_meta("profile", self.profile.model_dump())
            self.store.set_meta("baseline_config", self.baseline.model_dump())
            self.noise_floor()
            self.store.set_meta("candidate_queue", [q.model_dump(mode="json") for q in
                                                    stats.candidate_queue(self.baseline,
                                                                          self.snap.claims.values(), [])])
            self.trigger = self.noise_did
            outcomes = [self.evaluate(f) for f in list(self.fixtures.values())]
            self._chain([o for o in outcomes if o.analysis is not None])
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
