"""Orchestrator: plain Python, one golden path (BUILD_PLAN s7).

LLMs propose and interpret. Everything that becomes a number, a label or a decision is computed
here by `stats.py` from stored runs, with a rule that was written before the first run.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import gaps, stats
from .agents import Critic, GapScientist, Scientist
from .bundle import fresh_dir, seal
from .config import baseline_config, bundle_config, get_profile, load_config, path_of, protocol
from .literature import LiteratureAgent, Snapshot
from .llm import LLM, BudgetExceeded, Mode
from .mock_llm import make_mock
from .runner import LiveRunner, RecordedRunner, RunBudgetExceeded
from .schemas import Analysis, ExperimentConfig, Fixture, PriorArtVerdictKind, QueueItem
from .search.corpus import build_corpus, copy_corpus, freeze_curated, load_niche
from .search.coverage import candidate_text, t4_overlaps
from .search.scout import DIRECTION_STATUS_RULE, direction_status, link_claims
from .search.tighten import Tightener
from .store import Store


@dataclass
class SessionOpts:
    session: str
    profile: str = "full"
    llm_mode: Mode = "live"
    force: bool = False
    niche: str | None = None      # path to a niche.yaml: run the live literature search for it
    corpus: str | None = None     # path to an already frozen corpus (from `make search`): no network
    fetch: Callable[[str], str] | None = None      # HTTP transport override (tests)
    sleep: Callable[[float], None] | None = None
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


def gap_fixture(item: QueueItem, best: gaps.Gap, baseline: ExperimentConfig, tag: str = "") -> Fixture:
    """A hypothesis seed for a graph gap. The delta, the label and the explanation path are code's;
    the Scientist only writes the text and names the claims it rests on. `tag` distinguishes the
    same change tried again as a delta on a later incumbent (a different experiment)."""
    change = " and ".join(f"{f} to {v} instead of {getattr(baseline, f)}" for f, v in sorted(item.delta.items()))
    return Fixture(
        fixture_id="gap_" + re.sub(r"\W+", "_", item.key) + tag, kind="gap", origin="generated",
        candidate_key=item.key, title=f"{item.key.replace('+', ' + ')} ({item.novelty_type}, from a graph gap)"
        + (f" — on incumbent {tag.lstrip('_i')}" if tag else ""),
        statement=(f"Setting {change} lowers the validation loss of a tiny character-level Transformer "
                   "language model trained for a fixed 2,000-step budget."),
        allowed_fields=sorted(item.delta), cited_claim_ids=list(item.supporting_claim_ids),
        queue_priority=list(item.priority), required_delta=dict(item.delta), novelty_type=item.novelty_type,
        gap_ids=list(item.gap_ids), gap=best.model_dump(),
        mock={"same_comparison": False, "claim_id": "",
              "rationale": "(mock) no retrieved claim tests this exact change.",
              "config_changes": [{"field": f, "value": v} for f, v in sorted(item.delta.items())],
              "grounded_in": list(item.supporting_claim_ids)[:1]})


class Session:
    def __init__(self, o: SessionOpts):
        self.o = o
        self.replay = o.llm_mode == "replay"
        root = Path(__file__).resolve().parents[1]
        self.src = root / load_config()["paths"]["results"] / o.session
        # replay runs under the config the bundle was recorded with, never today's config.yaml
        self.cfg = bundle_config(self.src) if self.replay else load_config()
        self.protocol = protocol(self.cfg)
        self.profile = get_profile(o.profile, self.cfg)
        self.out = (root / self.cfg["paths"]["work"] / f"replay-{o.session}") if self.replay else self.src
        fresh_dir(self.out, force=self.replay or o.force)
        self.store = Store(self.out / "notebook.sqlite")
        fx = json.loads(path_of("fixtures").read_text())["fixtures"]
        self.fixtures = {f["fixture_id"]: Fixture(**f) for f in fx}
        self._load_corpus()
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
        self.lit = LiteratureAgent(self.llm, self.snap, overrides=tuple(self.niche.pi_overrides),
                                   version=self.protocol["gate_version"])
        self.scientist, self.critic = Scientist(self.llm, self.snap), Critic(self.llm)
        self.baseline = baseline_config(self.profile, self.cfg)
        self.incumbent = self.baseline       # what candidates are deltas on; changes only by promotion
        self.incumbent_delta: dict[str, str] = {}
        self.seeds = self.cfg["seeds"]
        self.rule = self.cfg["rule"]
        self.noise: stats.NoiseFloor | None = None
        self.base_by_seed: dict = {}
        self.runs_avoided = 0
        self._init_explore()
        self.soft: set[str] = set()          # keys soft-rejected at the gate (end of the queue)
        self.holds = self._t4_holds()        # key -> [(unextracted paper id, similarity)]
        self.held = set(self.holds) - set(self.niche.pi_dismissed_holds)
        self.tested: set[str] = set()        # "field=value" keys tested or gated out this session
        self.followups = 0                   # queue candidates taken so far (bounded by the budget)
        self.extra_done: set[str] = set()    # hypotheses that already got extra seeds (never twice)
        self.exhausted = False
        self.lcfg = self.cfg.get("search_loop", {})
        self.explored: list[dict] = []
        self.explored_only: set[str] = set()     # explored on one seed, never confirmed
        self.promotions = 0
        self.noise_did = ""
        self.trigger = ""                    # decision that led to the hypothesis being evaluated now

    # ------------------------------------------------------------------ literature corpus
    def _load_corpus(self) -> None:
        """Search live, then freeze: the session only ever reads `<bundle>/corpus/`."""
        o = self.o
        if self.replay:
            cdir = self.src / "corpus"
            if not cdir.exists():        # bundle recorded before the corpus was frozen per session
                self.niche = load_niche()
                self.snap = Snapshot(path_of("papers"), path_of("claims"))
                self.corpus_dir = None
                return
        else:
            cdir = self.out / "corpus"
            if o.corpus:
                copy_corpus(Path(o.corpus), cdir)
            elif o.niche:
                lcfg = self.cfg["lit_search"]
                llm = LLM(o.llm_mode, self.cfg["llm"], recordings_path=cdir / "llm.jsonl",
                          mock_fn=make_mock(self.fixtures) if o.llm_mode == "mock" else None,
                          max_cost_usd=lcfg["max_llm_cost_usd"])
                kw = {k: v for k, v in (("fetch", o.fetch), ("sleep", o.sleep)) if v}
                build_corpus(cdir, load_niche(o.niche), llm, today=dt.date.today().isoformat(), **kw)
            else:
                freeze_curated(cdir, load_niche())
        self.corpus_dir = cdir
        self.niche = load_niche(cdir / "niche.yaml")
        self.snap = Snapshot.from_corpus(cdir)

    def _init_explore(self) -> None:
        """Hypothesis engine (NEXT_VERSION): active only when the frozen corpus has scout directions.
        Gaps are computed once, by code, from the frozen claims; they order the queue."""
        ecfg = self.cfg.get("explore", {})
        self.explore = bool(ecfg.get("enabled")) and bool(self.snap.directions) and self.corpus_dir is not None
        self.loop = self.explore and self.protocol["search_loop"]
        self.derived: list = []              # our own results written back into the gap graph
        self.cycle = 0
        if not self.explore:
            return
        # The gap engine generates norm=rmsnorm from claim c04, so Fixture B (RMSNorm) is redundant
        # here; its verdict also depended on corpus size (method_known_setting_untested on the
        # curated snapshot, not_found_in_corpus at 160 papers). Drop it; keep Fixture A as the
        # guaranteed prior-art demo. (FIXES3 P0-3.) Gated on the v2 protocol so that explore1, an
        # earlier (v1) session recorded with Fixture B, still replays unchanged.
        if self.protocol["gate_version"] >= 2:
            self.fixtures.pop("fixture_b_rmsnorm", None)
        self.ecfg = ecfg
        self.frozen_claims = sorted(self.snap.claims.values(), key=lambda c: c.claim_id)
        self.graph, self.gaps = gaps.find_gaps(self.frozen_claims)
        self.gap_scientist = GapScientist(self.llm, self.snap)
        lcfg, o = self.cfg["lit_search"], self.o
        llm = None if self.replay else LLM(
            o.llm_mode, self.cfg["llm"], recordings_path=self.corpus_dir / "llm.jsonl",
            mock_fn=make_mock(self.fixtures) if o.llm_mode == "mock" else None,
            max_cost_usd=lcfg["max_llm_cost_usd"])
        self.tightener = Tightener(self.corpus_dir, self.niche, replay=self.replay, llm=llm, fetch=o.fetch,
                                   sleep=o.sleep, min_interval_s=lcfg["arxiv_min_interval_s"],
                                   n_queries=ecfg["tighten_queries"], cap=ecfg["tighten_cap"],
                                   today=dt.date.today().isoformat())

    def _t4_holds(self) -> dict[str, list]:
        """Candidates whose hypothesis text is close to an unextracted (T4) abstract."""
        baseline = self.baseline
        thr = self.cfg["lit_search"]["t4_threshold"]
        holds = {}
        for item in stats.candidate_queue(baseline, self.snap.claims.values(), []):
            text = candidate_text(item.field, item.value,
                                  [self.snap.claims[c] for c in item.supporting_claim_ids])
            hits = t4_overlaps(text, list(self.snap.papers.values()), thr)
            if hits:
                holds[item.key] = [[pid, score] for pid, score in hits]
        return holds

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
        self.orig_noise, self.orig_base_by_seed = self.noise, dict(self.base_by_seed)   # never modified
        self.store.put_noise("noise_baseline", self.noise)
        did = self.noise_did = self.store.add_decision("noise_floor", "", {
            "baseline_config_hash": self.baseline.config_hash(), "profile": self.profile.name,
            "mean": self.noise.mean, "sd": self.noise.sd, "n": self.noise.n})
        for r in runs:
            self.store.link("decision", did, "measured_by", "run", r.run_id)

    # ------------------------------------------------------------------ one hypothesis
    def evaluate(self, fx: Fixture, seeds: list[int] | None = None) -> Outcome:
        self._check_wall()
        if fx.kind == "gap":
            return self._evaluate_gap(fx, seeds or self.seeds["paired"])
        hid = f"hyp_{fx.fixture_id}"
        self._set_status(hid, fx, "proposed", statement=fx.statement, title=fx.title, kind=fx.kind,
                         fixture=fx.origin == "fixture", origin=fx.origin.upper(),
                         supporting_claim_ids=list(fx.cited_claim_ids), candidate_key=fx.candidate_key,
                         queue_priority=fx.queue_priority)
        if fx.candidate_key:
            self.tested.add(fx.candidate_key)
        seeds = seeds or self.seeds["paired"]

        # 1. Prior-art gate (declared LLM channel: verdict shown with the exact passage). For a
        # generated candidate, only claims that test exactly this change may decide the verdict.
        require = stats.parse_delta_key(fx.candidate_key) if fx.origin == "generated" and fx.candidate_key else None
        pa = self.lit.check(fx, require_change=require)
        self.store.link("hypothesis", hid, "gated_by", "event", pa.event_id)
        if pa.claim_id:
            self.store.link("hypothesis", hid, "nearest_claim", "claim", pa.claim_id)
            self.store.link("claim", pa.claim_id, "from_paper", "paper", pa.paper_id)
        if pa.gate == "soft_reject":
            # covered by an auto-extracted / preprint claim: not run, end of the queue, PI may override
            self._set_status(hid, fx, "soft_rejected_prior_art", prior_art=pa.model_dump(mode="json"))
            did = self.store.add_decision("prior_art_soft_rejection", hid, self._chained({
                "runs_deferred": len(seeds), "passage": pa.passage, "paper_id": pa.paper_id,
                "label": pa.label, "claim_id": pa.claim_id, "tier": pa.tier,
                "note": "covered by an auto-extracted / preprint claim; the PI may override "
                        "(niche.yaml: pi_overrides)"}))
            self.store.link("decision", did, "rests_on", "claim", pa.claim_id)
            self._link_trigger(did)
            if fx.candidate_key:
                self.tested.discard(fx.candidate_key)
                self.soft.add(fx.candidate_key)
            return Outcome(fx.fixture_id, hid, "soft_rejected_prior_art")
        if pa.verdict == PriorArtVerdictKind.known and pa.gate == "reject":
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
            self.tested.add(stats.delta_key(delta))
        self._set_status(hid, fx, "proposed", prior_art=pa.model_dump(mode="json"),
                         proposal=prop.model_dump(mode="json"), config_delta=delta,
                         config_hash=cfg.config_hash(), candidate_config=cfg.model_dump())
        return self._screen(hid, fx, prop, cfg, delta, seeds)

    def _screen(self, hid: str, fx: Fixture, prop, cfg: ExperimentConfig, delta: dict, seeds: list[int]) -> Outcome:
        """Critic before compute, then the paired runs and the deterministic analysis."""
        rejected = self._critic(hid, fx, prop, cfg, delta)
        return rejected or self._paired(hid, fx, cfg, seeds)

    def _critic(self, hid: str, fx: Fixture, prop, cfg: ExperimentConfig, delta: dict) -> Outcome | None:
        review, crit_eid = self.critic.review(fx, prop, self.incumbent, cfg, delta)
        self.store.link("hypothesis", hid, "reviewed_by", "event", crit_eid)
        self._set_status(hid, fx, "queued", critic_pre=review.model_dump(mode="json"))
        if review.verdict == "reject":
            self._set_status(hid, fx, "rejected_critic")
            did = self.store.add_decision("critic_rejection", hid, self._chained({
                "reasons": review.reasons, "confounds": review.confounds}))
            self.store.link("decision", did, "rests_on", "event", crit_eid)
            self._link_trigger(did)
            return Outcome(fx.fixture_id, hid, "rejected_critic")
        return None

    def _paired(self, hid: str, fx: Fixture, cfg: ExperimentConfig, seeds: list[int]) -> Outcome:
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
                          promising_sd=self.rule["promising_sd"], harmful_sd=self.rule["harmful_sd"],
                          rule_version=self.protocol["rule_version"])
        self.store.put_analysis(a)
        self.store.link("analysis", a.analysis_id, "for", "hypothesis", hid)
        for p in a.pairs:
            self.store.link("analysis", a.analysis_id, "uses", "run", p.candidate_run_id)
            self.store.link("analysis", a.analysis_id, "uses", "run", p.baseline_run_id)
        pa = self._hyp(hid).get("prior_art", {})
        cited = fx.cited_claim_ids or ([pa["claim_id"]] if pa.get("claim_id") else [])
        if fx.kind == "gap":
            # a result is compared only with claims about exactly this change; a combination has
            # none by construction, so no relation is recorded (it would overclaim)
            key = stats.delta_key(self._hyp(hid).get("config_delta") or {})
            cited = [c for c in fx.cited_claim_ids if gaps.method_key(self.snap.claims[c]) == key]
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
        if fx.kind == "gap" and self.protocol["rule_version"] >= 2:     # score the engine's own prediction
            pred = (fx.gap or {}).get("predicted_direction", "")
            payload["predicted_direction"] = pred
            payload["prediction_outcome"] = stats.prediction_outcome(pred, a.branch)
        if triggered_by:
            payload["triggered_by"] = triggered_by
        did = self.store.add_decision(kind, hid, payload)
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        self.store.link("decision", did, "interpreted_in", "event", eid)
        if triggered_by:
            self.store.link("decision", did, "triggered_by", "decision", triggered_by)
        return a, did

    # ------------------------------------------------------------------ hypothesis from a graph gap
    def _evaluate_gap(self, fx: Fixture, seeds: list[int], explore_only: bool = False) -> Outcome:
        """Scientist writes the hypothesis → tighten (targeted search for exactly this change) →
        gate → at most ONE narrowing revision → Critic → paired runs. Code decides every step."""
        hid = f"hyp_{fx.fixture_id}"
        self._set_status(hid, fx, "proposed", statement=fx.statement, title=fx.title, kind=fx.kind,
                         fixture=False, origin="GENERATED", supporting_claim_ids=list(fx.cited_claim_ids),
                         candidate_key=fx.candidate_key, queue_priority=fx.queue_priority,
                         novelty_type=fx.novelty_type, gap_ids=fx.gap_ids, gap=fx.gap)
        self.tested.add(fx.candidate_key)
        delta, revision_of, overlap, rounds = dict(fx.required_delta), None, "", []
        tensor = gaps.coverage_tensor(self.graph)
        for rnd in (0, 1):
            prop, cfg, delta, sci_eid = self.gap_scientist.propose(fx, self.incumbent, delta,
                                                                   revision_of=revision_of, overlap=overlap)
            self.store.link("hypothesis", hid, "proposed_by", "event", sci_eid)
            key = stats.delta_key(delta)
            self.tested.add(key)
            tr = self.tightener.run(fx.fixture_id, rnd, delta, set(self.snap.papers))
            self.snap.add(tr["papers"], tr["claims"])
            for p in tr["papers"]:
                self.store.put_paper(p)
            for c in tr["claims"]:
                self.store.put_claim(c)
            mine = [c for c in self.frozen_claims if gaps.method_key(c) == key] if len(delta) == 1 else []
            contested_ids = stats.contested_claim_ids(mine)
            pa = self.lit.check(fx, question=prop.statement, extra_claim_ids=tuple(c.claim_id for c in tr["claims"]),
                                contested_ids=tuple(contested_ids), require_change=delta)
            self.store.link("hypothesis", hid, "gated_by", "event", pa.event_id)
            # code-level cover for a narrowed single change (e.g. a curated claim already covers it)
            status, cov_ids, _ = stats.literature_status(key, mine, overrides=self.niche.pi_overrides,
                                                         contested_open=True)
            gate = pa.gate if (rnd == 0 or status == "open" or pa.gate != "run") else (
                "reject" if status == "rejected_prior_art" else "soft_reject")
            # A claim that is one side of a disagreement is not prior art for the question
            # "which side holds in our setting?": the resolution hypothesis still runs.
            contested = pa.claim_id in contested_ids
            if gate != "run" and contested and pa.gate == gate:
                gate = "run"
            # A combination is never rejected by a claim about one of its methods (the verdict above
            # only sees claims about the WHOLE change). Such a partial-overlap claim instead drives a
            # narrowing: on the first round of a two-field hypothesis, defer to the tighten revision.
            partial = list(pa.partial_overlap_claim_ids)
            narrow = rnd == 0 and len(delta) == 2 and gate == "run" and bool(partial)
            quote_id = (pa.claim_id if pa.gate != "run" else
                        partial[0] if narrow else (cov_ids[0] if cov_ids else None))
            new_ids = {p.paper_id for p in tr["papers"]}
            texts = [self.snap.claims[c] for c in fx.cited_claim_ids if gaps.method_key(self.snap.claims[c]) in key.split("+")]
            hold = [h for h in t4_overlaps(candidate_text(key.replace("+", " "), "", texts),
                                           list(self.snap.papers.values()), self.cfg["lit_search"]["t4_threshold"])
                    if h[0] in new_ids] if fx.candidate_key not in self.niche.pi_dismissed_holds else []
            rounds.append({"round": rnd, "delta_key": key, "queries": tr["queries"],
                           "new_papers": len(tr["papers"]), "new_claims": len(tr["claims"]),
                           "already_in_corpus": len(tr.get("already_in_corpus", [])),
                           "verdict": pa.verdict.value, "gate": gate, "claim_id": quote_id,
                           "contested": contested,
                           # omitted when empty so bundles recorded before this field replay unchanged
                           **({"partial_overlap_claim_ids": partial} if partial else {}),
                           "passage": self.snap.claims[quote_id].source_span if quote_id else None,
                           "possible_overlap": [[pid, sc] for pid, sc in hold], "degraded": tr["degraded"]})
            label = gaps.novelty_type(delta, self.graph, tensor)
            common = dict(prior_art=pa.model_dump(mode="json"), proposal=prop.model_dump(mode="json"),
                          grounded_in=prop.grounded_in, config_delta=delta, config_hash=cfg.config_hash(),
                          candidate_config=cfg.model_dump(),
                          novelty_type=label if label in gaps.NOVELTY_ORDER else fx.novelty_type)
            if gate == "run" and not narrow and hold:
                self._set_status(hid, fx, "held_pi_review", **common, tighten={"outcome": "held", "rounds": rounds})
                did = self.store.add_decision("pi_review_hold", hid, self._chained({
                    "candidate_key": key, "status": "held", "possible_overlap": [[p, sc] for p, sc in hold],
                    "note": "possible overlap with an unextracted paper found by the targeted search; "
                            "dismiss with pi_dismissed_holds in the niche file"}))
                self._link_trigger(did)
                self.held.add(fx.candidate_key)
                self.tested.discard(fx.candidate_key)
                return Outcome(fx.fixture_id, hid, "held_pi_review")
            if gate == "run" and not narrow:
                outcome = "narrowed" if rnd else "passed"
                self._set_status(hid, fx, "proposed", **common, tighten={"outcome": outcome, "rounds": rounds})
                if explore_only:
                    return self._explore_run(hid, fx, prop, cfg, delta)
                return self._screen(hid, fx, prop, cfg, delta, seeds)
            if rnd == 0 and len(delta) == 2:        # one revision: narrow to the untested part.
                # Triggered by an exact-delta reject (rare) or a partial overlap on one method.
                revision_of = delta
                overlap = f'[{quote_id}] "{rounds[-1]["passage"]}"'
                continue
            break
        # overlap that could not be narrowed, or a second overlap: rejected, with the quote
        hard = gate == "reject"
        st_ = "rejected_prior_art" if hard else "soft_rejected_prior_art"
        self._set_status(hid, fx, st_, **common, tighten={"outcome": "rejected", "rounds": rounds})
        if hard:
            self.runs_avoided += len(seeds)
        did = self.store.add_decision("prior_art_rejection" if hard else "prior_art_soft_rejection", hid, self._chained({
            ("runs_avoided" if hard else "runs_deferred"): len(seeds), "passage": rounds[-1]["passage"],
            "claim_id": quote_id, "label": pa.label, "tier": self.snap.claims[quote_id].tier if quote_id else None,
            "paper_id": self.snap.claims[quote_id].paper_id if quote_id else None,
            "tighten_rounds": len(rounds),
            "note": "overlap found by the tighten pass" + ("" if hard else "; the PI may override (pi_overrides)")}))
        if quote_id:
            self.store.link("decision", did, "rests_on", "claim", quote_id)
        self._link_trigger(did)
        if not hard:
            self.tested.discard(fx.candidate_key)
            self.soft.add(fx.candidate_key)
        return Outcome(fx.fixture_id, hid, st_)

    # ------------------------------------------------------------------ explore cheaply, confirm rigorously
    def _explore_run(self, hid: str, fx: Fixture, prop, cfg: ExperimentConfig, delta: dict) -> Outcome:
        """One seed, against the incumbent's run on the same seed. Never a result: it only decides
        which hypotheses earn a paired confirmation."""
        rejected = self._critic(hid, fx, prop, cfg, delta)
        if rejected:
            return rejected
        seed = self.lcfg.get("explore_seed", 0)
        r = self.runner.get(cfg, seed)
        self.store.link("hypothesis", hid, "candidate_run", "run", r.run_id)
        if r.status != "ok":
            self._set_status(hid, fx, "run_failed", errors=[r.error])
            did = self.store.add_decision("run_failure", hid, self._chained({"errors": [r.error]}))
            self._link_trigger(did)
            return Outcome(fx.fixture_id, hid, "run_failed")
        base = self.base_by_seed[seed]
        est = stats.single_seed_estimate(r, base, self.noise.sd)
        row = {"key": stats.delta_key(delta), "hypothesis_id": hid, "fixture_id": fx.fixture_id,
               "cell": gaps.archive_cell(delta, self.graph), "seed": seed, "candidate_run_id": r.run_id,
               "baseline_run_id": base.run_id, "noise_sd": self.noise.sd, **est}
        self._set_status(hid, fx, "explored", exploration={**row, "label": stats.EXPLORATION_LABEL, "cycle": self.cycle})
        did = self.store.add_decision("exploration_result", hid, self._chained({
            **row, "label": stats.EXPLORATION_LABEL, "cycle": self.cycle,
            "incumbent_config_hash": self.incumbent.config_hash()}))
        self.store.link("decision", did, "measured_by", "run", r.run_id)
        self.store.link("decision", did, "measured_by", "run", base.run_id)
        self._link_trigger(did)
        self.explored.append(row)
        return Outcome(fx.fixture_id, hid, "explored", None, did)

    def _confirm(self, row: dict, select_did: str) -> tuple | None:
        """Paired screening of a finalist → the pre-registered rule → extra seeds. Returns the
        arguments for a promotion if the finalist qualifies (the cycle promotes at most one, after
        every finalist has been confirmed against the same incumbent)."""
        hid, fx = row["hypothesis_id"], self.fixtures[row["fixture_id"]]
        cfg, seeds = self._candidate_cfg(hid), list(self.seeds["paired"])
        self.trigger = select_did
        out = self._paired(hid, fx, cfg, seeds)
        if out.analysis is None:
            return None
        a = out.analysis
        shrink = stats.shrinkage(row["improvement_in_noise_sd"], a.improvement_in_noise_sd)
        rec = {"cycle": self.cycle, "candidate_key": row["key"], "analysis_id": a.analysis_id,
               "single_seed_sd": row["improvement_in_noise_sd"], "paired_sd": a.improvement_in_noise_sd,
               "shrinkage_sd": shrink, "single_seed_delta": row["delta"], "paired_mean_delta": a.mean_delta,
               "branch": a.branch, "triggered_by": out.decision_id}
        d_conf = self.store.add_decision("confirmation", hid, rec)
        self.store.link("decision", d_conf, "based_on", "analysis", a.analysis_id)
        self.store.link("decision", d_conf, "triggered_by", "decision", out.decision_id)
        self._set_status(hid, fx, self._hyp(hid)["status"], shrinkage=rec)
        if self.incumbent is not self.baseline:
            self._headline(hid, fx, cfg, seeds)
        did = self._decide(out)                         # rule v2: extra seeds / literature check
        latest = next(x for x in self.store.analyses() if x["analysis_id"] == self._hyp(hid)["latest_analysis_id"])
        final = Analysis(**latest)
        delta = {k: stats.fmt_value(v) for k, v in self.incumbent.diff(cfg).items()}
        dc = gaps.derived_claim(delta, final.branch)    # write the result back into the gap graph
        if dc:
            self.derived.append(dc)
        n_all = len(self.seeds["paired"]) + len(self.seeds["extra"])
        return (hid, fx, cfg, delta, final, did) if stats.should_promote(final, n_all) else None

    def _headline(self, hid: str, fx: Fixture, cfg: ExperimentConfig, seeds: list[int]) -> None:
        """After a promotion, deltas are measured on the incumbent; the headline number for every
        candidate is still its paired comparison with the ORIGINAL baseline and its noise floor."""
        cand = [self.runner.get(cfg, s) for s in seeds]
        a = stats.analyze(hid, cand, [self.orig_base_by_seed[s] for s in seeds], self.orig_noise,
                          all_base_runs=list(self.orig_base_by_seed.values()),
                          promising_sd=self.rule["promising_sd"], harmful_sd=self.rule["harmful_sd"],
                          rule_version=self.protocol["rule_version"])
        self.store.put_analysis(a)
        self.store.link("analysis", a.analysis_id, "for", "hypothesis", hid)
        self._set_status(hid, fx, self._hyp(hid)["status"], headline_analysis_id=a.analysis_id)
        did = self.store.add_decision("headline_vs_original_baseline", hid, {
            "analysis_id": a.analysis_id, "branch": a.branch, "improvement_in_noise_sd": a.improvement_in_noise_sd,
            "note": "informational: the decision used the delta on the incumbent"})
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)

    def _promote(self, hid: str, fx: Fixture, cfg: ExperimentConfig, delta: dict, a: Analysis, trigger: str) -> None:
        """The finalist becomes the incumbent. Its own runs on every seed become the new noise
        floor (no extra compute). The original baseline and its noise floor are never modified."""
        runs = [self.runner.get(cfg, s) for s in self.seeds["baseline"]]
        if any(r.status != "ok" for r in runs):
            return
        self.promotions += 1
        nid = f"noise_incumbent_{self.promotions}"
        noise = stats.noise_floor(runs)
        self.store.put_noise(nid, noise)
        did = self.store.add_decision("promotion", hid, {
            "cycle": self.cycle, "candidate_key": stats.delta_key(delta), "delta": delta,
            "analysis_id": a.analysis_id, "improvement_in_noise_sd": a.improvement_in_noise_sd,
            "previous_incumbent_hash": self.incumbent.config_hash(), "incumbent_config_hash": cfg.config_hash(),
            "noise_id": nid, "noise_mean": noise.mean, "noise_sd": noise.sd, "run_ids": noise.run_ids,
            "triggered_by": trigger,
            "note": "still promising on every seed: later candidates are deltas on this config; "
                    "headline numbers stay relative to the original baseline"})
        self.store.link("decision", did, "based_on", "analysis", a.analysis_id)
        self.store.link("decision", did, "triggered_by", "decision", trigger)
        for r in runs:
            self.store.link("decision", did, "measured_by", "run", r.run_id)
        # changes that were only explored on the previous incumbent may be explored again on this one
        self.tested -= self.explored_only
        self.explored_only = set()
        self.incumbent, self.noise = cfg, noise
        self.incumbent_delta = {**self.incumbent_delta, **delta}
        self.base_by_seed = {r.seed: r for r in runs}
        self._set_status(hid, fx, self._hyp(hid)["status"], promoted=True)

    def _loop(self) -> None:
        """Cycle: recompute gaps (with our own results written back) → explore the top open gaps on
        one seed → select finalists (code) → paired confirmation → shrinkage → promotion."""
        lc = self.lcfg
        cycles = []
        for cycle in range(1, lc.get("cycles", 1) + 1):
            if self.exhausted:
                break
            self.cycle, self.explored = cycle, []
            self.graph, self.gaps = gaps.find_gaps([*self.frozen_claims, *self.derived])
            cycles.append({"cycle": cycle, "derived": [c.model_dump() for c in self.derived],
                           "incumbent_delta": dict(self.incumbent_delta),
                           "gaps": [g.model_dump() for g in self.gaps]})
            self.store.set_meta("gap_cycles", cycles)
            taken = set(self.incumbent_delta)
            queue = [q for q in stats.open_items(self._queue()) if not set(q.delta) & taken][: lc.get("explore_k", 6)]
            if not queue:
                break
            start = self.store.add_decision("cycle_start", "", {
                "cycle": cycle, "candidates": [q.key for q in queue], "incumbent_delta": dict(self.incumbent_delta),
                "queue_inputs": self._decision_inputs()})
            for item in queue:
                if self.runner.n_executed + 1 > self.budget["max_runs"]:
                    self._stop("max_runs", f"run budget {self.budget['max_runs']} reached while exploring", start)
                    break
                self.trigger = start
                self._evaluate_gap(self._generate(item), list(self.seeds["paired"]), explore_only=True)
            if not self.explored:
                if self.exhausted:
                    break
                continue
            keys = stats.select_finalists(self.explored, lc.get("finalists_per_cycle", 2), lc.get("min_finalists", 1))
            sel = self.store.add_decision("finalists_selected", "", {
                "cycle": cycle, "explored": self.explored, "finalists": keys,
                "top_n": lc.get("finalists_per_cycle", 2), "min_n": lc.get("min_finalists", 1),
                "rule_text": stats.FINALIST_RULE_TEXT, "triggered_by": start})
            self.store.link("decision", sel, "triggered_by", "decision", start)
            promotable: list[tuple] = []
            self.explored_only |= {r["key"] for r in self.explored} - set(keys)
            for row in [r for k in keys for r in self.explored if r["key"] == k]:
                need = len(self.seeds["paired"]) - 1
                if self.exhausted or self.runner.n_executed + need > self.budget["max_runs"]:
                    if not self.exhausted:
                        self._stop("max_runs", f"run budget {self.budget['max_runs']} would be exceeded by "
                                   "confirming the next finalist", sel)
                    break
                cand = self._confirm(row, sel)
                if cand:
                    promotable.append(cand)
            if promotable:      # the best finalist that held on every seed becomes the incumbent for the next cycle
                # ties: the simpler change (fewer fields), then alphabetical
                best = min(promotable, key=lambda c: (-c[4].improvement_in_noise_sd, len(c[3]), stats.delta_key(c[3])))
                self._promote(*best)

    # ------------------------------------------------------------------ candidate queue
    def _queue_inputs(self) -> dict:
        return {"tested": sorted(self.tested), "soft_rejected": sorted(self.soft),
                "held": sorted(self.held), "overrides": sorted(self.niche.pi_overrides)}

    def _queue(self) -> list[QueueItem]:
        if self.explore:      # ordered by gap_score (NEXT_VERSION conflict 1)
            return gaps.gap_queue(self.gaps, self.graph, [*self.frozen_claims, *self.derived],
                                  **self._queue_inputs(), top_n=self.ecfg["top_gaps_for_hypotheses"])
        return stats.candidate_queue(self.baseline, self.snap.claims.values(), **self._queue_inputs())

    def _decision_inputs(self) -> dict:
        extra = {"mode": "gaps", "top_n": self.ecfg["top_gaps_for_hypotheses"]} if self.explore else {}
        if self.loop:
            extra["cycle"] = self.cycle
        return {**self._queue_inputs(), **extra}

    def _store_explore_meta(self) -> None:
        """Everything the gap map and the gap cards render from (all computed by code)."""
        links = link_claims(self.snap.directions, self.frozen_claims, self.snap.papers,
                            self.ecfg["link_threshold"])
        dirs = []
        for d in self.snap.directions:
            linked = [self.snap.claims[c] for c in links[d.direction_id]]
            dirs.append({**d.model_dump(), **direction_status(d, linked), "claim_ids": links[d.direction_id]})
        self.store.set_meta("directions", dirs)
        self.store.set_meta("gaps", [g.model_dump() for g in self.gaps])
        self.store.set_meta("gap_graph", {**self.graph.model_dump(), "communities": gaps.communities(self.graph),
                                          "shared_mechanisms": gaps.shared_mechanisms(self.graph),
                                          "tensor": gaps.coverage_tensor(self.graph)})
        self.store.set_meta("explore", {"config": self.ecfg, "gap_score_rule": gaps.GAP_SCORE_TEXT,
                                        "direction_status_rule": DIRECTION_STATUS_RULE})

    def _record_queue_gates(self) -> None:
        """What the frozen literature already rules out, before any LLM gate or compute."""
        kinds = {"rejected_prior_art": "queue_rejection", "soft_rejected": "queue_soft_rejection",
                 "held": "pi_review_hold"}
        for q in self._queue():
            if q.status == "open":
                continue
            claims = [self.snap.claims[c] for c in q.covering_claim_ids]
            did = self.store.add_decision(kinds[q.status], "", {
                "candidate_key": q.key, "status": q.status, "note": q.note,
                "covering": [{"claim_id": c.claim_id, "paper_id": c.paper_id, "tier": c.tier,
                              "published": self.snap.papers[c.paper_id].published,
                              "passage": c.source_span} for c in claims],
                "possible_overlap": self.holds.get(q.key, []) if q.status == "held" else []})
            for c in claims:
                self.store.link("decision", did, "rests_on", "claim", c.claim_id)

    def _generate(self, item: QueueItem) -> Fixture:
        fx = (gap_fixture(item, next(g for g in self.gaps if g.gap_id == item.gap_ids[0]), self.incumbent,
                          tag=f"_i{self.promotions}" if self.promotions else "")
              if self.explore else generate_fixture(item, self.baseline, self.snap))
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
        na = stats.next_action(a, extra_seeds=self.seeds["extra"], queue=queue,
                               rule_version=self.protocol["rule_version"])
        did = self.store.add_decision("next_action", cur.hypothesis_id, {
            "analysis_id": a.analysis_id, "next_action": na.model_dump(mode="json"),
            "triggered_by": cur.decision_id,
            "inputs": {"extra_seeds": self.seeds["extra"], **self._decision_inputs(),
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
            queue = stats.open_items(self._queue())
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
            self.store.set_meta("rule_text", stats.RULE_TEXTS[self.protocol["rule_version"]])
            self.store.set_meta("rule_version", self.protocol["rule_version"])
            self.store.set_meta("protocol", self.protocol)
            self.store.set_meta("rule_thresholds", self.rule)
            self.store.set_meta("snapshot", {"papers": self.snap.size, "claims": len(self.snap.claims),
                                             "dimensions": self.snap.dimensions(),
                                             "fetched": self.snap.fetched, "source": self.snap.source})
            self.store.set_meta("niche", self.niche.model_dump())
            self.store.set_meta("corpus", self.snap.meta)
            self.store.set_meta("t4_holds", self.holds)
            if self.explore:
                self._store_explore_meta()
            self.store.set_meta("objective", "Improve validation loss of a tiny character-level "
                                "Transformer under a fixed 2,000-step training budget.")
            self.store.set_meta("profile", self.profile.model_dump())
            self.store.set_meta("baseline_config", self.baseline.model_dump())
            self.noise_floor()
            self.store.set_meta("candidate_queue", [q.model_dump(mode="json") for q in self._queue()])
            if self.snap.meta.get("degraded"):
                self.store.add_decision("search_degraded", "", {"reasons": self.snap.meta["degraded"]})
            self._record_queue_gates()
            self.trigger = self.noise_did
            outcomes = [self.evaluate(f) for f in list(self.fixtures.values())]
            if self.loop:
                for o in outcomes:                       # the rule still fires on the fixtures' results
                    if o.analysis is not None and not self.exhausted:
                        self._decide(o)
                self._loop()
            else:
                self._chain([o for o in outcomes if o.analysis is not None])
            self.store.set_meta("runs_avoided", self.runs_avoided)
        except (BudgetExceeded, RunBudgetExceeded) as e:
            status = f"budget_exhausted: {e}"
            self.store.add_decision("budget_exhausted", "", {"reason": str(e)})
        meta = {"session": self.o.session, "profile": self.profile.name, "llm_mode": self.o.llm_mode,
                "status": status, "llm_cost_usd": round(self.llm.cost_usd, 4),
                "llm_calls": self.llm.n_calls, "runs_executed": self.runner.n_executed,
                "runs_avoided": self.runs_avoided, "model": self.cfg["llm"]["model"],
                "corpus": self.snap.meta.get("label", ""),
                "search_llm_cost_usd": self.snap.meta.get("llm_cost_usd", 0.0)}
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
