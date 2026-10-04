"""Stage 4: hypotheses from graph gaps, novelty labels, the tighten pass and the gap queue."""
import json

import pytest

from noesis_lab import gaps, stats
from noesis_lab.agents import GapScientist
from noesis_lab.cli import main
from noesis_lab.config import ROOT
from noesis_lab.literature import Snapshot
from noesis_lab.llm import LLM, LLMError
from noesis_lab.mock_llm import make_mock
from noesis_lab.orchestrator import Session, SessionOpts, gap_fixture
from noesis_lab.rederive import rederive
from noesis_lab.schemas import ExperimentConfig, GapScientistOutput, LiteratureOutputV2, QueueItem
from noesis_lab.search.corpus import load_niche
from noesis_lab.search.tighten import Tightener, tighten_queries
from noesis_lab.store import Store

from .test_gaps import DROP, LM_LARGE, MECH, OURS, RMS, ROPE, SWIGLU, claim
from .test_search import CFG, FakeRunner, fake_fetch


# --------------------------------------------------------------------------- novelty type (code)
def _g(cs):
    g = gaps.build_graph(cs)
    return g, gaps.coverage_tensor(g)


@pytest.mark.parametrize("delta,claims,expected", [
    ({"activation": "swiglu"}, [claim("a", SWIGLU, "improves", "T3", "none")], "transfer"),
    ({"activation": "swiglu"}, [claim("a", SWIGLU, "improves", "T3", "covers", fields=OURS)], "none"),      # covered
    ({"activation": "swiglu"}, [claim("a", SWIGLU, "context", "T3", "none")], "none"),                      # no result anywhere
    ({"activation": "swiglu"}, [claim("a", SWIGLU, "improves", "T3", "partial"),
                                claim("b", SWIGLU, "worse", "T3", "partial")], "resolution"),
    ({"activation": "swiglu", "pos_encoding": "rope"}, [claim("a", SWIGLU), claim("b", ROPE)], "combination"),
    ({"activation": "swiglu", "pos_encoding": "rope"},
     [claim("a", SWIGLU, paper="p"), claim("b", ROPE, paper="p")], "none"),                                 # a paper mentions both
    ({"activation": "swiglu", "pos_encoding": "rope"}, [claim("a", SWIGLU)], "none"),                       # one side has no claims
    ({"attention": "sparse"}, [claim("a", SWIGLU)], "new_mechanism"),                                       # not a typed delta
])
def test_novelty_type_rules(delta, claims, expected):
    assert gaps.novelty_type(delta, *_g(claims)) == expected


# --------------------------------------------------------------------------- gap queue
def _toy():
    return [claim("c1", SWIGLU, "improves", "T3", "none", mech=MECH, fields=LM_LARGE),
            claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS),
            claim("c3", ROPE, "improves", "T3", "partial", mech=MECH, fields=LM_LARGE),
            claim("c4", ROPE, "worse", "T3", "partial", fields=LM_LARGE),
            claim("c5", DROP, "improves", "T1", covers=True),
            claim("c6", RMS, "improves", "T3", "none", mech="gradient_stability", fields=LM_LARGE)]


def test_gap_queue_orders_by_score_merges_gaps_per_delta_and_applies_literature_status():
    cs = _toy()
    g, found = gaps.find_gaps(cs)
    q = gaps.gap_queue(found, g, cs, top_n=20)
    keys = [i.key for i in q]
    assert len(keys) == len(set(keys))                                   # one item per config delta
    opens = [i for i in q if i.status == "open"]
    assert [i.gap_score for i in opens] == sorted((i.gap_score for i in opens), reverse=True)
    sw = next(i for i in q if i.key == "activation=swiglu")
    assert {x.split(":")[0] for x in sw.gap_ids} >= {"coverage", "abc"}  # same delta from two gaps: one item, both linked
    assert sw.delta == {"activation": "swiglu"} and sw.novelty_type == "transfer" and set(sw.supporting_claim_ids) >= {"c1", "c2"}
    rms = next(i for i in q if i.key == "norm=rmsnorm")
    assert "dropout=0.1" not in keys                                     # covered by a curated claim: no gap
    combo = next(i for i in q if "+" in i.key)
    assert combo.status == "open" and combo.novelty_type == "combination" and len(combo.delta) == 2
    soft = gaps.gap_queue(found, g, cs, soft_rejected=["norm=rmsnorm"], top_n=20)
    assert [i.key for i in soft if i.status != "open"] == ["norm=rmsnorm"] and soft[-1].key == rms.key   # end of the queue


def test_gap_queue_top_n_tested_overrides_and_determinism():
    cs = _toy()
    g, found = gaps.find_gaps(cs)
    full = gaps.gap_queue(found, g, cs, top_n=20)
    top2 = gaps.gap_queue(found, g, cs, top_n=2)
    ranked_keys = list(dict.fromkeys(x.delta_key for x in found if x.score > 0))
    assert {i.key for i in top2} == set(ranked_keys[:2])                 # only the top deltas get hypotheses
    assert "activation=swiglu" not in [i.key for i in gaps.gap_queue(found, g, cs, ["activation=swiglu"], top_n=20)]
    over = gaps.gap_queue(found, g, cs, soft_rejected=["norm=rmsnorm"], overrides=["norm=rmsnorm"], top_n=20)
    assert next(i for i in over if i.key == "norm=rmsnorm").status == "open"
    held = gaps.gap_queue(found, g, cs, held=["activation=swiglu"], top_n=20)
    assert next(i for i in held if i.key == "activation=swiglu").status == "held"
    assert [i.model_dump() for i in gaps.gap_queue(found, g, list(reversed(cs)), top_n=20)] == [i.model_dump() for i in full]


def test_tested_candidates_do_not_use_up_top_n_slots():
    cs = _toy()
    g, found = gaps.find_gaps(cs)
    ranked = list(dict.fromkeys(x.delta_key for x in found if x.score > 0))
    q = gaps.gap_queue(found, g, cs, ranked[:2], top_n=2)                # the two best are already tested
    assert [i.key for i in sorted(q, key=lambda i: i.priority)] == ranked[2:4]    # filter first, then cut


def test_a_contested_topic_is_queued_as_open_not_soft_rejected():
    cs = _toy()                                                          # rope: improves vs worse, both partial
    g, found = gaps.find_gaps(cs)
    rope = next(i for i in gaps.gap_queue(found, g, cs, top_n=20) if i.key == "pos_encoding=rope")
    assert rope.status == "open" and rope.novelty_type == "resolution" and rope.note.startswith("open (contested)")
    agree = [c for c in cs if c.claim_id != "c4"]                        # without the dissenting claim it is covered
    g, found = gaps.find_gaps(agree)
    rope = next(i for i in gaps.gap_queue(found, g, agree, top_n=20) if i.key == "pos_encoding=rope")
    assert rope.status == "soft_rejected" and rope.covering_claim_ids == ["c3"]


def test_queue_items_without_gap_fields_serialize_as_before():
    old = QueueItem(key="a=b", field="a", value="b", priority=[0, -1, "a=b"], supporting_claim_ids=["c"])
    assert set(old.model_dump()) == {"key", "field", "value", "priority", "supporting_claim_ids", "status",
                                     "covering_claim_ids", "note"}


# --------------------------------------------------------------------------- tighten search
def test_tighten_queries_name_every_method_of_the_delta():
    assert tighten_queries({"activation": "swiglu"}) == ['"SwiGLU" AND "language model"', '"SwiGLU" AND "Transformer"']
    both = tighten_queries({"pos_encoding": "rope", "activation": "swiglu"}, 1)
    assert both == ['"SwiGLU" AND "rotary position embedding" AND "language model"']


def test_tightener_freezes_its_result_and_replay_reads_it_without_network(tmp_path):
    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    live = Tightener(tmp_path, load_niche(), replay=False, llm=llm, fetch=fake_fetch, sleep=lambda s: None,
                     min_interval_s=0.0, today="2026-10-03")
    out = live.run("gap_activation_swiglu", 0, {"activation": "swiglu"}, known_papers={"2405.00005"})
    ids = [p.paper_id for p in out["papers"]]
    assert "2405.00005" not in ids and "2402.00002" in ids and out["already_in_corpus"] == ["2405.00005"]
    assert [q["n_results"] for q in out["queries"]] == [4, 4]
    assert all(c.source_span in next(p for p in out["papers"] if p.paper_id == c.paper_id).abstract for c in out["claims"])
    assert (tmp_path / "tighten" / "gap_activation_swiglu_r0.json").exists()

    def no_network(url):
        raise AssertionError("replay must not fetch")

    again = Tightener(tmp_path, load_niche(), replay=True, fetch=no_network).run(
        "gap_activation_swiglu", 0, {"activation": "swiglu"}, set())
    assert [p.model_dump() for p in again["papers"]] == [p.model_dump() for p in out["papers"]]
    assert [c.model_dump() for c in again["claims"]] == [c.model_dump() for c in out["claims"]]


# --------------------------------------------------------------------------- gap Scientist: code checks
class _Snap:
    def __init__(self, cs):
        self.claims = {c.claim_id: c for c in cs}


def _scientist(reply):
    def fn(role, system, user, schema):
        return GapScientistOutput(statement="s", mechanism="m", predicted_direction="lower_val_loss",
                                  falsification_rule="f", **reply)
    cs = _toy()
    g, found = gaps.find_gaps(cs)
    item = next(i for i in gaps.gap_queue(found, g, cs, top_n=20) if "+" in i.key)
    fx = gap_fixture(item, next(x for x in found if x.gap_id == item.gap_ids[0]), ExperimentConfig())
    return GapScientist(LLM("mock", CFG["llm"], mock_fn=fn), _Snap(cs)), fx, item


def _changes(delta):
    return [{"field": f, "value": v} for f, v in delta.items()]


def test_gap_hypothesis_must_use_the_required_delta_and_ground_itself_in_path_claims():
    sci, fx, item = _scientist({})
    ok = {"config_changes": _changes(item.delta), "grounded_in": fx.cited_claim_ids[:1]}
    sci, fx, item = _scientist(ok)
    out, cfg, delta, _ = sci.propose(fx, ExperimentConfig(), item.delta)
    assert delta == item.delta and out.grounded_in == fx.cited_claim_ids[:1]
    assert all(str(getattr(cfg, f)) == v for f, v in item.delta.items())
    for bad in ({**ok, "grounded_in": ["c_invented"]}, {**ok, "grounded_in": []},
                {**ok, "config_changes": _changes({"norm": "rmsnorm"})},                     # not the gap's delta
                {**ok, "config_changes": _changes({**item.delta, "lr": "0.01"})}):
        sci, fx, item = _scientist(bad)
        with pytest.raises(LLMError):
            sci.propose(fx, ExperimentConfig(), item.delta)


def test_a_revision_must_be_a_strict_subset_of_the_previous_delta():
    sci, fx, item = _scientist({})
    first = dict(list(item.delta.items())[:1])
    ok = {"config_changes": _changes(first), "grounded_in": fx.cited_claim_ids[:1]}
    sci, fx, item = _scientist(ok)
    _, _, delta, _ = sci.propose(fx, ExperimentConfig(), item.delta, revision_of=item.delta, overlap="[c] quote")
    assert delta == first
    for bad in (_changes(item.delta), _changes({"schedule": "constant"}), []):               # same, unrelated, empty
        sci, fx, item = _scientist({**ok, "config_changes": bad})
        with pytest.raises(LLMError):
            sci.propose(fx, ExperimentConfig(), item.delta, revision_of=item.delta, overlap="[c] quote")


# --------------------------------------------------------------------------- session
def explore_session(name, monkeypatch, literature=None, overrides=(), effect=lambda cfg: 0.0):
    """A mock session on the recorded search fixtures. `literature(user)` may override the gate's
    one LLM judgment for chosen hypotheses (returns a LiteratureOutputV2 or None). `overrides` are
    PI overrides written into the niche file, so replay sees them too."""
    import yaml
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    niche = ROOT / "niche.yaml"
    if overrides:
        doc = {**yaml.safe_load(niche.read_text()), "pi_overrides": list(overrides)}
        niche = ROOT / "work" / f"niche_{name}.yaml"
        niche.parent.mkdir(exist_ok=True)
        niche.write_text(yaml.safe_dump(doc))
    s = Session(SessionOpts(session=name, profile="smoke", llm_mode="mock", force=True,
                            niche=str(niche), fetch=fake_fetch, sleep=lambda x: None))
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", effect)
    base = s.llm.mock_fn

    def fn(role, system, user, schema):
        if schema is LiteratureOutputV2 and literature and literature(user):
            return literature(user)
        return base(role, system, user, schema)

    s.llm.mock_fn = fn
    return s


def test_a_queued_hypothesis_comes_from_a_graph_gap_and_the_session_replays(session_name, monkeypatch, capsys):
    s = explore_session(session_name, monkeypatch)
    assert s.explore and s.gaps
    out = s.run()
    d = ROOT / "results" / session_name
    st = Store(d / "notebook.sqlite", readonly=True)
    hyps = {h["hypothesis_id"]: h for h in st.hypotheses()}
    gap_hyps = [h for h in hyps.values() if h.get("kind") == "gap"]
    assert out["status"] == "complete" and gap_hyps
    h = gap_hyps[0]
    assert h["novelty_type"] in ("transfer", "combination", "resolution") and h["origin"] == "GENERATED"
    assert h["gap"]["path"] and {"plausibility", "coverage", "testability", "evidence_quality", "score"} <= set(h["gap"])
    claims = {c["claim_id"] for c in st.claims()}
    assert h["grounded_in"] and set(h["grounded_in"]) <= set(h["supporting_claim_ids"]) <= claims
    assert h["tighten"]["outcome"] == "passed" and h["tighten"]["rounds"][0]["queries"]
    assert h["status"].startswith("screened_") and stats.delta_key(h["config_delta"]) == h["candidate_key"]
    na = [x for x in st.decisions() if x["kind"] == "next_action"]
    assert na and all(x["inputs"]["mode"] == "gaps" for x in na)
    assert st.get_meta("gaps") and st.get_meta("gap_graph")["edges"] and st.get_meta("directions")
    combo = [x for x in gap_hyps if len(x["config_delta"]) == 2]
    ev = st.derived_evidence()
    assert all(e["analysis_id"] != x.get("latest_analysis_id") for x in combo for e in ev)   # no relation for a combination
    st.close()
    assert (d / "corpus" / "tighten").exists()
    manifest = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert any(k.startswith("corpus/tighten/") for k in manifest) and "corpus/scout.json" in manifest
    assert rederive(d) == []

    import urllib.request

    def no_network(*a, **k):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert main(["replay", "--session", session_name]) == 0
    assert "REPLAY OK" in capsys.readouterr().out


def _gap_fx(s, delta):
    key = stats.delta_key(delta)
    cited = sorted(c.claim_id for c in s.frozen_claims if gaps.method_key(c) in key.split("+"))
    item = QueueItem(key=key, field="+".join(sorted(delta)), value="x", priority=[-0.5, 1, key],
                     supporting_claim_ids=cited, delta=delta, gap_ids=["link:" + key], gap_score=0.5,
                     novelty_type="combination")
    best = gaps.Gap(gap_id="link:" + key, gap_type="link", novelty_type="combination", delta_key=key, delta=delta,
                    plausibility=0.5, coverage=0, testability=1, evidence_quality=1, score=0.5,
                    path=[{"from": key, "to": "x", "claim_id": cited[0]}], claim_ids=cited)
    fx = gap_fixture(item, best, s.baseline)
    s.fixtures[fx.fixture_id] = fx
    return fx


def _overlap_on_first_round(claim_id):
    def lit(user):
        if "(mock, narrowed)" in user or "Hypothesis id: gap_" not in user:
            return None
        return LiteratureOutputV2(same_comparison_claim_ids=[claim_id], rationale="tests the same combination")
    return lit


def test_tighten_overlap_gets_one_revision_and_the_narrowed_hypothesis_runs(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch, _overlap_on_first_round("x2401_00001_1"))
    s.noise_floor()
    fx = _gap_fx(s, {"dropout": "0.1", "pos_encoding": "rope"})           # rope part is covered (T2); dropout part is not
    out = s.evaluate(fx)
    h = s._hyp(out.hypothesis_id)
    assert out.status.startswith("screened_") and h["tighten"]["outcome"] == "narrowed"
    r0, r1 = h["tighten"]["rounds"]
    # the combination itself is not prior art (gate run); the rope claim covers only PART of it, so
    # it drives the narrowing rather than deciding the verdict
    assert (r0["gate"], r0["delta_key"]) == ("run", "dropout=0.1+pos_encoding=rope")
    assert r0["partial_overlap_claim_ids"] == ["x2401_00001_1"] and r0["claim_id"] == "x2401_00001_1"
    assert r0["passage"] in s.snap.papers["2401.00001"].abstract           # the overlapping quote is shown
    assert (r1["gate"], r1["delta_key"]) == ("run", "dropout=0.1")
    assert h["config_delta"] == {"dropout": "0.1"} and h["novelty_type"] == "transfer"   # relabeled by code
    assert len([e for e in s.store.events() if e["role"] == "scientist_revision"]) == 1  # exactly one revision
    s.store.close()


def test_second_overlap_is_rejected_with_the_quote(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch, _overlap_on_first_round("x2402_00002_1"))
    s.noise_floor()
    fx = _gap_fx(s, {"activation": "swiglu", "dropout": "0.1"})          # narrows to swiglu, which a claim covers
    n_runs = s.runner.n_executed
    out = s.evaluate(fx)
    h = s._hyp(out.hypothesis_id)
    assert out.status == "soft_rejected_prior_art" and h["tighten"]["outcome"] == "rejected"
    # round 0: a partial overlap on the swiglu part narrows the combination (gate run); round 1: the
    # narrowed single change swiglu is covered exactly, so it is soft-rejected -> no untested part left
    assert [r["gate"] for r in h["tighten"]["rounds"]] == ["run", "soft_reject"]
    dec = [d for d in s.store.decisions() if d["kind"] == "prior_art_soft_rejection"][-1]
    assert dec["passage"] and dec["claim_id"] == "x2402_00002_1" and dec["tighten_rounds"] == 2
    assert s.runner.n_executed == n_runs and "activation=swiglu+dropout=0.1" in s.soft   # no compute spent
    s.store.close()


def test_a_resolution_hypothesis_runs_even_when_the_gate_cites_one_side_of_the_disagreement(session_name, monkeypatch):
    from noesis_lab.schemas import Claim
    s = explore_session(session_name, monkeypatch, _overlap_on_first_round("x2402_00002_1"))
    pro = s.snap.claims["x2402_00002_1"]                                   # SwiGLU improves (T2, partial)
    con = Claim(**{**pro.model_dump(), "claim_id": "x_dissent", "expected_outcome": "worse", "tier": "T3"})
    s.snap.claims["x_dissent"] = con
    s.frozen_claims = sorted(s.snap.claims.values(), key=lambda c: c.claim_id)
    s.graph, s.gaps = gaps.find_gaps(s.frozen_claims)
    item = next(i for i in s._queue() if i.key == "activation=swiglu")
    assert item.status == "open" and item.novelty_type == "resolution"     # was soft-rejected before the fix
    s.niche.pi_dismissed_holds = ["activation=swiglu"]    # the fixture's targeted search also finds an unextracted re-upload
    s.noise_floor()
    out = s.evaluate(s._generate(item))
    h = s._hyp(out.hypothesis_id)
    r0 = h["tighten"]["rounds"][0]
    assert r0["verdict"] == "known_in_corpus" and r0["contested"] and r0["gate"] == "run"
    assert out.status.startswith("screened_") and h["tighten"]["outcome"] == "passed"
    s.store.close()


def test_single_field_overlap_cannot_be_narrowed_and_is_rejected_without_a_revision(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch, _overlap_on_first_round("x2405_00005_1"))
    s.noise_floor()
    item = next(i for i in s._queue() if i.key == "dropout=0.1")
    out = s.evaluate(s._generate(item))
    h = s._hyp(out.hypothesis_id)
    # x2405 does not cover our setting → method known, setting untested → the gate lets it run
    assert h["tighten"]["rounds"][0]["verdict"] == "method_known_setting_untested" and out.status.startswith("screened_")
    assert not [e for e in s.store.events() if e["role"] == "scientist_revision"]
    s.store.close()


def test_frozen_snapshot_of_a_bundle_still_loads_scout_and_gaps_offline(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch)
    s.run()
    snap = Snapshot.from_corpus(ROOT / "results" / session_name / "corpus")
    g, found = gaps.find_gaps(sorted(snap.claims.values(), key=lambda c: c.claim_id),
                              predict_combinations=True)          # prediction_version 2 (config.yaml)
    st = Store(ROOT / "results" / session_name / "notebook.sqlite", readonly=True)
    assert [x.model_dump() for x in found] == st.get_meta("gaps") and snap.directions
