"""End-to-end on CPU with the mock LLM (no key, no GPU): golden path, replay, rederive, tamper checks."""
import json

import pytest

from noesis_lab.bundle import verify_manifest
from noesis_lab.cli import main
from noesis_lab.config import ROOT
from noesis_lab.orchestrator import Session, SessionOpts, run_session
from noesis_lab.rederive import rederive
from noesis_lab.runner import BaseRunner
from noesis_lab.schemas import CurvePoint, RunResult
from noesis_lab.store import Store


@pytest.fixture
def smoke(session_name):
    out = run_session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True))
    return session_name, out, ROOT / "results" / session_name


def test_golden_path_structure(smoke):
    name, out, d = smoke
    assert out["status"] == "complete" and out["runs_avoided"] >= 3
    st = Store(d / "notebook.sqlite", readonly=True)
    hyps = {h["fixture_id"]: h for h in st.hypotheses()}
    a = hyps["fixture_a_preln_nowarmup"]
    assert a["status"] == "rejected_prior_art"
    claims = {c["claim_id"]: c for c in st.claims()}
    papers = {p["paper_id"]: p for p in st.papers()}
    pa = a["prior_art"]
    assert pa["passage"] in papers[pa["paper_id"]]["abstract"]                 # verbatim, from the snapshot
    assert pa["passage"] == claims[pa["claim_id"]]["source_span"]
    b = hyps["fixture_b_rmsnorm"]
    assert b["status"].startswith("screened_") and b["config_delta"] == {"norm": "rmsnorm"}
    kinds = [x["kind"] for x in st.decisions()]
    assert kinds[0] == "noise_floor" and "next_action" in kinds
    # every analysis number traces to stored runs
    for an in st.analyses():
        for p in an["pairs"]:
            assert st.run(p["candidate_run_id"])["val_loss"] == p["candidate_val_loss"]
            assert st.run(p["baseline_run_id"])["val_loss"] == p["baseline_val_loss"]
    prov = st.provenance("decision", next(x for x in st.decisions() if x["kind"] == "screening_result")["decision_id"])
    assert json.dumps(prov).count('"type": "run"') >= 6


def test_followup_depends_on_measurement(smoke):
    _, _, d = smoke
    st = Store(d / "notebook.sqlite", readonly=True)
    dec: dict = {}
    for x in st.decisions():                               # first decision of each kind
        dec.setdefault(x["kind"], x)
    first = next(x for x in st.decisions() if x["kind"] == "screening_result")
    na = dec["next_action"]["next_action"]
    expected = {"promising": "extra_seeds", "harmful": "literature_check", "no_improvement": "next_candidate"}
    assert na["action"] == expected[first["branch"]] and na["branch"] == first["branch"]
    assert dec["next_action"]["triggered_by"] == first["decision_id"]


def test_replay_reproduces_state_exactly(smoke, capsys):
    name, out, _ = smoke
    assert main(["replay", "--session", name]) == 0
    assert "REPLAY OK" in capsys.readouterr().out


def test_rederive_and_manifest(smoke):
    name, _, d = smoke
    assert rederive(d) == [] and verify_manifest(d) == []


def test_tampering_with_a_stored_run_is_detected(smoke):
    name, _, d = smoke
    lines = (d / "runs.jsonl").read_text().splitlines()
    r = json.loads(lines[0])
    r["val_loss"] -= 0.5
    lines[0] = json.dumps(r)
    (d / "runs.jsonl").write_text("\n".join(lines) + "\n")
    assert verify_manifest(d) and rederive(d)          # manifest AND recomputation both object


# ----------------------------------------------------------------------------- the rule on every result
BASE_NOISE = [0.0, 0.01, -0.01, 0.005, -0.005]          # SD ~ 0.0079


class FakeRunner(BaseRunner):
    """Deterministic runs whose val_loss is chosen by the test, so branches can be forced."""

    def __init__(self, store, profile, runs_path, effect):
        super().__init__(store, profile)
        self.runs_path, self.effect = runs_path, effect

    def _produce(self, cfg, seed, rid):
        val = 1.50 + BASE_NOISE[seed] + self.effect(cfg)
        curve = [CurvePoint(train_s=0, step=0, tokens_seen=0, val_loss=4.0),
                 CurvePoint(train_s=1, step=10, tokens_seen=1000, val_loss=val)]
        return RunResult(run_id=rid, config_hash=cfg.config_hash(), config=cfg,
                         profile_hash=self.profile.profile_hash(), seed=seed, status="ok", val_loss=val,
                         tokens_seen=1000, steps=10, train_seconds=1.0, curve=curve,
                         run_order=len(self.cache) + 1)

    def _persist(self, r):
        with self.runs_path.open("a") as f:
            f.write(r.model_dump_json() + "\n")


def run_forced(name, effect, **budget):
    s = Session(SessionOpts(session=name, profile="smoke", llm_mode="mock", force=True))
    s.budget = {**s.budget, **budget}
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", effect)
    out = s.run()
    return out, Store(s.out / "notebook.sqlite", readonly=True)


def swiglu_effect(delta):
    return lambda cfg: delta if cfg.activation == "swiglu" else 0.0      # everything else: noise only


def test_promising_followup_gets_extra_seeds_and_a_decision_chain(session_name):
    _, st = run_forced(session_name, swiglu_effect(-0.05))
    decs = st.decisions()
    hyps = {h["hypothesis_id"]: h for h in st.hypotheses()}
    sw = hyps["hyp_gen_activation_swiglu"]
    assert sw["status"].startswith("screened_") and sw["origin"] == "GENERATED"
    assert sw["supporting_claim_ids"] == ["c09"]
    extra = [d for d in decs if d["kind"] == "extra_seeds_result" and d["hypothesis_id"] == sw["hypothesis_id"]]
    assert len(extra) == 1                                                  # extra seeds, exactly once
    by_id = {d["decision_id"]: d for d in decs}
    depth = 0
    for d in decs:                                                          # longest triggered_by chain
        n, cur = 1, d
        while cur.get("triggered_by") in by_id:
            cur, n = by_id[cur["triggered_by"]], n + 1
        depth = max(depth, n)
    assert depth >= 3


def test_harmful_followup_triggers_the_literature_check(session_name):
    _, st = run_forced(session_name, swiglu_effect(+0.05))
    checks = [d for d in st.decisions() if d["kind"] == "literature_check"]
    assert any(d["hypothesis_id"] == "hyp_gen_activation_swiglu" for d in checks)


def test_followups_come_from_the_queue_not_from_config(session_name):
    _, st = run_forced(session_name, swiglu_effect(0.0))
    fups = [d for d in st.decisions() if d["kind"] == "followup_candidate"]
    assert fups and all(d["supporting_claim_ids"] for d in fups)
    queue = st.get_meta("candidate_queue")
    assert [d["candidate_key"] for d in fups] == [q["key"] for q in queue
                                                  if q["key"] != "norm=rmsnorm"][:len(fups)]
    for d in st.decisions():                                                 # queue stored with every rule firing
        if d["kind"] == "next_action":
            assert "norm=rmsnorm" in d["inputs"]["tested"]
            assert all("norm=rmsnorm" != q["key"] for q in d["inputs"]["queue"])


def test_bounds_are_recorded_as_budget_exhausted(session_name):
    out, st = run_forced(session_name, swiglu_effect(0.0), max_followup_candidates=1)
    stop = [d for d in st.decisions() if d["kind"] == "budget_exhausted"]
    assert stop and stop[-1]["bound"] == "max_followup_candidates"
    session2 = session_name + "-b"
    try:
        _, st2 = run_forced(session2, swiglu_effect(0.0), max_runs=9)      # 5 baseline + 3 (B) + 3 > 9
    finally:
        import shutil
        shutil.rmtree(ROOT / "results" / session2, ignore_errors=True)
    assert [d["bound"] for d in st2.decisions() if d["kind"] == "budget_exhausted"] == ["max_runs"]


def test_forced_session_replays_and_rederives(session_name, capsys):
    run_forced(session_name, swiglu_effect(-0.05))
    assert main(["replay", "--session", session_name]) == 0
    assert "REPLAY OK" in capsys.readouterr().out
    assert rederive(ROOT / "results" / session_name) == []
