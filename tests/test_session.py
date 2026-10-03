"""End-to-end on CPU with the mock LLM (no key, no GPU): golden path, replay, rederive, tamper checks."""
import json

import pytest

from prior_art.bundle import verify_manifest
from prior_art.cli import main
from prior_art.config import ROOT
from prior_art.orchestrator import SessionOpts, run_session
from prior_art.rederive import rederive
from prior_art.store import Store


@pytest.fixture
def smoke(session_name):
    out = run_session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True))
    return session_name, out, ROOT / "results" / session_name


def test_golden_path_structure(smoke):
    name, out, d = smoke
    assert out["status"] == "complete" and out["runs_avoided"] == 3
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
    dec = {x["kind"]: x for x in st.decisions()}
    branch = dec["screening_result"]["branch"]
    na = dec["next_action"]["next_action"]
    expected = {"promising": "extra_seeds", "harmful": "literature_check", "no_improvement": "next_candidate"}
    assert na["action"] == expected[branch] and na["branch"] == branch
    assert any(e.get("triggered_by") for e in st.decisions() if e["kind"] != "next_action") or branch == "promising"


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
