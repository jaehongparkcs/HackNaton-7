"""FIXES4: our own results (derived claims) in the graph must resolve by id, never act as prior
art or a search query, carry weak tiers, and a crash must seal a partial bundle not lose the runs."""
import json

from noesis_lab import gaps
from noesis_lab.cli import main
from noesis_lab.config import ROOT
from noesis_lab.rederive import rederive
from noesis_lab.search.coverage import counts_as_covered, is_derived
from noesis_lab.store import Store

from .test_explore import _gap_fx, explore_session
from .test_gaps import SWIGLU, claim
from .test_loop import OPEN, _promising_dropout


# --------------------------------------------------------------------------- tiers never cover
def test_derived_claims_are_never_prior_art_whatever_their_coverage():
    for tier in ("D-explore", "D-confirmed"):
        dc = gaps.derived_claim({"norm": "rmsnorm"}, "promising", tier=tier)
        assert dc.tier == tier and is_derived(dc) and dc.coverage == "covers"
        assert not counts_as_covered(dc)          # covers the graph bucket, but never prior art
    lit = claim("c", SWIGLU, "improves", "T1", covers=True)
    assert counts_as_covered(lit) and not is_derived(lit)


def test_tier_weights_rank_confirmed_above_explore():
    assert gaps.TIER_WEIGHT["D-confirmed"] > gaps.TIER_WEIGHT["D-explore"] == 0.3


def test_an_exploration_lowers_a_gap_score_but_a_confirmation_closes_it():
    cs = [claim("a", SWIGLU, "improves", "T3", "none")]               # swiglu has a coverage gap
    base_score = next(g for g in gaps.find_gaps(cs)[1] if g.delta_key == "activation=swiglu").score
    explored = gaps.derived_claim({"activation": "swiglu"}, "promising", tier="D-explore")
    g_ex = {x.delta_key: x for x in gaps.find_gaps([*cs, explored])[1]}
    assert "activation=swiglu" in g_ex and g_ex["activation=swiglu"].score < base_score   # lowered, not gone
    confirmed = gaps.derived_claim({"activation": "swiglu"}, "promising", tier="D-confirmed")
    assert "activation=swiglu" not in {x.delta_key for x in gaps.find_gaps([*cs, confirmed])[1]}   # closed


# --------------------------------------------------------------------------- resolve by id, gate stays literature
def test_a_gap_that_cites_our_own_claim_resolves_runs_and_never_reaches_the_gate(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch)
    s.noise_floor()
    ours = gaps.derived_claim({"norm": "rmsnorm"}, "promising", tier="D-confirmed")
    s._add_derived(ours)
    assert s._claim(ours.claim_id) is ours and ours.claim_id not in s.snap.claims   # combined map, not literature
    # a combination gap whose explanation path includes our own claim (the crash: KeyError ours_…)
    fx = _gap_fx(s, {"norm": "rmsnorm", "dropout": "0.1"})
    fx.cited_claim_ids = [*fx.cited_claim_ids, ours.claim_id]
    fx.gap["path"] = [*fx.gap["path"], {"from": "norm=rmsnorm", "to": "quality", "claim_id": ours.claim_id}]
    out = s.evaluate(fx)                                  # must not raise KeyError
    assert out.status.startswith("screened_")
    ev = s.store.events()
    gate_calls = [e for e in ev if e["role"] == "literature"]
    assert gate_calls and ours.claim_id not in " ".join(e["user"] for e in gate_calls)   # never shown to the gate
    h = s._hyp(out.hypothesis_id)
    assert ours.claim_id in h["supporting_claim_ids"]     # kept for provenance / the graph
    assert not any(e["claim_id"] == ours.claim_id for e in s.store.derived_evidence())    # never our-own evidence
    s.store.close()


def test_the_tighten_search_never_queries_our_own_result(session_name, monkeypatch):
    from noesis_lab.search.tighten import CHANGE_TERMS
    s = explore_session(session_name, monkeypatch)
    # the derived claim's method string is not a search term source; only CHANGE_TERMS drives queries
    assert "ours_" not in json.dumps(CHANGE_TERMS)
    s.store.close()


# --------------------------------------------------------------------------- full loop with derived claims
def test_a_loop_that_writes_results_back_completes_replays_and_rederives(session_name, monkeypatch, capsys):
    explore_session(session_name, monkeypatch, overrides=OPEN, effect=_promising_dropout).run()
    d = ROOT / "results" / session_name
    st = Store(d / "notebook.sqlite", readonly=True)
    claims = {c["claim_id"]: c for c in st.claims()}
    derived = [cid for cid in claims if cid.startswith("ours_")]
    assert derived and all(claims[cid]["tier"] in ("D-explore", "D-confirmed") for cid in derived)
    assert "ours_dropout_0_1" in claims and claims["ours_dropout_0_1"]["tier"] == "D-confirmed"
    # every stored claim id referenced by a hypothesis resolves (literature + our own)
    for h in st.hypotheses():
        for cid in h.get("supporting_claim_ids", []):
            assert cid in claims, cid
    st.close()
    assert rederive(d) == []
    assert main(["replay", "--session", session_name]) == 0 and "REPLAY OK" in capsys.readouterr().out


# --------------------------------------------------------------------------- crash seals a partial bundle
def test_an_unexpected_crash_seals_a_partial_bundle_marked_crashed(session_name, monkeypatch):
    from noesis_lab.bundle import verify_manifest
    from noesis_lab.orchestrator import Session, SessionOpts

    from .test_search import FakeRunner
    monkeypatch.setitem(__import__("noesis_lab.config", fromlist=["load_config"]).load_config()["lit_search"],
                        "min_papers", 3)
    from .test_search import fake_fetch
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True,
                            niche=str(ROOT / "niche.yaml"), fetch=fake_fetch, sleep=lambda x: None))
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", lambda cfg: 0.0)
    # blow up partway, AFTER the baseline runs have been produced and written to runs.jsonl
    orig = s.evaluate

    def boom(fx, *a, **k):
        if s.runner.n_executed >= 5:
            raise RuntimeError("synthetic explosion")
        return orig(fx, *a, **k)

    s.evaluate = boom
    out = s.run()
    d = ROOT / "results" / session_name
    assert out["status"].startswith("crashed:") and "synthetic explosion" in out["status"]
    assert (d / "runs.jsonl").exists() and len((d / "runs.jsonl").read_text().splitlines()) >= 5   # runs kept
    assert (d / "notebook.sqlite").exists() and (d / "MANIFEST.json").exists() and verify_manifest(d) == []
    st = Store(d / "notebook.sqlite", readonly=True)
    crashed = [x for x in st.decisions() if x["kind"] == "crashed"]
    assert crashed and "synthetic explosion" in crashed[0]["error"] and crashed[0]["traceback"]
    st.close()
