"""FIXES2 P1: explore cheaply (1 seed), confirm rigorously (paired), shrinkage, promotion."""
import pytest

from noesis_lab import gaps, stats
from noesis_lab.cli import main
from noesis_lab.config import ROOT
from noesis_lab.rederive import rederive
from noesis_lab.store import Store

from .conftest import make_run
from .test_explore import explore_session
from .test_gaps import DROP, MECH, ROPE, SWIGLU, claim
from .test_stats import BASE, run

OPEN = ("activation=swiglu", "pos_encoding=rope")      # PI overrides: more open candidates in the fixture corpus


# --------------------------------------------------------------------------- pure functions
def test_single_seed_estimate_is_the_same_seed_difference_in_noise_sds():
    est = stats.single_seed_estimate(make_run(1.47, 0), make_run(1.50, 0), 0.015)
    assert est == {"delta": pytest.approx(-0.03), "improvement": pytest.approx(0.03),
                   "improvement_in_noise_sd": pytest.approx(2.0)}
    assert stats.single_seed_estimate(make_run(1.52, 0), make_run(1.50, 0), 0.0)["improvement_in_noise_sd"] == 0.0


def _row(key, cell, imp):
    return {"key": key, "cell": cell, "improvement_in_noise_sd": imp}


def test_finalists_best_per_cell_then_top_n_by_single_seed_improvement():
    explored = [_row("a", "c1", 2.0), _row("b", "c1", 3.0),            # same cell: only the better one survives
                _row("c", "c2", 1.0), _row("d", "c3", 0.5), _row("e", "c4", -1.0)]
    assert stats.select_finalists(explored, top_n=2) == ["b", "c"]
    assert stats.select_finalists(explored, top_n=5) == ["b", "c", "d"]              # "e" did not improve
    assert stats.select_finalists(list(reversed(explored)), top_n=2) == ["b", "c"]  # deterministic
    tie = [_row("z", "c1", 1.0), _row("y", "c2", 1.0)]
    assert stats.select_finalists(tie, top_n=1) == ["y"]                             # ties: alphabetical


def test_finalists_minimum_confirms_the_best_even_if_nothing_improved():
    worse = [_row("a", "c1", -0.5), _row("b", "c2", -2.0)]
    assert stats.select_finalists(worse, top_n=2, min_n=1) == ["a"]
    assert stats.select_finalists(worse, top_n=2, min_n=0) == []
    assert stats.select_finalists([], top_n=2, min_n=1) == []


def test_shrinkage_is_single_minus_paired():
    assert stats.shrinkage(2.5, 1.0) == 1.5 and stats.shrinkage(-0.5, 0.5) == -1.0


def test_promotion_needs_promising_on_every_one_of_the_full_seed_set():
    five = run(BASE, [1.45, 1.47, 1.43, 1.46, 1.44])
    assert stats.should_promote(five, 5)
    assert not stats.should_promote(run(BASE, [1.45, 1.47, 1.43]), 5)                 # only 3 seeds
    assert not stats.should_promote(run(BASE, [1.40, 1.40, 1.40, 1.40, 1.50]), 5)     # one seed is not better
    assert not stats.should_promote(run(BASE, [1.60, 1.58, 1.57, 1.59, 1.60]), 5)     # harmful


def test_archive_cell_is_method_family_times_mechanisms():
    g = gaps.build_graph([claim("a", SWIGLU, mech=MECH), claim("b", ROPE), claim("c", DROP, mech="implicit_regularization")])
    assert gaps.archive_cell({"activation": "swiglu"}, g) == f"activation | {MECH}"
    assert gaps.archive_cell({"pos_encoding": "rope"}, g) == "pos_encoding | no mechanism"
    assert gaps.archive_cell({"dropout": "0.1", "activation": "swiglu"}, g) == f"activation+dropout | implicit_regularization, {MECH}"


def test_our_result_is_written_back_as_a_derived_edge_and_closes_the_coverage_gap():
    cs = [claim("a", SWIGLU, "improves", "T3", "none")]
    assert [x.gap_type for x in gaps.find_gaps(cs)[1]] == ["coverage"]
    dc = gaps.derived_claim({"activation": "swiglu"}, "harmful")
    assert (dc.expected_outcome, dc.tier, dc.coverage, dc.source_span) == ("worse", "T1", "covers", "")
    assert gaps.find_gaps([*cs, dc])[1] == []                           # measured in our setting: no longer a gap
    assert gaps.derived_claim({"activation": "swiglu", "dropout": "0.1"}, "promising") is None   # a combination adds no edge
    assert gaps.derived_claim({"dropout": "0.1"}, "no_improvement").expected_outcome == "no_worse"


# --------------------------------------------------------------------------- one cycle
def test_one_cycle_explores_on_one_seed_confirms_a_finalist_and_records_shrinkage(session_name, monkeypatch, capsys):
    s = explore_session(session_name, monkeypatch, overrides=OPEN)
    assert s.loop
    out = s.run()
    d = ROOT / "results" / session_name
    st = Store(d / "notebook.sqlite", readonly=True)
    decs = st.decisions()
    explored = [x for x in decs if x["kind"] == "exploration_result" and x["cycle"] == 1]
    assert out["status"] == "complete" and len(explored) >= 3
    assert all(x["label"] == "EXPLORATION — NOT A RESULT" and x["seed"] == 0 for x in explored)
    runs = {r["run_id"]: r for r in st.runs()}
    base0 = st.noise_floors()["noise_baseline"]["run_ids"][0]
    assert all(x["baseline_run_id"] == base0 and runs[x["candidate_run_id"]]["seed"] == 0 for x in explored)   # baseline seed 0 reused
    sel = next(x for x in decs if x["kind"] == "finalists_selected")
    assert 1 <= len(sel["finalists"]) <= 2 and set(sel["finalists"]) <= {x["key"] for x in explored}
    conf = [x for x in decs if x["kind"] == "confirmation"]
    assert len(conf) >= 1 and all({"single_seed_sd", "paired_sd", "shrinkage_sd"} <= set(x) for x in conf)
    analyses = {a["analysis_id"]: a for a in st.analyses()}
    assert all(len(analyses[x["analysis_id"]]["pairs"]) == 3 for x in conf)          # confirmation is the paired protocol
    assert all(x["shrinkage_sd"] == pytest.approx(x["single_seed_sd"] - x["paired_sd"]) for x in conf)
    hyps = {h["candidate_key"]: h for h in st.hypotheses() if h.get("kind") == "gap"}
    not_final = [k for k in hyps if k not in sel["finalists"]]
    assert not_final and all(hyps[k]["status"] == "explored" for k in not_final)     # explored only: no label, no branch
    assert all(not [a for a in analyses.values() if a["hypothesis_id"] == hyps[k]["hypothesis_id"]] for k in not_final)
    assert all(hyps[k]["status"].startswith("screened_") and hyps[k]["shrinkage"] for k in sel["finalists"])
    assert st.get_meta("rule_version") == 2 and st.get_meta("protocol")["search_loop"]
    st.close()
    assert rederive(d) == []
    assert main(["replay", "--session", session_name]) == 0 and "REPLAY OK" in capsys.readouterr().out


# --------------------------------------------------------------------------- promotion
def _promising_dropout(cfg):
    return ((-0.05 if cfg.dropout == 0.1 else 0.0) + (0.02 if cfg.pos_encoding == "rope" else 0.0)
            + (0.02 if cfg.activation == "swiglu" else 0.0))


def test_a_promising_finalist_is_promoted_and_later_deltas_are_on_the_incumbent(session_name, monkeypatch, capsys):
    s = explore_session(session_name, monkeypatch, overrides=OPEN, effect=_promising_dropout)
    orig_hash = s.baseline.config_hash()
    s.run()
    d = ROOT / "results" / session_name
    st = Store(d / "notebook.sqlite", readonly=True)
    decs = st.decisions()
    promo = [x for x in decs if x["kind"] == "promotion"]
    assert len(promo) == 1 and promo[0]["candidate_key"] == "dropout=0.1" and promo[0]["cycle"] == 1
    assert promo[0]["previous_incumbent_hash"] == orig_hash
    # the new noise floor is the candidate's own 5 runs (no extra compute); the original is untouched
    noise = st.noise_floors()
    runs = {r["run_id"]: r for r in st.runs()}
    inc = noise[promo[0]["noise_id"]]
    assert sorted(inc["seeds"]) == [0, 1, 2, 3, 4] and inc["run_ids"] == promo[0]["run_ids"]
    assert all(runs[i]["config"]["dropout"] == 0.1 for i in inc["run_ids"])
    assert all(runs[i]["config_hash"] == orig_hash for i in noise["noise_baseline"]["run_ids"])
    # the promoted finalist was confirmed on 3 seeds, then extra seeds, before promotion
    kinds = [x["kind"] for x in decs if x["hypothesis_id"] == "hyp_gap_dropout_0_1"]
    assert kinds.index("screening_result") < kinds.index("extra_seeds_result") < kinds.index("promotion")
    # cycle 2: candidates are deltas on the incumbent, compared with the incumbent's seed-0 run
    later = [x for x in decs if x["kind"] == "exploration_result" and x["cycle"] == 2]
    assert later, "a second cycle should have explored on the incumbent"
    for x in later:
        assert x["baseline_run_id"] in inc["run_ids"] and x["incumbent_config_hash"] == promo[0]["incumbent_config_hash"]
        assert runs[x["candidate_run_id"]]["config"]["dropout"] == 0.1          # config = incumbent + the new change
        assert "dropout" not in x["key"]
    rope = next(x for x in later if x["key"] == "pos_encoding=rope")
    assert rope["delta"] == pytest.approx(0.02)                                  # its own effect only, not −0.05 + 0.02
    # a finalist confirmed after the promotion keeps a headline number vs the ORIGINAL baseline
    head = [x for x in decs if x["kind"] == "headline_vs_original_baseline"]
    if head:
        a = next(a for a in st.analyses() if a["analysis_id"] == head[0]["analysis_id"])
        assert a["noise"]["run_ids"] == noise["noise_baseline"]["run_ids"]
        assert all(runs[p["baseline_run_id"]]["config_hash"] == orig_hash for p in a["pairs"])
    # write-back: the confirmed result is a derived edge in the next cycle's graph
    cyc = st.get_meta("gap_cycles")
    assert cyc[0]["derived"] == [] and "ours_dropout_0_1" in [c["claim_id"] for c in cyc[1]["derived"]]
    assert cyc[1]["incumbent_delta"] == {"dropout": "0.1"}
    st.close()
    assert rederive(d) == []
    assert main(["replay", "--session", session_name]) == 0 and "REPLAY OK" in capsys.readouterr().out


def test_no_promotion_without_a_promising_finalist_and_the_baseline_stays_the_incumbent(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch, overrides=OPEN)             # every effect is zero
    s.run()
    assert s.incumbent is s.baseline and s.promotions == 0 and s.noise is s.orig_noise


def test_the_run_budget_stops_the_loop_and_is_recorded(session_name, monkeypatch):
    s = explore_session(session_name, monkeypatch, overrides=OPEN)
    s.budget = {**s.budget, "max_runs": 10}                                     # 5 baseline + 3 fixture B + 2 explorations
    s.run()
    st = Store(ROOT / "results" / session_name / "notebook.sqlite", readonly=True)
    stops = [x for x in st.decisions() if x["kind"] == "budget_exhausted"]
    assert stops and stops[-1]["bound"] == "max_runs" and len(st.runs()) <= 10


# --------------------------------------------------------------------------- noise floor is an estimate / views
def test_sd_interval_is_exhaustive_deterministic_and_brackets_the_estimate():
    import statistics
    iv = stats.sd_interval(BASE)
    assert iv == stats.sd_interval(list(reversed(BASE)))                       # no randomness
    assert iv[0] < statistics.stdev(BASE) < iv[1] * 1.5 and iv[0] >= 0
    assert stats.sd_interval([1.0, 1.0, 1.0]) == (0.0, 0.0)
    assert stats.sd_interval([1.0, 2.0]) is None and stats.sd_interval(list(range(7))) is None


def test_loop_views_label_exploration_and_show_shrinkage_and_hits(session_name, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from noesis_lab import gap_views
    explore_session(session_name, monkeypatch, overrides=OPEN, effect=_promising_dropout).run()
    nb = ROOT / "results" / session_name / "notebook.sqlite"
    st = Store(nb, readonly=True)
    decs = st.decisions()
    ex = gap_views.exploration_rows(decs)
    assert ex and all(r["label"] == "EXPLORATION — NOT A RESULT" for r in ex) and any(r["finalist"] == "yes" for r in ex)
    sh = gap_views.shrinkage_rows(decs)
    assert any(r["promoted"] == "yes" and r["candidate"] == "dropout=0.1" for r in sh)
    res = gap_views.our_results(st.hypotheses(), {a["analysis_id"]: a for a in st.analyses()})
    assert {"hit"} <= {r["outcome"] for r in res}                              # dropout: predicted +, measured promising
    st.close()
    monkeypatch.setenv("NOESIS_NOTEBOOK", str(nb))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [h.value for h in at.subheader]
    assert "Explore cheaply, confirm rigorously" in heads and "How good were the gaps?" in heads
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) + " ".join(s_.value for s_ in at.success)
    assert "EXPLORATION — NOT A RESULT" in text and "Shrinkage" in text and "Promotion (cycle 1)" in text
    assert "exhaustive-bootstrap 95% interval" in text
