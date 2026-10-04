"""FINAL_FIXES: a bundle holds only what its manifest lists, and the polish / speed changes keep
old bundles replaying unchanged."""
import json

from noesis_lab import dashboard
from noesis_lab.bundle import verify_manifest, write_manifest
from noesis_lab.config import ROOT
from noesis_lab.store import Store


# --------------------------------------------------------------------------- 0. unexpected files
def _bundle(tmp_path):
    d = tmp_path / "b"
    (d / "corpus" / "raw").mkdir(parents=True)
    (d / "state.json").write_text("{}\n")
    (d / "corpus" / "raw" / "001_arxiv.xml").write_text("<feed/>")
    write_manifest(d)
    return d


def test_verify_passes_on_a_clean_bundle(tmp_path):
    assert verify_manifest(_bundle(tmp_path)) == []


def test_verify_fails_on_files_the_manifest_does_not_list(tmp_path):
    d = _bundle(tmp_path)
    (d / "corpus" / "raw" / "001_arxiv 2.xml").write_text("<feed/>")      # an iCloud conflict copy
    (d / "state 2.json").write_text("{}\n")
    problems = verify_manifest(d)
    assert problems == ["corpus/raw/001_arxiv 2.xml: not in MANIFEST.json (unexpected file)",
                        "state 2.json: not in MANIFEST.json (unexpected file)"]


def test_verify_ignores_finder_metadata_but_not_the_manifest_contents(tmp_path):
    d = _bundle(tmp_path)
    (d / ".DS_Store").write_bytes(b"\0")
    assert verify_manifest(d) == []
    files = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert "MANIFEST.json" not in files and set(files) == {"state.json", "corpus/raw/001_arxiv.xml"}


# --------------------------------------------------------------------------- A1-A4 presentation
EXPLORE2 = ROOT / "results" / "explore2" / "notebook.sqlite"


def test_dashboard_opens_on_explore2_then_golden_and_labels_the_rest():
    keys = ["results/explore1/notebook.sqlite", "results/golden/notebook.sqlite", "results/smoke/notebook.sqlite",
            "work/replay-explore2/notebook.sqlite", "results/explore2/notebook.sqlite", "results/mps-mock/notebook.sqlite"]
    order = [dashboard.session_name(k) for k in dashboard.order_notebooks(keys)]
    assert order[:3] == ["explore2", "replay-explore2", "golden"]
    assert "superseded (scout schema bug, rule v1)" in dashboard.notebook_label(keys[0])
    for k in ("results/smoke/notebook.sqlite", "results/mps-mock/notebook.sqlite"):
        assert "pipeline checks: not results" in dashboard.notebook_label(k)


def test_a_sealed_bundle_states_its_limitation_instead_of_asking_to_fix_it():
    reason = "thin mechanism graph: only 2 node(s) (want >= 3); gaps will be sparse. Fix before recording."
    assert dashboard.degraded_note(reason, sealed=False) == f"Search degraded: {reason}"
    sealed = dashboard.degraded_note(reason, sealed=True)
    assert sealed.startswith("Recorded with this limitation: thin mechanism graph")
    assert "Fix before recording" not in sealed and "Search degraded" not in sealed


def _explore2():
    s = Store(EXPLORE2, readonly=True)
    d = dict(hyps=s.hypotheses(), analyses=s.analyses(), decisions=s.decisions(), evidence=s.derived_evidence(),
             claims={c["claim_id"]: c for c in s.claims()}, baseline=s.get_meta("baseline_config"))
    s.close()
    return d


def test_a_change_on_an_incumbent_is_credited_with_its_own_effect_only():
    d = _explore2()
    ana = {a["analysis_id"]: a for a in d["analyses"]}
    h = next(x for x in d["hyps"] if x["hypothesis_id"] == "hyp_gap_activation_relu2_norm_rmsnorm_i1")
    cr = dashboard.credit(h, ana, d["baseline"])
    assert cr["text"] == ("activation=relu2 + norm=rmsnorm on top of pos_encoding=rope: no improvement vs incumbent "
                          "(-0.67 SD). vs original +1.46 SD, which includes pos_encoding=rope.")
    assert dashboard.analysis_role(h, h["headline_analysis_id"], d["baseline"]).startswith("vs original baseline (cumulative")
    assert dashboard.analysis_role(h, h["latest_analysis_id"], d["baseline"]).startswith("vs incumbent pos_encoding=rope")
    rope = next(x for x in d["hyps"] if x["hypothesis_id"] == "hyp_gap_pos_encoding_rope")   # tested on the baseline
    assert dashboard.credit(rope, ana, d["baseline"]) is None
    assert dashboard.analysis_role(rope, rope["latest_analysis_id"], d["baseline"]) == ""


def test_tables_never_show_the_cumulative_gain_as_the_changes_result():
    from noesis_lab.evidence_chain import evidence_grid
    from noesis_lab.gap_views import our_results, prediction_rows
    d = _explore2()
    ana = {a["analysis_id"]: a for a in d["analyses"]}
    row = next(r for r in evidence_grid(d) if r["candidate"].startswith("activation=relu2 + norm=rmsnorm")
               and "incumbent" in r["candidate"])
    assert row["measured"].startswith("no_improvement (-0.67") and row["measured"].endswith("vs incumbent")
    assert row["vs original baseline (cumulative, includes the incumbent)"].startswith("promising (+1.46")
    res = {r["delta_key"]: r for r in our_results(d["hyps"], ana, d["baseline"])}
    assert res["activation=relu2+norm=rmsnorm"]["branch"] == "no_improvement"
    assert res["activation=relu2+norm=rmsnorm"]["incumbent"] == "pos_encoding=rope"
    assert res["pos_encoding=rope"]["incumbent"] == ""
    pr = {r["hypothesis"]: r for r in prediction_rows(d["hyps"], ana, d["baseline"])}
    assert pr["activation=relu2+norm=rmsnorm"]["measured vs"] == "incumbent pos_encoding=rope"


def test_lion_hypotheses_carry_the_scaling_caveat():
    d = _explore2()
    lion = [h for h in d["hyps"] if dashboard.is_lion(h)]
    assert lion and all("optimizer=lion" in h["candidate_key"] for h in lion)
    assert "not a test of Lion" in dashboard.LION_CAVEAT


def test_dashboard_renders_the_explore2_exhibit():
    import os

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    os.environ["NOESIS_NOTEBOOK"] = str(EXPLORE2)
    try:
        at.run()
    finally:
        os.environ.pop("NOESIS_NOTEBOOK", None)
    assert not at.exception, [e.value for e in at.exception]
    md = [m.value for m in at.markdown]
    assert any("no improvement vs incumbent (-0.67 SD). vs original +1.46 SD, which includes pos_encoding=rope" in m
               for m in md)
    assert any(i.value.startswith("Recorded with this limitation") for i in at.info)
    assert not any("Fix before recording" in w.value for w in at.warning)
    assert any("not a test of Lion" in c.value for c in at.caption)


# --------------------------------------------------------------------------- A5 combination predictions
from noesis_lab import gaps  # noqa: E402

from .test_gaps import DROP, MECH, RMS, ROPE, SWIGLU, claim  # noqa: E402


def _combo_graph():
    return [claim("a", SWIGLU, "improves", "T3", "none", mech=MECH),
            claim("b", ROPE, "improves", "T3", "partial", mech=MECH),
            claim("c", RMS, "worse", "T3", "none", mech=MECH),
            claim("d", DROP, "no_worse", "T3", "none", mech=MECH)]


def test_combination_sign_is_the_sum_of_the_components_literature_signs():
    g = gaps.build_graph(_combo_graph())
    cd = gaps.combination_direction
    assert cd(g, ["activation=swiglu", "pos_encoding=rope"]) == "+"            # both + -> +
    assert cd(g, ["activation=swiglu", "norm=rmsnorm"]) == ""                  # mixed -> no prediction
    assert cd(g, ["activation=swiglu", "dropout=0.1"]) == "+"                  # + and 0 -> +
    assert cd(g, ["dropout=0.1", "norm=rmsnorm"]) == "-"
    assert cd(g, ["activation=swiglu", "optimizer=lion"]) == ""                # no claim -> no prediction


def test_our_own_results_never_set_a_components_sign():
    ours = gaps.derived_claim({"norm": "rmsnorm"}, "promising", tier="D-confirmed")
    g = gaps.build_graph([*_combo_graph(), ours, ours.model_copy(update={"claim_id": ours.claim_id + "b"})])
    assert gaps.literature_sign(g, "norm=rmsnorm") == "-"


def test_combination_predictions_only_under_prediction_version_2():
    cs = _combo_graph()
    old = [x for x in gaps.find_gaps(cs)[1] if x.novelty_type == "combination"]
    new = {x.gap_id: x for x in gaps.find_gaps(cs, predict_combinations=True)[1] if x.novelty_type == "combination"}
    assert old and all(x.predicted_direction == "" for x in old)              # how explore2 was recorded
    assert {x.gap_id for x in old} == set(new)                                # same gaps, same scores and order
    assert all(x.score == new[x.gap_id].score for x in old)
    assert new["link:activation=swiglu+pos_encoding=rope"].predicted_direction == "+"
    assert gaps.COMBINATION_RULE_TEXT.count("→") >= 3


def test_config_turns_on_prediction_version_2_and_old_bundles_stay_on_1():
    from noesis_lab.config import bundle_config, load_config, protocol
    assert protocol(load_config())["prediction_version"] == 2
    for s in ("golden", "explore1", "explore2"):
        assert protocol(bundle_config(ROOT / "results" / s))["prediction_version"] == 1


# --------------------------------------------------------------------------- A6 timestamps
def test_events_and_decisions_carry_a_timestamp_outside_the_digest(tmp_path):
    a, b = Store(tmp_path / "a.sqlite"), Store(tmp_path / "b.sqlite")
    for s in (a, b):
        s.add_event("llm_call", {"role": "x"})
        s.add_decision("noise_floor", "", {"sd": 0.01})
    b.db.execute("UPDATE decisions SET payload=json_set(payload, '$.ts', '1999-01-01T00:00:00.000+00:00')")
    assert a.events()[0]["ts"] and a.decisions()[0]["ts"].endswith("+00:00")
    assert a.state_digest() == b.state_digest()                               # wall time never enters the digest
    b.db.execute("UPDATE decisions SET payload=json_set(payload, '$.sd', 0.02)")
    assert a.state_digest() != b.state_digest()                               # content still does


def test_rows_recorded_before_timestamps_keep_their_digest(tmp_path):
    s = Store(tmp_path / "s.sqlite")
    s.add_decision("noise_floor", "", {"sd": 0.01})
    with_ts = s.state_digest()
    p = s.decisions()[0]
    old = json.dumps({k: v for k, v in p.items() if k not in ("ts", "kind", "hypothesis_id")}, sort_keys=True)
    s.db.execute("UPDATE decisions SET payload=?", (old,))                    # as Store wrote rows before A6
    assert s.state_digest() == with_ts


def test_timeline_flags_a_long_silence():
    ev = [{"event_id": "evt_0001", "role": "critic_pre", "ts": "2026-10-03T18:00:00.000+00:00"}]
    dec = [{"decision_id": "dec_001", "kind": "confirmation", "ts": "2026-10-03T18:00:30.000+00:00"},
           {"decision_id": "dec_002", "kind": "next_action", "ts": "2026-10-03T18:57:30.000+00:00"}]
    runs = [{"run_id": "r1", "seed": 0, "train_seconds": 36.0, "started_at": "2026-10-03T18:00:10+00:00"}]
    rows = dashboard.timeline_rows(ev, dec, runs)
    assert [r["kind"] for r in rows] == ["LLM call", "run", "decision", "decision"]
    assert rows[-1]["gap before (s)"] == 3420.0 and rows[-1]["stall"] and not any(r["stall"] for r in rows[:-1])
