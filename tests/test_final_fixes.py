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
