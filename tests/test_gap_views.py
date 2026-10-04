"""Stage 5: dashboard views of the hypothesis engine (pure DOT/table builders + a render test)."""
from noesis_lab import gap_views, gaps
from noesis_lab.config import ROOT

from .test_explore import _toy, explore_session


def _data():
    cs = _toy()
    g, found = gaps.find_gaps(cs)
    graph = {**g.model_dump(), "communities": gaps.communities(g), "tensor": gaps.coverage_tensor(g)}
    claims = {c.claim_id: c.model_dump() for c in cs}
    return graph, [x.model_dump() for x in found], claims


def test_gap_rows_report_the_components_not_just_the_product():
    _, found, _ = _data()
    rows = gap_views.gap_rows(found)
    assert [r["rank"] for r in rows] == list(range(1, len(found) + 1))
    r = rows[0]
    assert {"gap score", "plausibility", "1 − coverage", "testability", "evidence quality"} <= set(r)
    assert r["gap score"] == found[0]["score"] and r["1 − coverage"] == round(1 - found[0]["coverage"], 4)
    assert {x["type"] for x in rows} <= {"transfer", "ABC", "combination", "resolution", "bridge"}


def test_gap_card_draws_the_path_with_verbatim_quotes_on_the_edges():
    _, found, claims = _data()
    abc = next(g for g in found if g["gap_type"] == "abc")
    claims[abc["claim_ids"][0]]["source_span"] = "a verbatim quote"
    dot = gap_views.gap_card_dot(abc, claims)
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")
    assert "a verbatim quote" in dot and all(cid in dot for cid in abc["claim_ids"])
    assert "style=dashed" in dot and "GAP:" in dot and f'score {abc["score"]:.2f}' in dot
    assert dot.count("->") == len(abc["path"]) + 1                      # every path step + the dashed gap edge


def test_overview_map_signs_results_and_gaps():
    graph, found, _ = _data()
    dot = gap_views.overview_dot(graph, found, [{"delta_key": "activation=swiglu", "branch": "promising", "sd": 1.5}])
    assert gap_views.SIGN_COLOR["+"] in dot and gap_views.SIGN_COLOR["-"] in dot       # signed claim edges
    assert "penwidth=4" in dot and "ours: promising (+1.50× SD)" in dot               # our result, thick
    assert "style=dashed" in dot and "gap " in dot                                     # gaps, dashed with score
    assert all(gap_views._id(m) in dot for m in graph["methods"])
    assert "shape=ellipse" in dot                                                      # mechanisms
    assert len({ln for ln in dot.splitlines() if "style=dashed" in ln}) <= 12


def _dir(i, status, outside=False, papers=("p1",)):
    return {"direction_id": f"dir_{i:02d}", "title": f"Direction {i}", "status": status, "outside_testbed": outside,
            "n_papers": len(papers), "n_claims": 1, "paper_ids": list(papers), "building_blocks": [] if outside else ["activation"],
            "claim_ids": []}


def test_direction_map_colors_by_status_and_connects_shared_papers():
    dirs = [_dir(1, "covered"), _dir(2, "open", papers=("p1", "p2")), _dir(3, "unexplored", outside=True, papers=())]
    dot = gap_views.direction_map_dot(dirs)
    assert gap_views.STATUS_STYLE["covered"][0] in dot and gap_views.STATUS_STYLE["open"][0] in dot
    assert 'style="dashed,filled"' in dot and "outside testbed" in dot
    assert dot.count("dir_01 -- n_p1") == 1 and dot.count("dir_02 -- n_p1") == 1       # shared paper links both
    assert dot.count("shape=circle") == 2


def test_direction_view_shows_quotes_blocks_hypotheses_and_results():
    _, _, claims = _data()
    claims["c1"]["source_span"] = "SwiGLU quote"
    d = {**_dir(1, "open", papers=("p_c1",)), "claim_ids": ["c1"]}
    papers = {"p_c1": {"title": "A paper", "tier": "T3", "published": "2024-01-01", "peer_reviewed": None, "source": "arxiv"}}
    hyps = [{"hypothesis_id": "hyp_x", "candidate_key": "activation=swiglu", "config_delta": {"activation": "swiglu"},
             "novelty_type": "transfer", "status": "screened_promising", "latest_analysis_id": "ana_1"},
            {"hypothesis_id": "hyp_other", "candidate_key": "norm=rmsnorm", "config_delta": {"norm": "rmsnorm"}, "status": "x"}]
    analyses = {"ana_1": {"analysis_id": "ana_1", "branch": "promising", "improvement_in_noise_sd": 2.0}}
    dot = gap_views.direction_view_dot(d, claims, papers, hyps, analyses)
    assert "SwiGLU quote" in dot and "review status unknown" in dot and "T3 · 2024-01-01" in dot
    assert "building block: activation" in dot and "[transfer]" in dot and "our result: promising" in dot
    assert "hyp_other" not in dot                                                      # other blocks' hypotheses are not shown


def test_our_results_are_keyed_by_delta():
    hyps = [{"latest_analysis_id": "a", "config_delta": {"pos_encoding": "rope", "activation": "swiglu"}},
            {"latest_analysis_id": "", "config_delta": {"norm": "rmsnorm"}},
            {"latest_analysis_id": "b", "config_delta": {"dropout": 0.1}, "kind": "gap", "gap": {"predicted_direction": "+"}}]
    out = gap_views.our_results(hyps, {"a": {"branch": "harmful", "improvement_in_noise_sd": -2.0, "pairs": [1, 2, 3]},
                                       "b": {"branch": "harmful", "improvement_in_noise_sd": -1.5, "pairs": [1, 2, 3]}})
    assert out == [{"delta_key": "activation=swiglu+pos_encoding=rope", "branch": "harmful", "sd": -2.0, "seeds": 3, "outcome": "",
                    "incumbent": ""},
                   {"delta_key": "dropout=0.1", "branch": "harmful", "sd": -1.5, "seeds": 3, "outcome": "miss", "incumbent": ""}]


def test_dashboard_renders_gap_list_card_and_maps(session_name, monkeypatch):
    from streamlit.testing.v1 import AppTest
    explore_session(session_name, monkeypatch).run()
    monkeypatch.setenv("NOESIS_NOTEBOOK", str(ROOT / "results" / session_name / "notebook.sqlite"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [h.value for h in at.subheader]
    assert all(h in heads for h in ("Gap list", "Gap card", "Overview map", "Direction map"))
    md = " ".join(m.value for m in at.markdown)
    assert "Tighten pass: PASSED" in md and "grounded in" in md
    assert "structural hint" in " ".join(c.value for c in at.caption) + md
    assert "novel" not in md.replace("“novel”", "")


def test_empty_gap_families_say_no_structural_support_found():
    graph, found, _ = _data()
    rows = {r["family"]: r for r in gap_views.gap_type_summary(found, {**graph, "shared_mechanisms": {}})}
    assert all(r["note"] == "" for r in rows.values() if r["gaps"])                       # a note only when empty
    only_coverage = [g for g in found if g["gap_type"] == "coverage"]
    rows = {r["family"]: r for r in gap_views.gap_type_summary(only_coverage, {**graph, "shared_mechanisms": {}})}
    combo = rows["Combination (link prediction + bridges)"]
    assert combo["gaps"] == 0 and combo["note"].startswith("No structural support found")
    assert "0 mechanism node(s) are linked to 2 or more runnable methods" in combo["note"]
    assert rows["Transfer (coverage gap)"]["gaps"] == len(only_coverage) and rows["Transfer (coverage gap)"]["note"] == ""
    assert rows["Transfer by mechanism (ABC closure)"]["note"].startswith("No structural support found")


def test_gap_rows_show_the_predicted_direction():
    _, found, _ = _data()
    rows = gap_views.gap_rows(found)
    assert {r["predicts"] for r in rows} <= {"+", "-", "0", "—"} and any(r["predicts"] == "+" for r in rows)
