"""Gap engine: pure functions on hand-built toy graphs with known answers."""
import pytest

from noesis_lab import gaps
from noesis_lab.schemas import Claim, SettingFields


def claim(cid, change=None, outcome="improves", tier="T3", coverage="none", mech="", paper=None,
          fields=None, covers=False):
    sf = SettingFields(**fields, evidence="empirical") if fields else None
    return Claim(claim_id=cid, paper_id=paper or f"p_{cid}", dimension="d", method="m", setting="s", claim="c",
                 source_span="x", expected_outcome=outcome, config_change=change, tier=tier,
                 coverage=None if tier == "T1" else coverage, covers_our_setting=covers, mechanism_category=mech,
                 mechanism=f"words quoted for {mech}" if mech else "",
                 setting_fields=sf)


MECH = "loss_landscape_smoothness"
SWIGLU, ROPE, RMS, DROP = {"activation": "swiglu"}, {"pos_encoding": "rope"}, {"norm": "rmsnorm"}, {"dropout": "0.1"}


# --------------------------------------------------------------------------- mechanisms
def test_mechanism_nodes_are_the_fixed_categories_backed_by_quotes():
    cs = [claim("a", SWIGLU, mech="gradient_stability"), claim("b", ROPE, mech="gradient_stability"),
          claim("c", RMS, mech="position_generalization"), claim("d", DROP),
          claim("e", DROP, mech="not_a_listed_mechanism")]
    nodes = gaps.mechanism_nodes(cs)
    assert sorted(nodes) == ["gradient_stability", "position_generalization"]        # unknown names never become nodes
    assert nodes["gradient_stability"]["members"] == ["words quoted for gradient_stability"] * 2
    no_quote = claim("f", SWIGLU, mech="expressivity")
    no_quote.mechanism = ""
    assert gaps.mechanism_nodes([no_quote]) == {}                                       # a category needs its quote


def test_shared_mechanisms_counts_nodes_linked_to_two_or_more_runnable_methods():
    cs = [claim("a", SWIGLU, mech="gradient_stability"), claim("b", ROPE, mech="gradient_stability"),
          claim("c", RMS, mech="position_generalization"), claim("d", None, mech="position_generalization"),
          claim("e", SWIGLU, mech="expressivity"), claim("f", SWIGLU, mech="expressivity")]
    shared = gaps.shared_mechanisms(gaps.build_graph(cs))
    assert shared == {"gradient_stability": ["activation=swiglu", "pos_encoding=rope"]}
    # an unmapped method (None) and the same method twice do not make a node "shared"


# --------------------------------------------------------------------------- graph + tensor
def test_setting_similarity_uses_the_fixed_table():
    ours = claim("a", fields=dict(model_family="transformer", task="char_language_modeling", scale="tiny"))
    far = claim("b", fields=dict(model_family="cnn", task="vision", scale="large"))
    assert gaps.setting_similarity(ours) == 1.0
    assert gaps.setting_similarity(far) == pytest.approx(0.3 * 0.2 * 0.4)
    assert gaps.setting_similarity(claim("c", tier="T1", covers=True)) == 1.0       # curated: human flag
    assert gaps.setting_similarity(claim("d", tier="T1", covers=False)) == 0.4


def test_graph_edges_and_tier_weighted_tensor():
    cs = [claim("c1", SWIGLU, "improves", "T1", covers=True),
          claim("c2", SWIGLU, "improves", "T3", "partial"),
          claim("c3", SWIGLU, "worse", "T2", "none", mech=MECH),
          claim("c4", ROPE, "context", "T3", "covers"),                      # context: no signed edge
          claim("c5", None, "improves", "T3", "covers", mech=MECH)]
    g = gaps.build_graph(cs)
    assert g.methods == ["activation=swiglu", "pos_encoding=rope"]
    by = {e.claim_id: e for e in g.edges}
    assert by["c3"].mechanism_id == by["c5"].mechanism_id == MECH and by["c1"].mechanism_id is None
    assert (by["c4"].sign, by["c5"].method, by["c3"].weight) == (None, None, 0.8)
    t = gaps.coverage_tensor(g)
    assert t["activation=swiglu"] == {"covers": {"+": 1.0, "0": 0.0, "-": 0.0},
                                      "partial": {"+": 0.6, "0": 0.0, "-": 0.0},
                                      "none": {"+": 0.0, "0": 0.0, "-": 0.8}}
    assert sum(sum(c.values()) for c in t["pos_encoding=rope"].values()) == 0
    assert gaps.coverage_in_our_setting(t, "activation=swiglu") == 1.0           # 1.0 + 0.5*0.6, capped
    assert gaps.coverage_in_our_setting(t, "pos_encoding=rope") == 0.0
    assert gaps.coverage_in_our_setting(t, "norm=rmsnorm") == 0.0                # no claims at all


# --------------------------------------------------------------------------- gap formulas (toy graphs)
OURS = dict(model_family="transformer", task="char_language_modeling", scale="tiny")       # similarity 1.0
LM_LARGE = dict(model_family="transformer", task="language_modeling", scale="large")       # 0.9 * 0.4 = 0.36


def test_gap_score_is_the_product_of_its_four_components():
    assert gaps.gap_score(0.5, 0.2, 1.0, 0.8) == pytest.approx(0.5 * 0.8 * 1.0 * 0.8)
    assert gaps.gap_score(0.9, 1.0, 1.0, 1.0) == 0.0                 # fully covered: nothing to test
    assert gaps.gap_score(0.9, 0.0, 0.0, 1.0) == 0.0                 # outside the testbed: never queued


def test_coverage_gap_missing_cell_with_transfer_plausibility():
    cs = [claim("a", ROPE, "improves", "T3", "none", fields=LM_LARGE),        # w 0.6, sim 0.36
          claim("b", ROPE, "improves", "T2", "none", fields=LM_LARGE),        # w 0.8, sim 0.36
          claim("c", ROPE, "worse", "T3", "none", fields=OURS | {"task": "translation"}),
          claim("d", SWIGLU, "improves", "T1", covers=True)]                  # covered: not a gap
    g = gaps.build_graph(cs)
    found = gaps.coverage_gaps(g, gaps.coverage_tensor(g))
    assert [x.delta_key for x in found] == ["pos_encoding=rope"]
    x = found[0]
    assert x.novelty_type == "transfer" and x.detail["expected_sign"] == "+"
    assert x.plausibility == pytest.approx((1.4 / 2.0) * 0.36)                # same-sign share × similarity
    assert x.coverage == 0.0 and x.testability == 1.0
    assert x.evidence_quality == pytest.approx(0.7) and x.claim_ids == ["a", "b"]
    assert x.score == pytest.approx(x.plausibility * 0.7)


def test_partial_coverage_lowers_the_score_but_keeps_the_gap():
    cs = [claim("a", ROPE, "improves", "T3", "partial", fields=LM_LARGE)]
    g = gaps.build_graph(cs)
    x = gaps.coverage_gaps(g, gaps.coverage_tensor(g))[0]
    assert x.coverage == pytest.approx(0.3) and x.score == pytest.approx(1.0 * 0.36 * 0.7 * 0.6)


def test_abc_triangle_is_found_with_its_path():
    cs = [claim("c1", SWIGLU, "improves", "T3", "none", mech=MECH, fields=LM_LARGE),          # A → B
          claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS)]   # B → C
    g = gaps.build_graph(cs)
    found = gaps.abc_gaps(g, gaps.coverage_tensor(g))
    assert len(found) == 1
    x = found[0]
    assert (x.gap_type, x.novelty_type, x.delta_key) == ("abc", "transfer", "activation=swiglu")
    assert x.plausibility == pytest.approx(0.6 * 0.8)                         # support(m→k) × support(k→o*)
    label = x.detail["mechanism"]                                             # both claims sit on one fixed node
    assert label == "loss landscape smoothness" and x.detail["mechanism_id"] == MECH and len(g.mechanisms) == 1
    assert [(p["from"], p["claim_id"]) for p in x.path] == [("activation=swiglu", "c1"), (f"mechanism: {label}", "c2")]
    assert x.evidence_quality == pytest.approx(0.7) and x.score == pytest.approx(0.48 * 0.7)


def test_abc_is_not_a_gap_when_the_direct_claim_exists_or_the_mechanism_has_no_other_support():
    direct = [claim("c1", SWIGLU, "improves", "T3", "none", mech=MECH),
              claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS),
              claim("c3", SWIGLU, "improves", "T3", "covers", fields=OURS)]          # A → C already reported
    g = gaps.build_graph(direct)
    assert gaps.abc_gaps(g, gaps.coverage_tensor(g)) == []
    alone = [claim("c1", SWIGLU, "improves", "T3", "none", mech=MECH)]                # only its own claim
    g = gaps.build_graph(alone)
    assert gaps.abc_gaps(g, gaps.coverage_tensor(g)) == []


def test_contradiction_scores_one_when_evenly_split_in_our_setting():
    cs = [claim("p", SWIGLU, "improves", "T3", "covers", fields=OURS),
          claim("n", SWIGLU, "worse", "T3", "covers", fields=OURS)]
    g = gaps.build_graph(cs)
    x = gaps.contradiction_gaps(g, gaps.coverage_tensor(g))[0]
    assert x.plausibility == 1.0 and x.novelty_type == "resolution" and x.detail["balance"] == 1.0
    assert x.coverage == 0.0                                    # contested evidence is not coverage
    uneven = cs + [claim("p2", SWIGLU, "improves", "T3", "covers", fields=OURS)]
    g = gaps.build_graph(uneven)
    assert gaps.contradiction_gaps(g, gaps.coverage_tensor(g))[0].plausibility == pytest.approx(0.5)
    g = gaps.build_graph(cs[:1] + [claim("z", SWIGLU, "no_worse", "T3", "covers", fields=OURS)])
    assert gaps.contradiction_gaps(g, gaps.coverage_tensor(g)) == []     # "no worse" is not the opposite sign


def test_link_prediction_scores_on_a_toy_graph():
    import math
    cs = [claim("a", SWIGLU, "improves", "T3", "none", mech=MECH),
          claim("b", ROPE, "improves", "T3", "partial", mech=MECH),
          claim("c", RMS, "context", "T3", "none", mech=MECH),                  # context: no setting neighbor
          claim("d", DROP, "improves", "T3", "none", mech="implicit_regularization"),
          claim("e", {"activation": "relu2"}, "improves", "T3", "none", mech=MECH)]
    g = gaps.build_graph(cs)
    s = gaps.link_scores(g)
    assert ("activation=relu2", "activation=swiglu") not in s                    # same field: not combinable
    sr = s[("activation=swiglu", "pos_encoding=rope")]
    assert sr["common_neighbors"] == 1 and sr["shared"][0].startswith("K:")
    assert sr["adamic_adar"] == pytest.approx(1 / math.log(4)) and sr["resource_allocation"] == pytest.approx(0.25)
    sd = s[("activation=swiglu", "dropout=0.1")]                                 # share only the "none" bucket
    assert sd["common_neighbors"] == 1 and sd["shared"] == ["S:none"]
    found = {x.delta_key: x for x in gaps.link_gaps(g)}
    assert "activation=swiglu+dropout=0.1" not in found                          # a setting bucket alone is no hint
    x = found["activation=swiglu+pos_encoding=rope"]
    assert x.novelty_type == "combination" and x.delta == {"activation": "swiglu", "pos_encoding": "rope"}
    assert x.detail["complementarity"] == 0.5                                    # shared mechanism: likely redundant
    assert x.plausibility == pytest.approx(sr["adamic_adar"] / max(v["adamic_adar"] for v in s.values()) * 0.5)
    assert sorted(x.claim_ids) == ["a", "b"] and x.coverage == 0.0


def test_no_combination_hint_for_methods_already_tested_together():
    cs = [claim("a", SWIGLU, mech=MECH, paper="same"), claim("b", ROPE, mech=MECH, paper="same")]
    assert gaps.link_gaps(gaps.build_graph(cs)) == [] and gaps.bridge_gaps(gaps.build_graph(cs)) == []


def test_structural_hole_bridges_two_communities():
    m = ["loss_landscape_smoothness", "gradient_stability", "position_generalization", "implicit_regularization"]
    cs = [claim("a1", SWIGLU, mech=m[0]), claim("a2", SWIGLU, mech=m[1]),
          claim("b1", RMS, mech=m[0]), claim("b2", RMS, mech=m[1]),
          claim("c1", ROPE, mech=m[2]), claim("c2", ROPE, mech=m[3]),
          claim("d1", DROP, mech=m[2]), claim("d2", DROP, mech=m[3]),
          claim("x", RMS, mech=m[2])]                                             # the one cross-community edge
    g = gaps.build_graph(cs)
    comm = gaps.communities(g)
    assert comm["M:activation=swiglu"] == comm["M:norm=rmsnorm"] != comm["M:pos_encoding=rope"] == comm["M:dropout=0.1"]
    found = {x.delta_key: x for x in gaps.bridge_gaps(g)}
    assert sorted(found) == ["dropout=0.1+norm=rmsnorm", "norm=rmsnorm+pos_encoding=rope"]
    x = found["norm=rmsnorm+pos_encoding=rope"]
    assert x.detail["cross_edge_fraction"] == pytest.approx(1 / 9) and x.detail["path_support"] == 0.6
    assert x.plausibility == pytest.approx((1 - 1 / 9) * 0.6) and x.novelty_type == "combination"
    assert [p["claim_id"] for p in x.path] == ["x", "c1"]


def test_ranking_ties_break_by_novelty_type_then_alphabetically():
    def mk(gid, novelty, key, score):
        return gaps.Gap(gap_id=gid, gap_type="coverage", novelty_type=novelty, delta_key=key, delta={},
                        plausibility=score, coverage=0, testability=1, evidence_quality=1, score=score,
                        path=[], claim_ids=[])
    ranked = gaps.rank_gaps([mk("t", "transfer", "a=1", 0.5), mk("c", "combination", "z=1", 0.5),
                             mk("r", "resolution", "z=2", 0.5), mk("c2", "combination", "b=1", 0.5),
                             mk("top", "transfer", "z=9", 0.9)])
    assert [x.gap_id for x in ranked] == ["top", "r", "c2", "c", "t"]


def test_find_gaps_is_deterministic_and_every_gap_explains_itself():
    cs = [claim("c1", SWIGLU, "improves", "T3", "none", mech=MECH, fields=LM_LARGE),
          claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS),
          claim("c3", ROPE, "improves", "T3", "partial", mech=MECH, fields=LM_LARGE),
          claim("c4", ROPE, "worse", "T3", "partial", fields=LM_LARGE)]
    g1, a = gaps.find_gaps(cs)
    g2, b = gaps.find_gaps(list(reversed(cs)))
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b] and g1 == g2
    assert {x.gap_type for x in a} >= {"coverage", "abc", "link", "contradiction"}
    known = {c.claim_id for c in cs}
    assert all(x.path and set(x.claim_ids) <= known and 0 <= x.score <= 1 for x in a)
    assert [x.score for x in a] == sorted((x.score for x in a), reverse=True)


# --------------------------------------------------------------------------- ABC respects signs
def test_a_method_that_is_worse_via_a_mechanism_gets_no_positive_abc_gap():
    cs = [claim("c1", SWIGLU, "worse", "T3", "none", mech=MECH, fields=LM_LARGE),            # worse via k
          claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS)]             # k → better quality
    g = gaps.build_graph(cs)
    assert gaps.abc_gaps(g, gaps.coverage_tensor(g)) == []
    ctx = [claim("c1", SWIGLU, "context", "T3", "none", mech=MECH), cs[1]]                   # no reported benefit either
    g = gaps.build_graph(ctx)
    assert gaps.abc_gaps(g, gaps.coverage_tensor(g)) == []


def test_abc_counts_only_benefit_edges_and_stores_the_predicted_direction():
    cs = [claim("good", SWIGLU, "improves", "T3", "none", mech=MECH, fields=LM_LARGE),
          claim("ok", SWIGLU, "no_worse", "T2", "none", mech=MECH, fields=LM_LARGE),
          claim("bad", SWIGLU, "worse", "T1", mech=MECH),                                    # must not add support
          claim("c2", None, "improves", "T2", "covers", mech=MECH, fields=OURS)]
    g = gaps.build_graph(cs)
    x = gaps.abc_gaps(g, gaps.coverage_tensor(g))[0]
    assert x.predicted_direction == "+" and "bad" not in x.claim_ids and {"good", "ok", "c2"} == set(x.claim_ids)
    assert x.detail["support_method_mechanism"] == 1.0                                       # 0.6 + 0.8, capped; T1 "bad" excluded


def test_predicted_direction_per_gap_type():
    cs = [claim("a", ROPE, "worse", "T3", "none", fields=LM_LARGE),                          # transfer of a harm
          claim("p", SWIGLU, "improves", "T3", "partial", mech=MECH, fields=LM_LARGE),
          claim("n", SWIGLU, "worse", "T3", "partial", fields=LM_LARGE),
          claim("r", RMS, "improves", "T3", "none", mech=MECH, fields=LM_LARGE)]
    _, found = gaps.find_gaps(cs)
    by = {x.gap_id.split(":")[0] + ":" + x.delta_key: x.predicted_direction for x in found}
    assert by["coverage:pos_encoding=rope"] == "-" and by["coverage:norm=rmsnorm"] == "+"
    assert by["contradiction:activation=swiglu"] == ""                                       # a disagreement predicts nothing
    assert by["link:activation=swiglu+norm=rmsnorm"] == ""                                   # nor does a combination hint
