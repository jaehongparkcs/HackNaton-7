import pytest

from noesis_lab import stats
from noesis_lab.config import path_of
from noesis_lab.literature import Snapshot
from noesis_lab.schemas import Claim, ExperimentConfig, PriorArtVerdictKind, QueueItem

from .conftest import make_run

CAND = ExperimentConfig(norm="rmsnorm")


def setup(base_vals, cand_vals, cand_tokens=1000):
    base = [make_run(v, s) for s, v in enumerate(base_vals)]
    cand = [make_run(v, s, tokens=cand_tokens, cfg=CAND) for s, v in enumerate(cand_vals)]
    noise = stats.noise_floor(base)
    return base, cand, noise


def run(base_vals, cand_vals, **kw):
    base, cand, noise = setup(base_vals, cand_vals, **kw)
    return stats.analyze("h", cand, base[: len(cand)], noise, all_base_runs=base)


BASE = [1.50, 1.52, 1.48, 1.51, 1.49]       # SD ~ 0.0158


def test_noise_floor_is_sample_sd():
    n = stats.noise_floor([make_run(v, s) for s, v in enumerate(BASE)])
    assert n.n == 5 and n.sd == pytest.approx(0.0158114, rel=1e-4)


def test_promising_needs_gt_1sd_and_all_seeds_agree():
    a = run(BASE, [1.45, 1.47, 1.43])
    assert a.branch == "promising" and a.all_seeds_improve and a.improvement_in_noise_sd > 1
    assert a.label == "SCREENING RESULT — NOT CONFIRMED"


def test_improvement_with_seed_disagreement_is_not_promising():
    a = run(BASE, [1.40, 1.40, 1.50])          # mean improvement > 1 SD but seed 2 is worse
    assert a.branch == "no_improvement" and a.seed_disagreement


def test_within_noise_is_no_improvement():
    a = run(BASE, [1.505, 1.515, 1.475])
    assert a.branch == "no_improvement" and not a.seed_disagreement


def test_harmful():
    a = run(BASE, [1.60, 1.58, 1.57])
    assert a.branch == "harmful" and a.all_seeds_worse


def _q(*keys):
    return [QueueItem(key=k, field=k.split("=")[0], value=k.split("=")[1], priority=[0, -1, k],
                      supporting_claim_ids=["c"]) for k in keys]


def test_rule_selects_action_without_any_llm():
    kw = dict(extra_seeds=[3, 4], queue=_q("a=1", "b=2"))
    assert stats.next_action(run(BASE, [1.45, 1.47, 1.43]), **kw).action == "extra_seeds"
    assert stats.next_action(run(BASE, [1.60, 1.58, 1.57]), **kw).action == "literature_check"
    n = stats.next_action(run(BASE, [1.505, 1.515, 1.475]), **kw)
    assert n.action == "next_candidate" and n.detail["candidate"] == "a=1"      # head of the queue
    n = stats.next_action(run(BASE, [1.505, 1.515, 1.475]), extra_seeds=[3, 4], queue=[])
    assert n.detail["candidate"] is None


def test_counterfactual_headline_is_same_seed_and_cross_seed_is_secondary():
    # same-seed: seeds 0,1 keep, seed 2 revert; cross-seed pairings are mixed
    cf = run(BASE, [1.505, 1.515, 1.475]).counterfactual
    assert cf.n_pairings == 15 and cf.decision_depends_on_seed
    assert cf.same_seed_summary.startswith("A single-run keep/revert loop with a fixed seed")
    assert cf.summary.index("fixed seed") < cf.summary.index("If the seed varies")


# Baseline val_loss 1.50..1.54 (5 seeds). A candidate at value v wins #{baselines > v} pairings.
CF_BASE = [make_run(1.50 + 0.01 * i, i) for i in range(5)]
WINS = {5: 1.40, 4: 1.505, 3: 1.515, 2: 1.525, 1: 1.535, 0: 1.60}


def _cf(wins: tuple[int, int, int]):
    cands = [make_run(WINS[w], s, cfg=CAND) for s, w in enumerate(wins)]
    return stats.counterfactual(cands, CF_BASE)


def test_counterfactual_wording_bands():
    # never: every pairing agrees
    cf = _cf((5, 5, 5))
    assert cf.wording == "never" and cf.n_keep == 15
    assert "seed never changed the decision" in cf.cross_seed_summary
    assert _cf((0, 0, 0)).wording == "never"
    # almost always: at most 2 pairings flip, either way
    cf = _cf((5, 5, 4))
    assert cf.n_keep == 14 and cf.wording == "almost_always"
    assert "almost always KEEP" in cf.cross_seed_summary and "(1 of 15 pairings flip)" in cf.cross_seed_summary
    assert _cf((5, 4, 4)).n_keep == 13 and _cf((5, 4, 4)).wording == "almost_always"
    cf = _cf((1, 1, 0))
    assert cf.n_keep == 2 and cf.wording == "almost_always" and "almost always REVERT" in cf.cross_seed_summary
    # luck: otherwise
    cf = _cf((3, 3, 1))
    assert cf.n_keep == 7 and cf.wording == "luck"
    assert "decided by seed luck: KEEP in 7 of 15" in cf.cross_seed_summary
    assert _cf((5, 4, 3)).n_keep == 12 and _cf((5, 4, 3)).wording == "luck"      # 3 flips


def test_counterfactual_no_dependence_when_effect_is_large():
    cf = run(BASE, [1.30, 1.31, 1.29]).counterfactual
    assert not cf.decision_depends_on_seed and cf.n_keep == 15 and cf.wording == "never"


def test_equal_token_delta_separates_throughput_from_learning():
    # Candidate sees half the tokens; at equal tokens it is identical to baseline.
    base = [make_run(1.50, s, tokens=1000) for s in range(5)]
    cand = [make_run(1.50, s, tokens=500, cfg=CAND) for s in range(3)]
    a = stats.analyze("h", cand, base[:3], stats.noise_floor(base))
    assert a.token_ratio == 0.5
    # candidate curve at 500 tokens is 4.0 + (1.5-4.0)*0.5 on both -> equal-token delta uses min tokens
    assert a.pairs[0].equal_token_delta is not None


def test_analysis_is_exactly_reproducible():
    a1, a2 = run(BASE, [1.45, 1.47, 1.43]), run(BASE, [1.45, 1.47, 1.43])
    assert a1.model_dump() == a2.model_dump()


def _claim(outcome, covers, speed=False):
    return Claim(claim_id="c", paper_id="p", dimension="d", method="m", setting="s", claim="c",
                 source_span="x", expected_outcome=outcome, covers_our_setting=covers, claims_speed=speed)


PROMISING, NOISE, HARMFUL = [1.45, 1.47, 1.43], [1.505, 1.515, 1.475], [1.60, 1.58, 1.57]


@pytest.mark.parametrize("cand,outcome,covers,relation", [
    (PROMISING, "improves", True, "agrees"),
    (HARMFUL, "improves", True, "contradicts"),
    (NOISE, "improves", True, "inconclusive"),
    (NOISE, "no_worse", True, "agrees"),
    (HARMFUL, "no_worse", True, "contradicts"),
    (PROMISING, "context", True, "inconclusive"),
    # the claim does not cover our setting: never agrees / contradicts
    (PROMISING, "improves", False, "consistent_in_our_setting"),
    (HARMFUL, "improves", False, "not_reproduced_in_our_setting"),
    (NOISE, "improves", False, "inconclusive"),
    (NOISE, "no_worse", False, "consistent_in_our_setting"),
    (HARMFUL, "no_worse", False, "not_reproduced_in_our_setting"),
    (HARMFUL, "context", False, "inconclusive"),
])
def test_derived_evidence_relation(cand, outcome, covers, relation):
    assert stats.relate_to_claim(run(BASE, cand), _claim(outcome, covers)).relation == relation


def _slow_candidate_analysis(seconds: float):
    base = [make_run(v, s) for s, v in enumerate(BASE)]
    for r in base:
        r.train_seconds = 10.0
    cand = [make_run(v, s, cfg=CAND) for s, v in enumerate(NOISE)]
    for r in cand:
        r.train_seconds = seconds
    return stats.analyze("h", cand, base[:3], stats.noise_floor(base), all_base_runs=base)


def test_speed_claim_not_reproduced_note():
    slow, same = _slow_candidate_analysis(11.0), _slow_candidate_analysis(10.2)
    assert slow.throughput_ratio < 0.95 <= same.throughput_ratio
    assert "partial: speed claim not reproduced" in stats.relate_to_claim(slow, _claim("no_worse", False, True)).note
    assert "partial" not in stats.relate_to_claim(same, _claim("no_worse", False, True)).note
    assert "partial" not in stats.relate_to_claim(slow, _claim("no_worse", False, False)).note   # no speed claim


@pytest.mark.parametrize("same,covers,verdict", [
    (False, False, PriorArtVerdictKind.not_found), (False, True, PriorArtVerdictKind.not_found),
    (True, False, PriorArtVerdictKind.setting_untested), (True, None, PriorArtVerdictKind.setting_untested),
    (True, True, PriorArtVerdictKind.known),
    (True, "covers", PriorArtVerdictKind.known), (True, "partial", PriorArtVerdictKind.known),   # lean "covered"
    (True, "none", PriorArtVerdictKind.setting_untested), (False, "covers", PriorArtVerdictKind.not_found),
])
def test_prior_art_verdict_is_a_pure_function_of_two_inputs(same, covers, verdict):
    assert stats.prior_art_verdict(same, covers) == verdict


# --------------------------------------------------------------------------- candidate queue
SNAP = Snapshot(path_of("papers"), path_of("claims"))


def test_candidate_queue_is_deterministic_and_ordered_as_specified():
    base = ExperimentConfig()
    q = stats.candidate_queue(base, SNAP.claims.values(), [])
    assert [(i.key, i.status) for i in q] == [
        ("activation=swiglu", "open"), ("dropout=0.1", "open"), ("pos_encoding=rope", "open"),   # improves
        ("norm=rmsnorm", "open"),                                                                  # no_worse
        ("norm_position=post", "open"), ("optimizer=sgd_momentum", "open"),                        # context
        ("schedule=constant", "open"),
        ("activation=relu2", "rejected_prior_art")]       # covered by curated T1 claim c10: never run
    assert q[-1].covering_claim_ids == ["c10"]
    assert q == stats.candidate_queue(base, list(reversed(list(SNAP.claims.values()))), [])
    rms = next(i for i in q if i.key == "norm=rmsnorm")
    assert rms.supporting_claim_ids == ["c04", "c05"] and rms.priority == [1, -2, "norm=rmsnorm"]
    assert all(i.field not in ("lr", "batch_size") for i in q)


def test_candidate_queue_excludes_tested_and_baseline_values():
    base = ExperimentConfig()
    q = stats.candidate_queue(base, SNAP.claims.values(), ["norm=rmsnorm", "activation=swiglu"])
    assert "norm=rmsnorm" not in [i.key for i in q] and "activation=swiglu" not in [i.key for i in q]
    assert q[0].key == "dropout=0.1"
    # a candidate equal to the baseline value is not a change
    q2 = stats.candidate_queue(ExperimentConfig(activation="swiglu"), SNAP.claims.values(), [])
    assert "activation=swiglu" not in [i.key for i in q2]


def test_more_supporting_claims_rank_first_within_an_outcome_class():
    def mk(cid, field, value, outcome):
        return Claim(claim_id=cid, paper_id="p", dimension="d", method="m", setting="s", claim="c",
                     source_span="x", expected_outcome=outcome, config_change={field: value})
    claims = [mk("c1", "activation", "swiglu", "improves"), mk("c2", "dropout", "0.1", "improves"),
              mk("c3", "dropout", "0.1", "context")]
    q = stats.candidate_queue(ExperimentConfig(), claims, [])
    assert [i.key for i in q] == ["dropout=0.1", "activation=swiglu"]       # 2 supporting beats 1


# --------------------------------------------------------------------------- experiment language
def test_delta_keys_roundtrip_and_are_canonical():
    d = {"pos_encoding": "rope", "activation": "swiglu"}
    assert stats.delta_key(d) == "activation=swiglu+pos_encoding=rope"
    assert stats.parse_delta_key(stats.delta_key(d)) == d and stats.delta_key({"norm": "rmsnorm"}) == "norm=rmsnorm"


def test_combination_validator():
    assert stats.valid_delta({"norm": "rmsnorm"}) and stats.valid_delta({"norm": "rmsnorm", "activation": "swiglu"})
    assert not stats.valid_delta({}) and not stats.valid_delta({"norm": "batchnorm"})            # not allowlisted
    assert not stats.valid_delta({"norm": "rmsnorm", "lr": "0.01"})                              # pair outside the allowlist
    assert not stats.valid_delta({"norm": "rmsnorm", "activation": "swiglu", "dropout": "0.1"})  # three fields
    assert stats.combine("activation=swiglu", "pos_encoding=rope") == {"activation": "swiglu", "pos_encoding": "rope"}
    assert stats.combine("activation=swiglu", "activation=relu2") is None                        # same field
    assert stats.combine("activation=swiglu", "lr=0.01") is None


def test_testability_levels():
    assert stats.testability({"norm": "rmsnorm", "activation": "swiglu"}) == 1.0
    assert stats.testability({"warmup_frac": "0.02"}) == 1.0 and stats.testability({"norm": "rmsnorm", "qk_norm": "true"}) == 1.0
    assert stats.testability({"cosine_final_frac": "0.1"}) == 0.3            # a variant that is planned, not built
    assert stats.testability({"warmup_frac": "0.03"}) == 0.0                 # a value outside the allowlist
    assert stats.testability({"new_attention": "x"}) == 0.0 and stats.testability({}) == 0.0


# --------------------------------------------------------------------------- contested topics
def _lc(cid, outcome, coverage, tier="T3", covers=False):
    return Claim(claim_id=cid, paper_id="p", dimension="d", method="m", setting="s", claim="c", source_span="x",
                 expected_outcome=outcome, config_change={"activation": "swiglu"}, tier=tier,
                 coverage=None if tier == "T1" else coverage, covers_our_setting=covers)


KEY = "activation=swiglu"


def test_disagreeing_claims_do_not_cover_a_topic_in_the_hypothesis_engine():
    cs = [_lc("p", "improves", "partial"), _lc("n", "worse", "partial")]
    assert stats.contested_claim_ids(cs) == ["n", "p"]
    status, cov, note = stats.literature_status(KEY, cs, contested_open=True)
    assert (status, cov) == ("open", []) and note.startswith("open (contested)")
    # covers and partial are one bucket for the overlap question
    mixed = [_lc("p", "improves", "covers"), _lc("n", "worse", "partial")]
    assert stats.literature_status(KEY, mixed, contested_open=True)[0] == "open"
    # the literature-ordered queue of earlier bundles keeps the old behavior
    assert stats.literature_status(KEY, cs)[0] == "soft_rejected"


def test_agreeing_or_curated_claims_still_cover():
    agree = [_lc("a", "improves", "partial"), _lc("b", "improves", "covers")]
    assert stats.contested_claim_ids(agree) == []
    assert stats.literature_status(KEY, agree, contested_open=True)[0] == "soft_rejected"
    weak = [_lc("a", "improves", "partial"), _lc("b", "no_worse", "partial")]          # "no worse" is not the opposite sign
    assert stats.literature_status(KEY, weak, contested_open=True)[0] == "soft_rejected"
    elsewhere = [_lc("a", "improves", "partial"), _lc("b", "worse", "none")]           # the disagreement is in another setting
    assert stats.literature_status(KEY, elsewhere, contested_open=True) == (
        "soft_rejected", ["a"], "covered by an auto-extracted / preprint claim; the PI may override")
    curated = [_lc("t1", "improves", None, tier="T1", covers=True), _lc("n", "worse", "partial"), _lc("p", "improves", "partial")]
    status, cov, _ = stats.literature_status(KEY, curated, contested_open=True)
    assert (status, cov) == ("rejected_prior_art", ["t1"])                              # a curated claim is never contested away
    held = stats.literature_status(KEY, [_lc("p", "improves", "partial"), _lc("n", "worse", "partial")],
                                   held=[KEY], contested_open=True)
    assert held[0] == "held"
