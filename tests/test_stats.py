import pytest

from noesis_lab import stats
from noesis_lab.schemas import Claim, ExperimentConfig

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


def test_rule_selects_action_without_any_llm():
    kw = dict(extra_seeds=[3, 4], candidate_order=["x", "y", "z"], already_considered=["x"])
    assert stats.next_action(run(BASE, [1.45, 1.47, 1.43]), **kw).action == "extra_seeds"
    assert stats.next_action(run(BASE, [1.60, 1.58, 1.57]), **kw).action == "literature_check"
    n = stats.next_action(run(BASE, [1.505, 1.515, 1.475]), **kw)
    assert n.action == "next_candidate" and n.detail["fixture_id"] == "y"      # skips 'x'


def test_single_seed_counterfactual_exposes_coin_flip():
    # Candidate is ~ noise: some pairings beat the baseline run, some don't.
    a = run(BASE, [1.505, 1.515, 1.475])
    cf = a.counterfactual
    assert cf.n_pairings == 15 and cf.decision_depends_on_seed and 0 < cf.n_keep < 15
    assert "seed luck" in cf.summary


def test_counterfactual_no_dependence_when_effect_is_large():
    cf = run(BASE, [1.30, 1.31, 1.29]).counterfactual
    assert not cf.decision_depends_on_seed and cf.n_keep == 15


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


@pytest.mark.parametrize("cand,expected_outcome,relation", [
    ([1.45, 1.47, 1.43], "improves", "agrees"),
    ([1.60, 1.58, 1.57], "improves", "contradicts"),
    ([1.505, 1.515, 1.475], "improves", "inconclusive"),
    ([1.505, 1.515, 1.475], "no_worse", "agrees"),
    ([1.60, 1.58, 1.57], "no_worse", "contradicts"),
    ([1.45, 1.47, 1.43], "context", "inconclusive"),
])
def test_derived_evidence_relation(cand, expected_outcome, relation):
    claim = Claim(claim_id="c", paper_id="p", dimension="d", method="m", setting="s", claim="c",
                  source_span="x", expected_outcome=expected_outcome)
    assert stats.relate_to_claim(run(BASE, cand), claim).relation == relation
