"""FIXES2 P0: strict schema keeps a `title` field, deterministic gate v2, unstable gate stops a
recording, rule v2 (versioned, not retroactive), replay under the bundle's own config."""
import itertools

import pytest
from pydantic import BaseModel

from noesis_lab import stats
from noesis_lab.cli import main
from noesis_lab.config import ROOT, bundle_config, load_config, protocol
from noesis_lab.llm import strict_schema
from noesis_lab.rederive import rederive
from noesis_lab.schemas import (
    Claim,
    Direction,
    LiteratureOutputV2,
    PriorArtResult,
    PriorArtVerdictKind,
    ScoutOutput,
)

from .test_stats import BASE, setup


# --------------------------------------------------------------------------- P0-1
class _Book(BaseModel):
    title: str
    pages: int


class _Shelf(BaseModel):
    title: str
    books: list[_Book]


def test_strict_schema_keeps_a_field_named_title():
    s = strict_schema(_Shelf)
    assert "title" in s["properties"] and "title" in s["required"]
    book = s["$defs"]["_Book"]
    assert "title" in book["properties"] and set(book["required"]) == {"title", "pages"}
    assert "title" not in s and "title" not in book                      # Pydantic's metadata title is still stripped
    assert "title" not in s["properties"]["books"] and book["additionalProperties"] is False


def test_the_scout_schema_asks_for_its_title_field():
    s = strict_schema(ScoutOutput)
    d = s["$defs"]["Direction"]
    assert set(d["required"]) == set(Direction.model_fields) and "title" in d["properties"]


# --------------------------------------------------------------------------- P0-2
def _c(cid, tier="T3", coverage="none", outcome="improves", covers=False):
    return Claim(claim_id=cid, paper_id="p", dimension="d", method="m", setting="s", claim="c", source_span="x",
                 expected_outcome=outcome, tier=tier, coverage=None if tier == "T1" else coverage,
                 covers_our_setting=covers)


K = PriorArtVerdictKind
T1_COVER, T1_OTHER = _c("c03", "T1", covers=True), _c("c09", "T1", covers=False)
T3_COVER, T3_PART, T3_OTHER = _c("x2", "T3", "covers"), _c("x1", "T3", "partial"), _c("x9", "T3", "none")


@pytest.mark.parametrize("same,expected", [
    ([T3_OTHER, T3_COVER, T1_COVER, T1_OTHER], (K.known, "c03", "reject")),       # 1. any curated covering claim wins
    ([T3_OTHER, T3_COVER, T3_PART, T1_OTHER], (K.known, "x1", "soft_reject")),    # 2. else any auto covering claim; lowest id
    ([T3_OTHER, T1_OTHER], (K.setting_untested, "c09", "run")),                   # 3. else method known, setting untested
    ([], (K.not_found, None, "run")),                                             # 4. else not found
])
def test_gate_decision_strongest_first(same, expected):
    for order in itertools.permutations(same):                                   # the LLM's order never matters
        verdict, claim, gate, contested = stats.gate_decision(list(order))
        assert (verdict, claim.claim_id if claim else None, gate) == expected and not contested


def test_gate_decision_contested_and_override():
    pro, con = _c("x1", "T3", "partial", "improves"), _c("x2", "T3", "covers", "worse")
    verdict, claim, gate, contested = stats.gate_decision([pro, con], contested_ids=["x1", "x2"])
    assert (verdict, claim.claim_id, gate, contested) == (K.known, "x1", "run", True)        # a disagreement is not prior art
    assert stats.gate_decision([pro, con, T1_COVER], contested_ids=["x1", "x2"])[2] == "reject"   # a curated claim still is
    assert stats.gate_decision([T3_COVER], overridden=True)[2] == "run"                      # PI override of a soft-reject
    assert stats.gate_decision([T1_COVER], overridden=True)[2] == "reject"                   # never of a curated claim


def test_v1_prior_art_results_serialize_exactly_as_before():
    pa = PriorArtResult(verdict=K.not_found, claim_id=None, paper_id=None, paper_title=None, paper_url=None,
                        passage=None, rationale="r", snapshot_size=1, label="l", event_id="e")
    assert "same_comparison_claim_ids" not in pa.model_dump() and "contested" not in pa.model_dump()


# --------------------------------------------------------------------------- P0-3
def test_lit_dryrun_exits_nonzero_when_a_row_flips(monkeypatch, capsys):
    import noesis_lab.mock_llm as mock
    calls = itertools.count()

    def flipping(fixtures):
        def fn(role, system, user, schema):
            ids = ["c03"] if next(calls) % 2 == 0 and "[c03]" in user else []     # reject, then run, then reject …
            return LiteratureOutputV2(same_comparison_claim_ids=ids, rationale="mock")
        return fn

    monkeypatch.setattr(mock, "make_mock", flipping)
    assert main(["lit-dryrun", "--llm", "mock", "--repeat", "3"]) == 1
    out = capsys.readouterr().out
    assert "UNSTABLE" in out and "GATE UNSTABLE" in out and "Do not record" in out


def test_lit_dryrun_passes_when_every_row_is_stable(capsys):
    assert main(["lit-dryrun", "--llm", "mock", "--repeat", "3"]) == 0
    out = capsys.readouterr().out
    assert "UNSTABLE" not in out and "every row STABLE" in out


def test_make_record_runs_the_gate_check_before_the_session():
    recipe = (ROOT / "Makefile").read_text().split("record:")[1].split("\n\n")[0]
    steps = [ln.split("noesis_lab ")[1].split()[0] for ln in recipe.splitlines() if "noesis_lab " in ln]
    assert steps == ["search", "lit-dryrun", "session", "verify", "rederive", "replay"]
    assert "--repeat 3" in recipe and "ABORT" in recipe and "exit 1" in recipe


# --------------------------------------------------------------------------- P0-4
def _analyze(cand_vals, version):
    base, cand, noise = setup(BASE, cand_vals)
    return stats.analyze("h", cand, base[: len(cand)], noise, all_base_runs=base, rule_version=version)


def test_rule_v1_lets_one_seed_decide_harmful_and_v2_does_not():
    mixed = [1.49, 1.58, 1.53]            # mean worse by > 1 SD, but seed 0 is better
    v1, v2 = _analyze(mixed, 1), _analyze(mixed, 2)
    assert v1.branch == "harmful" and not v1.all_seeds_worse and not v1.seed_disagreement
    assert v2.branch == "no_improvement" and v2.seed_disagreement
    assert stats.next_action(v2, extra_seeds=[3, 4], queue=[], rule_version=2).action == "next_candidate"


def test_rule_v2_keeps_harmful_when_every_seed_is_worse_and_promising_is_unchanged():
    assert _analyze([1.60, 1.58, 1.57], 2).branch == "harmful"
    assert _analyze([1.45, 1.47, 1.43], 2).branch == _analyze([1.45, 1.47, 1.43], 1).branch == "promising"
    up = _analyze([1.40, 1.40, 1.50], 2)                                          # improvement with disagreement: as in v1
    assert up.branch == "no_improvement" and up.seed_disagreement
    assert _analyze([1.505, 1.515, 1.475], 2).branch == "no_improvement"


def test_rule_text_is_versioned():
    a = _analyze([1.60, 1.58, 1.57], 2)
    assert stats.next_action(a, extra_seeds=[], queue=[], rule_version=2).rule_text == stats.RULE_V2_TEXT
    assert stats.next_action(a, extra_seeds=[], queue=[]).rule_text == stats.RULE_TEXT
    assert "every paired seed is worse" in stats.RULE_V2_TEXT and "Rule version 2" in stats.RULE_V2_TEXT


@pytest.mark.parametrize("session", ["golden", "explore1"])
def test_recorded_bundles_are_still_judged_by_the_rule_they_were_recorded_under(session):
    d = ROOT / "results" / session
    if not d.exists():
        pytest.skip(f"{session} bundle not present")
    cfg = bundle_config(d)
    assert protocol(cfg) == {"rule_version": 1, "gate_version": 1, "search_loop": False}
    assert rederive(d) == []


# --------------------------------------------------------------------------- P0-5 / protocol
def test_current_protocol_and_engine_budget():
    cfg = load_config()
    assert protocol(cfg)["rule_version"] == 2 and protocol(cfg)["gate_version"] == 2
    assert cfg["budget"]["max_followup_candidates"] == 4 and cfg["budget"]["max_runs"] == 32


def test_replay_uses_the_bundles_config_not_todays(tmp_path):
    (tmp_path / "config.snapshot.yaml").write_text("budget: {max_runs: 7}\nrule: {promising_sd: 1.0}\n")
    assert bundle_config(tmp_path)["budget"]["max_runs"] == 7
    assert protocol(bundle_config(tmp_path))["rule_version"] == 1                  # no protocol block: version 1
    assert bundle_config(tmp_path / "missing") is load_config()


# --------------------------------------------------------------------------- scoring the engine's predictions
@pytest.mark.parametrize("pred,branch,outcome", [
    ("+", "promising", "hit"), ("+", "harmful", "miss"), ("+", "no_improvement", "null"),
    ("-", "harmful", "hit"), ("-", "promising", "miss"), ("-", "no_improvement", "null"),
    ("0", "no_improvement", "hit"), ("0", "promising", "hit"), ("0", "harmful", "miss"),
    ("", "promising", "no_prediction"), ("", "harmful", "no_prediction"), ("", "no_improvement", "no_prediction"),
])
def test_prediction_outcome_table(pred, branch, outcome):
    assert stats.prediction_outcome(pred, branch) == outcome


def test_prediction_counts_by_gap_type_and_label():
    from noesis_lab import gap_views

    def hyp(key, gtype, label, pred, aid):
        return {"kind": "gap", "candidate_key": key, "novelty_type": label, "latest_analysis_id": aid,
                "gap": {"gap_type": gtype, "predicted_direction": pred}}
    hyps = [hyp("a", "coverage", "transfer", "+", "1"), hyp("b", "coverage", "transfer", "+", "2"),
            hyp("c", "contradiction", "resolution", "", "3"), hyp("d", "abc", "transfer", "+", "4"),
            {"kind": "generated", "latest_analysis_id": "1"}, hyp("e", "coverage", "transfer", "+", "")]
    ana = {i: {"branch": b, "improvement_in_noise_sd": 1.0, "pairs": [1, 2, 3]}
           for i, b in (("1", "harmful"), ("2", "promising"), ("3", "promising"), ("4", "no_improvement"))}
    rows = gap_views.prediction_rows(hyps, ana)
    assert [r["outcome"] for r in rows] == ["miss", "hit", "no_prediction", "null"]      # unscreened / non-gap: skipped
    by_type = {r["gap type"]: r for r in gap_views.prediction_counts(rows, "gap type")}
    assert (by_type["transfer"]["hit"], by_type["transfer"]["miss"], by_type["ABC"]["null"]) == (1, 1, 1)
    by_label = {r["label"]: r for r in gap_views.prediction_counts(rows, "label")}
    assert by_label["transfer"] == {"label": "transfer", "hit": 1, "miss": 1, "null": 1, "no_prediction": 0}


def test_explore1_dropout_gap_is_reported_as_a_miss():
    from noesis_lab import gap_views
    from noesis_lab.store import Store
    nb = ROOT / "results" / "explore1" / "notebook.sqlite"
    if not nb.exists():
        pytest.skip("explore1 bundle not present")
    st = Store(nb, readonly=True)
    rows = gap_views.prediction_rows(st.hypotheses(), {a["analysis_id"]: a for a in st.analyses()})
    assert [(r["hypothesis"], r["predicted"], r["measured"], r["outcome"]) for r in rows] == [
        ("dropout=0.1", "+", "harmful", "miss")]
