"""Stage 1: scout → targeted search per direction → provenance (no network, mocked LLM)."""
import json

import pytest

from noesis_lab.literature import Snapshot
from noesis_lab.llm import LLM
from noesis_lab.mock_llm import make_mock
from noesis_lab.schemas import Claim, Direction, DirectionRecord, ScoutOutput
from noesis_lab.search.corpus import build_corpus, load_niche
from noesis_lab.search.http import RawCache
from noesis_lab.search.scout import (
    BLOCKS,
    direction_queries,
    direction_status,
    link_claims,
    run_scout,
    search_directions,
    title_matches,
    to_records,
)

from .test_search import CFG, fake_fetch, paper


def _dir(title="t", blocks=(), terms=("x",), titles=()):
    return Direction(title=title, idea="i", mechanism="m", search_terms=list(terms),
                     possibly_related_titles=list(titles), building_blocks=list(blocks))


def test_records_get_code_ids_and_one_deterministic_direction_per_unmentioned_block():
    recs = to_records(ScoutOutput(directions=[_dir("a", ["activation", "made_up_block"]), _dir("b", [])]))
    assert [r.direction_id for r in recs[:2]] == ["dir_01", "dir_02"]
    assert recs[0].building_blocks == ["activation"] and recs[0].dropped_blocks == ["made_up_block"]
    assert recs[1].building_blocks == []                                   # outside the testbed, kept for the map
    det = [r for r in recs if r.origin == "deterministic"]
    assert sorted(b for r in det for b in r.building_blocks) == sorted(set(BLOCKS) - {"activation"})
    assert all(r.search_terms for r in det)
    assert len(to_records(None)) == len(BLOCKS)                            # scout failed: the map still covers the testbed


def test_title_hint_must_be_nearly_the_exact_title():
    assert title_matches("Root Mean Square Layer Normalization", "Root mean square layer normalization.")
    assert not title_matches("Layer Normalization", "Root Mean Square Layer Normalization")
    assert not title_matches("", "x")


def test_direction_queries_are_quoted_deduped_and_capped():
    r = DirectionRecord(direction_id="d", origin="scout", title="t", search_terms=["SwiGLU", "swiglu", 'a "b"', "c", "d"])
    assert direction_queries(r, 3) == ['"SwiGLU"', '"a b"', '"c"']


def test_targeted_search_records_provenance_and_remembered_titles_are_only_searches(tmp_path):
    recs = to_records(ScoutOutput(directions=[
        _dir("gated", ["activation"], ["SwiGLU"], ["SwiGLU feed-forward blocks for character-level Transformers",
                                                  "A SwiGLU paper that does not exist"]),
        _dir("norm", ["norm"], ["RMSNorm"])]))[:2]
    found = search_directions(RawCache(tmp_path, fake_fetch, 0.0), load_niche(), recs,
                              date_to="2026-10-03", n_queries=3, cap=2)
    a, b = recs
    assert a.queries == [{"q": '"SwiGLU"', "n_results": 4}]              # per-query count: how hard we looked
    assert [h["status"] for h in a.title_hints] == ["found", "not_found"]
    assert a.title_hints[0]["paper_id"] == "2402.00002" and a.title_hints[1]["paper_id"] is None
    assert len(a.paper_ids) <= 2 and len(b.paper_ids) <= 2                # cap per direction
    assert all(p.directions in (["dir_01"], ["dir_02"]) for p in found)
    ids = {p.paper_id for p in found}
    assert "A SwiGLU paper that does not exist" not in json.dumps(sorted(ids))   # never becomes a paper


def test_directions_tied_to_building_blocks_are_searched_harder(tmp_path):
    recs = to_records(ScoutOutput(directions=[_dir("gated", ["activation"], ["RMSNorm"]), _dir("sparse", [], ["RMSNorm"])]))[:2]
    search_directions(RawCache(tmp_path, fake_fetch, 0.0), load_niche(), recs, date_to="2026-10-03",
                      n_queries=1, cap=1, block_cap=3)
    assert len(recs[0].paper_ids) == 3 and len(recs[1].paper_ids) == 1       # runnable: 3 papers; outside testbed: 1


def _claim(cid, pid, change=None, method="m", text="c", outcome="improves", tier="T3", coverage="none"):
    return Claim(claim_id=cid, paper_id=pid, dimension="d", method=method, setting="s", claim=text,
                 source_span="x", expected_outcome=outcome, config_change=change, tier=tier, coverage=coverage)


def test_claim_direction_linking_every_branch():
    recs = [DirectionRecord(direction_id="d1", origin="scout", title="Gated feed-forward SwiGLU blocks",
                            search_terms=["SwiGLU", "gated linear unit"], building_blocks=["activation"]),
            DirectionRecord(direction_id="d2", origin="scout", title="Sparse attention patterns",
                            search_terms=["sparse attention"], building_blocks=[])]
    papers = {"p1": paper("p1", directions=["d1"]), "p2": paper("p2", directions=["d2"]),
              "p3": paper("p3"), "p12": paper("p12", directions=["d1", "d2"])}
    claims = [
        _claim("block", "p1", {"activation": "swiglu"}),                                   # found_by + block
        _claim("sim", "p1", None, "SwiGLU gated linear unit", "gated feed-forward SwiGLU blocks help"),   # found_by + text
        _claim("neither", "p1", None, "weight decay", "decoupled weight decay helps"),     # found_by, no block, no text
        _claim("elsewhere", "p2", {"activation": "swiglu"}),                               # scouted, but by another direction
        _claim("niche_block", "p3", {"activation": "relu2"}),                              # niche/curated paper + block
        _claim("niche_text", "p3", None, "SwiGLU gated linear unit", "gated feed-forward SwiGLU blocks help"),
        _claim("both", "p12", None, "sparse attention", "sparse attention patterns reduce cost"),
    ]
    links = link_claims(recs, claims, papers, 0.25)
    assert links["d1"] == ["block", "niche_block", "sim"]
    assert links["d2"] == ["both"]


@pytest.mark.parametrize("claims,blocks,status", [
    ([], ["norm"], "unexplored"),
    ([("T1", "covers", "improves")], ["norm"], "covered"),
    ([("T2", "covers", "worse")], ["norm"], "covered"),
    ([("T3", "covers", "improves")], ["norm"], "partial"),                    # only a preprint covers it
    ([("T2", "partial", "improves")], ["norm"], "partial"),
    ([("T3", "partial", "improves"), ("T3", "none", "worse")], ["norm"], "contested"),
    ([("T3", "none", "improves"), ("T3", "none", "no_worse")], ["norm"], "contested"),
    ([("T3", "none", "improves")], ["norm"], "open"),
    ([("T1", "covers", "context")], ["norm"], "open"),                        # a context claim reports no result
    ([("T3", "none", "improves")], [], "open"),
])
def test_direction_gap_status_rules(claims, blocks, status):
    rec = DirectionRecord(direction_id="d", origin="scout", title="t", building_blocks=blocks, paper_ids=["p"])
    linked = [_claim(f"c{i}", "p", None, tier=t, coverage=cov, outcome=o) for i, (t, cov, o) in enumerate(claims)]
    out = direction_status(rec, linked)
    assert out["status"] == status and out["outside_testbed"] == (not blocks) and out["n_claims"] == len(claims)


def test_scout_failure_degrades_to_deterministic_directions():
    def broken(role, system, user, schema):
        raise RuntimeError("no LLM")

    recs, err = run_scout(LLM("mock", CFG["llm"], mock_fn=broken), load_niche())
    assert err and len(recs) == len(BLOCKS) and all(r.origin == "deterministic" for r in recs)


def test_corpus_freezes_scout_directions_and_paper_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    out = tmp_path / "corpus"
    llm = LLM("mock", CFG["llm"], recordings_path=out / "llm.jsonl", mock_fn=make_mock({}))
    meta = build_corpus(out, load_niche(), llm, today="2026-10-03", fetch=fake_fetch, sleep=lambda s: None)
    snap = Snapshot.from_corpus(out)
    assert meta["explore"]["enabled"] and meta["explore"]["title_hints_found"] == 1
    assert meta["explore"]["title_hints_not_found"] == 1
    assert len(snap.directions) == meta["explore"]["directions"] >= 8
    gated = snap.directions[0]
    assert gated.paper_ids and all(gated.direction_id in snap.papers[p].directions for p in gated.paper_ids)
    assert snap.papers["2402.00002"].directions                           # found by a direction search too
    assert any(not d.building_blocks for d in snap.directions)            # outside-testbed directions are kept
    # graph health check right after the search: mechanism nodes shared by 2+ runnable methods
    assert meta["explore"]["shared_mechanisms"] == {"loss_landscape_smoothness": ["activation=swiglu", "dropout=0.1"]}
    assert meta["explore"]["claims_with_mechanism"] == 3
    assert any("thin mechanism graph" in d for d in meta["degraded"])     # 1 < min_shared_mechanisms: flagged
    scout = json.loads((out / "scout.json").read_text())
    assert scout["directions"][3]["dropped_blocks"] == ["attention_sparsity"]      # not a building block
