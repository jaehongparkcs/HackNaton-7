"""Live literature search, tested with NO network: search clients run against recorded raw
responses under tests/data, and the LLM is the scripted mock."""
import json
import shutil
from pathlib import Path

import pytest

from noesis_lab import stats
from noesis_lab.bundle import verify_manifest
from noesis_lab.cli import main
from noesis_lab.config import ROOT, load_config
from noesis_lab.literature import LiteratureAgent, Snapshot
from noesis_lab.llm import LLM
from noesis_lab.mock_llm import make_mock
from noesis_lab.orchestrator import Session, SessionOpts
from noesis_lab.rederive import rederive
from noesis_lab.schemas import (
    Claim,
    ExperimentConfig,
    ExtractedClaim,
    Fixture,
    LiteratureOutputV2,
    NicheSpec,
    PriorArtVerdictKind,
    RetrievedPaper,
    SettingFields,
)
from noesis_lab.search import arxiv, openalex
from noesis_lab.search.corpus import CURATED_ONLY, build_corpus, load_niche
from noesis_lab.search.coverage import candidate_text, coverage_of, t4_overlaps, tier_of
from noesis_lab.search.extract import to_claim
from noesis_lab.search.http import RawCache
from noesis_lab.search.plan import deterministic_queries, plan_queries
from noesis_lab.search.rank import dedupe, filter_excluded, rank
from noesis_lab.store import Store

from .test_session import FakeRunner

DATA = Path(__file__).parent / "data"
CFG = load_config()


def fake_fetch(url: str) -> str:
    """Recorded responses only. Any other URL is a test failure, never a live call."""
    if "openalex.org" in url and "search=" in url:      # no recorded OpenAlex search: tighten falls back to arXiv
        raise OSError("no recorded OpenAlex search response")
    if "openalex.org" in url:
        return (DATA / "openalex.json").read_text()
    assert "export.arxiv.org" in url, url
    if "id_list=" in url:
        return (DATA / "arxiv_seeds.xml").read_text()
    if "RMSNorm" in url:
        return (DATA / "arxiv_q1.xml").read_text()
    if "SwiGLU" in url:
        return (DATA / "arxiv_q2.xml").read_text()
    return (DATA / "arxiv_empty.xml").read_text()


def paper(pid, title="t", abstract="a", **kw):
    return RetrievedPaper(paper_id=pid, title=title, abstract=abstract, url="u", abstract_sha256="h",
                          source=kw.pop("source", "arxiv"), **kw)


NICHE = NicheSpec(title="Tiny Transformers", description="normalization and activation in language models",
                  include_terms=["RMSNorm"], exclude_terms=["vision transformer"], max_papers=3,
                  date_from="2020-01-01")


# ------------------------------------------------------------------------------- coverage rule
@pytest.mark.parametrize("family,task,scale,expected", [
    ("transformer", "char_language_modeling", "tiny", "covers"),
    ("transformer", "language_modeling", "small", "covers"),
    ("general", "general", "tiny", "covers"),
    ("transformer", "language_modeling", "medium", "partial"),
    ("transformer", "language_modeling", "large", "partial"),
    ("general", "language_modeling", "unspecified", "partial"),
    ("transformer", "translation", "tiny", "none"),
    ("transformer", "classification", "small", "none"),
    ("transformer", "unspecified", "tiny", "none"),
    ("rnn", "language_modeling", "tiny", "none"),
    ("unspecified", "language_modeling", "small", "none"),
    ("cnn", "vision", "large", "none"),
])
def test_coverage_rule_every_row(family, task, scale, expected):
    s = SettingFields(model_family=family, task=task, scale=scale, evidence="empirical")
    assert coverage_of(s) == expected


def test_tiers():
    assert tier_of(curated=True, quote_verified=True, peer_reviewed=None) == "T1"
    assert tier_of(curated=False, quote_verified=True, peer_reviewed=True) == "T2"
    assert tier_of(curated=False, quote_verified=True, peer_reviewed=False) == "T3"
    assert tier_of(curated=False, quote_verified=True, peer_reviewed=None) == "T3"    # unknown = preprint
    assert tier_of(curated=False, quote_verified=False, peer_reviewed=True) == "T4"


@pytest.mark.parametrize("tier,verdict,override,outcome", [
    ("T1", PriorArtVerdictKind.known, False, "reject"),
    ("T1", PriorArtVerdictKind.known, True, "reject"),            # the PI cannot override a curated claim here
    ("T2", PriorArtVerdictKind.known, False, "soft_reject"),
    ("T3", PriorArtVerdictKind.known, False, "soft_reject"),
    ("T3", PriorArtVerdictKind.known, True, "run"),
    ("T3", PriorArtVerdictKind.setting_untested, False, "run"),
    (None, PriorArtVerdictKind.not_found, False, "run"),
])
def test_gate_outcome_per_tier(tier, verdict, override, outcome):
    assert stats.gate_outcome(verdict, tier, override) == outcome


# ------------------------------------------------------------------------------- rank / dedupe / filter
def test_dedupe_by_id_then_doi_then_title_and_logs_reasons():
    ps = [paper("1", "Alpha Beta", queries=["q1"]), paper("1", "Alpha Beta", queries=["q2"]),
          paper("2", "Other", doi="10.1/X"), paper("3", "Another", doi="10.1/x"),
          paper("4", "alpha:  BETA!")]
    kept, dropped = dedupe(ps)
    assert [p.paper_id for p in kept] == ["1", "2"] and kept[0].queries == ["q1", "q2"]
    assert {d["paper_id"]: d["reason"] for d in dropped} == {
        "3": "duplicate of 2 by doi", "4": "duplicate of 1 by title"}


def test_exclude_terms_is_a_hard_filter_with_a_logged_reason():
    ps = [paper("1", "RMSNorm in a Vision Transformer"), paper("2", "RMSNorm for language models"),
          paper("3", "curated vision transformer paper", source="curated")]
    kept, dropped = filter_excluded(ps, NICHE)
    assert [p.paper_id for p in kept] == ["2", "3"]              # curated papers are never dropped
    assert dropped == [{"paper_id": "1", "title": "RMSNorm in a Vision Transformer",
                        "reason": "exclude_term: vision transformer"}]


def test_rank_stores_components_keeps_top_n_and_breaks_ties_by_id():
    ps = [paper("b", "RMSNorm language models normalization", published="2024-01-01", cited_by=10),
          paper("a", "RMSNorm language models normalization", published="2024-01-01", cited_by=10),
          paper("c", "unrelated cooking recipes", published="2021-01-01"),
          paper("d", "normalization", published="2020-01-01"),
          paper("cur", "curated", source="curated", published="2019-01-01")]
    kept, dropped = rank(ps, NICHE, date_to="2026-01-01")
    ids = [p.paper_id for p in kept]
    assert ids.index("a") < ids.index("b")                     # equal relevance: arXiv id decides
    assert "cur" in ids and len(kept) == 3                     # curated is pinned; max_papers = 3
    assert {d["paper_id"] for d in dropped} == {"c", "d"} and all("max_papers" in d["reason"] for d in dropped)
    a = next(p for p in kept if p.paper_id == "a")
    assert set(a.relevance_components) == {"text", "citations", "recency"}
    assert a.relevance == pytest.approx(0.6 * a.relevance_components["text"] + 0.2 * a.relevance_components["citations"]
                                        + 0.2 * a.relevance_components["recency"], abs=1e-6)
    again, _ = rank([p.model_copy() for p in reversed(ps)], NICHE, date_to="2026-01-01")
    assert [p.paper_id for p in again] == ids                  # deterministic


# ------------------------------------------------------------------------------- clients (recorded responses)
def test_arxiv_client_parses_recorded_feed_and_builds_filtered_query():
    ps = arxiv.parse_feed((DATA / "arxiv_q1.xml").read_text(), "q")
    assert [p.paper_id for p in ps] == ["2401.00001", "2402.00002", "2403.00003", "2404.00004"]   # version stripped
    p = ps[0]
    assert p.abstract.startswith("We study rotary") and not p.abstract.endswith("\n")
    assert p.venue == "Journal of Tests 1 (2024)" and p.peer_reviewed is True and ps[1].peer_reviewed is None
    q = arxiv.to_arxiv_query('"RMSNorm" AND "language model"', load_niche(), "2026-10-03")
    assert q.startswith('all:"RMSNorm" AND all:"language model" AND (cat:cs.LG OR cat:cs.CL OR cat:cs.NE)')
    assert q.endswith("submittedDate:[201701010000 TO 202610032359]")


def test_openalex_only_adds_evidence_of_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_MAILTO", "team@example.org")
    assert "mailto=team%40example.org" in openalex.batch_url(["2402.00002"])
    ps = [paper("2402.00002"), paper("2401.00001"), paper("9999.00009")]
    n = openalex.enrich(ps, RawCache(tmp_path, fake_fetch, 0.0))
    assert n == 2
    assert (ps[0].peer_reviewed, ps[0].venue, ps[0].cited_by) == (True, "Proc. of Tests", 40)
    assert ps[1].peer_reviewed is None and ps[1].cited_by == 7     # repository-only: status stays unknown
    assert ps[2].peer_reviewed is None


def test_raw_cache_retries_once_then_records_the_failure(tmp_path):
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return "<ok/>"

    c = RawCache(tmp_path, flaky, 0.0)
    assert c.get("arxiv", "http://x", "xml") == "<ok/>" and len(calls) == 2 and not c.failures
    assert (tmp_path / c.index[0]["file"]).read_text() == "<ok/>"

    def dead(url):
        raise OSError("down")

    c2 = RawCache(tmp_path / "b", dead, 0.0)
    assert c2.get("arxiv", "http://y", "xml") is None and c2.failures[0]["error"].startswith("OSError")


# ------------------------------------------------------------------------------- extraction checks
def _ec(**kw):
    base = dict(paper_id="p1", method="RMSNorm", config_change="norm=rmsnorm", claim="c",
                source_span="RMSNorm lowers loss.", direction="improves", mechanism="", mechanism_category="none",
                setting=dict(model_family="transformer", task="language_modeling", scale="tiny", evidence="empirical"))
    return ExtractedClaim(**{**base, **kw})


def test_non_verbatim_quote_is_dropped_and_logged():
    p = paper("p1", abstract="We show RMSNorm lowers loss. More text.", peer_reviewed=None)
    dropped: list = []
    assert to_claim(_ec(source_span="RMSNorm lowers the loss."), p, 1, "evt", dropped) is None
    assert dropped[0]["reason"] == "non-verbatim source_span"
    c = to_claim(_ec(), p, 1, "evt", dropped)
    assert c.source_span in p.abstract and c.tier == "T3" and c.coverage == "covers" and c.covers_our_setting
    assert c.config_change == {"norm": "rmsnorm"} and c.extraction_event == "evt"


def test_mechanism_category_is_kept_only_with_a_verbatim_quote():
    p = paper("p1", abstract="We show RMSNorm lowers loss. It works by stabilizing the gradient norm.")
    dropped: list = []
    ok = to_claim(_ec(mechanism_category="gradient_stability", mechanism="stabilizing the gradient norm"), p, 1, "evt", dropped)
    assert (ok.mechanism_category, ok.mechanism) == ("gradient_stability", "stabilizing the gradient norm") and not dropped
    bad = to_claim(_ec(mechanism_category="gradient_stability", mechanism="stabilizes gradients"), p, 2, "evt", dropped)
    assert bad is not None and (bad.mechanism_category, bad.mechanism) == ("", "")           # paraphrase, not a quote
    assert "non-verbatim mechanism" in dropped[0]["reason"] and "without a verbatim quote" in dropped[1]["reason"]
    bare = to_claim(_ec(mechanism_category="expressivity", mechanism=""), p, 3, "evt", [])
    assert bare.mechanism_category == ""                                                    # a category alone is not kept
    none = to_claim(_ec(mechanism_category="none", mechanism="stabilizing the gradient norm"), p, 4, "evt", [])
    assert (none.mechanism_category, none.mechanism) == ("", "")


def test_extraction_prompt_lists_the_fixed_mechanism_vocabulary():
    from noesis_lab.schemas import MECHANISMS, ExtractedClaim
    from noesis_lab.search.extract import _prompt
    text = _prompt([paper("p1")])
    assert len(MECHANISMS) == 12 and all(f"- {k}:" in text for k in MECHANISMS)
    allowed = set(ExtractedClaim.model_json_schema()["properties"]["mechanism_category"]["enum"])
    assert allowed == set(MECHANISMS) | {"none"}                                           # the schema is closed


def test_config_change_outside_the_allowlist_becomes_null():
    p = paper("p1", abstract="We show RMSNorm lowers loss. More text.", peer_reviewed=True)
    dropped: list = []
    c = to_claim(_ec(config_change="norm=batchnorm"), p, 1, "evt", dropped)
    assert c.config_change is None and c.dimension == "outside_testbed" and c.tier == "T2"
    assert "not in the allowlist" in dropped[0]["reason"]


def test_query_plan_is_capped_and_every_runnable_change_is_always_searched(tmp_path):
    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    plan, queries, err = plan_queries(llm, load_niche(), max_queries=2)
    assert not err and plan is not None
    assert len([q for q in queries if q["kind"] != "deterministic"]) == 2      # cap enforced by code
    served = {k for q in queries if q["kind"] == "deterministic" for k in q["dimension"].split(", ")}
    assert served == set(stats.ALLOWED_CHANGES)                                 # every runnable change is searched
    det = [q["q"] for q in queries if q["kind"] == "deterministic"]
    assert len(det) == len(set(det)) < len(stats.ALLOWED_CHANGES)               # shared queries run once

    def broken(role, system, user, schema):
        raise RuntimeError("no LLM")

    plan, queries, err = plan_queries(LLM("mock", CFG["llm"], mock_fn=broken), load_niche())
    assert plan is None and err and queries == deterministic_queries()


# ------------------------------------------------------------------------------- queue from frozen claims
def _claim(cid, change, outcome="improves", tier="T1", coverage=None, covers=False):
    return Claim(claim_id=cid, paper_id="p", dimension="d", method="m", setting="s", claim="c",
                 source_span="x", expected_outcome=outcome, config_change=change, tier=tier,
                 coverage=coverage, covers_our_setting=covers)


def test_queue_statuses_from_tiers_holds_and_overrides():
    claims = [
        _claim("a1", {"activation": "relu2"}, covers=True),                              # T1 covers -> rejected
        _claim("a2", {"activation": "swiglu"}), _claim("x1", {"activation": "swiglu"}, tier="T3", coverage="partial"),
        _claim("a3", {"pos_encoding": "rope"}),
        _claim("a4", {"dropout": "0.1"}), _claim("x2", {"dropout": "0.1"}, tier="T2", coverage="none"),
        _claim("a5", {"norm": "rmsnorm"}, outcome="context", covers=True),               # context never covers
    ]
    base = ExperimentConfig()
    q = stats.candidate_queue(base, claims, [], held=["pos_encoding=rope"])
    assert [(i.key, i.status) for i in q] == [
        ("dropout=0.1", "open"), ("norm=rmsnorm", "open"),
        ("activation=swiglu", "soft_rejected"),                # preprint covers it: end of the queue
        ("pos_encoding=rope", "held"), ("activation=relu2", "rejected_prior_art")]
    assert q[2].covering_claim_ids == ["x1"] and [i.key for i in stats.open_items(q)] == ["dropout=0.1", "norm=rmsnorm"]
    q2 = stats.candidate_queue(base, claims, [], overrides=["activation=swiglu"])       # PI override / hold dismissed
    assert [i.key for i in stats.open_items(q2)] == ["activation=swiglu", "dropout=0.1", "pos_encoding=rope", "norm=rmsnorm"]
    q3 = stats.candidate_queue(base, claims, [], soft_rejected=["dropout=0.1"])         # soft-rejected at the gate
    assert next(i for i in q3 if i.key == "dropout=0.1").status == "soft_rejected"


def test_t4_overlap_flags_only_unextracted_papers_above_the_threshold():
    swiglu = ("SwiGLU feed-forward blocks", "We replace the GELU feed-forward block with SwiGLU gated linear units "
              "and report lower validation loss.")
    ps = [paper("u1", *swiglu, tier="T4"),
          paper("u2", "Tiny character-level Transformer language models",        # on-niche boilerplate only
                "We train a tiny character-level Transformer language model for a fixed step budget.", tier="T4"),
          paper("e1", *swiglu, tier="T3"),                                       # extracted: not a T4 question
          paper("o1", "Bread", "A recipe for bread with flour and water."),
          paper("o2", "Optimizers", "Adam and SGD with momentum for training deep networks.")]
    sup = [_claim("c", {"activation": "swiglu"})]
    sup[0].method, sup[0].claim = "SwiGLU gated linear units in the feed-forward block", "SwiGLU lowers validation loss."
    text = candidate_text("activation", "swiglu", sup)
    hits = t4_overlaps(text, ps, 0.35)
    assert [h[0] for h in hits] == ["u1"] and hits[0][1] >= 0.35


# ------------------------------------------------------------------------------- build + freeze
@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    out = tmp_path / "corpus"
    llm = LLM("mock", CFG["llm"], recordings_path=out / "llm.jsonl", mock_fn=make_mock({}))
    meta = build_corpus(out, load_niche(), llm, today="2026-10-03", fetch=fake_fetch, sleep=lambda s: None)
    return out, meta


def test_build_corpus_freezes_everything_with_reasons(corpus):
    out, meta = corpus
    for f in ("niche.yaml", "query_plan.json", "papers.json", "claims.json", "dropped.json", "meta.json",
              "raw/index.json", "llm.jsonl"):
        assert (out / f).exists(), f
    assert meta["mode"] == "live_search" and meta["search_date"] == "2026-10-03"
    snap = Snapshot.from_corpus(out)                       # loading re-checks every span verbatim
    papers, claims = snap.papers, snap.claims
    assert "2404.00004" not in papers and "2407.00007" not in papers
    reasons = {d["paper_id"]: d["reason"] for d in json.loads((out / "dropped.json").read_text())}
    assert reasons["2404.00004"] == "exclude_term: vision transformer"
    assert reasons["2407.00007"] == "duplicate of 2402.00002 by title"
    assert papers["2002.04745"].source == "curated"        # the seed did not replace the curated paper
    # tiers: journal_ref -> T2, OpenAlex conference -> T2, unknown status -> T3, no claim -> T4
    assert claims["x2401_00001_1"].tier == "T2" and claims["x2402_00002_1"].tier == "T2"
    assert claims["x2405_00005_1"].tier == "T3" and claims["x2405_00005_1"].coverage == "none"
    assert papers["2403.00003"].tier == "T4" and papers["2406.00006"].tier == "T4"
    assert claims["x2402_00002_1"].coverage == "partial" and claims["c01"].tier == "T1"
    assert meta["tiers"] == {"T1": 13, "T2": 2, "T3": 1, "T4": 2}
    assert all(p.map_xy is not None and p.relevance is not None for p in papers.values())
    assert meta["validation"]["n"] == 9 and "config_mapping" in meta["validation"]
    assert "retrieved on 2026-10-03" in snap.not_found_label() and "novel" not in snap.not_found_label()


def test_too_few_papers_falls_back_to_the_curated_snapshot(tmp_path):
    out = tmp_path / "corpus"                              # default min_papers = 10 > the 7 recorded papers
    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    meta = build_corpus(out, load_niche(), llm, today="2026-10-03", fetch=fake_fetch, sleep=lambda s: None)
    assert meta["mode"] == "curated_only" and meta["label"] == CURATED_ONLY
    assert any("fell back to the curated snapshot" in d for d in meta["degraded"])
    assert len(Snapshot.from_corpus(out).papers) == 10


def test_a_badly_degraded_search_aborts_and_freezes_nothing(tmp_path):
    def dead(url):
        raise OSError("network down")

    out = tmp_path / "c"
    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    meta = build_corpus(out, load_niche(), llm, today="2026-10-03", fetch=dead, sleep=lambda s: None)
    assert meta["mode"] == "aborted_degraded" and meta["failure_fraction"] == 1.0 and meta["failed"] >= 1
    assert not (out / "meta.json").exists() and not (out / "papers.json").exists()   # nothing frozen
    assert (out / "raw" / "index.json").exists()                                     # but the resume log is kept


def test_gate_soft_rejects_on_an_auto_claim_and_the_pi_can_override(corpus):
    out, _ = corpus
    snap = Snapshot.from_corpus(out)
    fx = Fixture(fixture_id="gen_activation_swiglu", kind="generated", origin="generated",
                 candidate_key="activation=swiglu", title="t", statement="SwiGLU feed-forward blocks",
                 allowed_fields=["activation"], cited_claim_ids=["x2402_00002_1"])

    def says_same(role, system, user, schema):
        return LiteratureOutputV2(same_comparison_claim_ids=["x2402_00002_1"], rationale="same change")

    pa = LiteratureAgent(LLM("mock", CFG["llm"], mock_fn=says_same), snap).check(fx)
    assert (pa.verdict, pa.tier, pa.coverage, pa.gate) == (PriorArtVerdictKind.known, "T2", "partial", "soft_reject")
    assert pa.passage in snap.papers["2402.00002"].abstract and "T2" in pa.label
    pa2 = LiteratureAgent(LLM("mock", CFG["llm"], mock_fn=says_same), snap, overrides=("activation=swiglu",)).check(fx)
    assert pa2.gate == "run"


# ------------------------------------------------------------------------------- session + replay, no network
def test_session_with_live_search_replays_from_the_frozen_corpus_without_network(session_name, monkeypatch, capsys):
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True,
                            niche=str(ROOT / "niche.yaml"), fetch=fake_fetch, sleep=lambda x: None))
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", lambda cfg: 0.0)
    out = s.run()
    d = ROOT / "results" / session_name
    assert out["status"] == "complete" and "live search" in out["corpus"]
    st = Store(d / "notebook.sqlite", readonly=True)
    queue = {q["key"]: q for q in st.get_meta("candidate_queue")}
    assert queue["activation=swiglu"]["status"] == "soft_rejected"       # covered by a recorded "preprint"
    assert queue["pos_encoding=rope"]["status"] == "soft_rejected"
    assert "activation=relu2" not in queue                               # curated claim covers it: not a gap at all
    soft = [x for x in st.decisions() if x["kind"] == "queue_soft_rejection"]
    assert {x["candidate_key"] for x in soft} == {"activation=swiglu", "pos_encoding=rope"}
    assert all(c["passage"] and c["tier"] in ("T2", "T3") for x in soft for c in x["covering"])
    ran = [x["key"] for x in st.decisions() if x["kind"] == "exploration_result"]
    assert ran and not {"activation=swiglu", "pos_encoding=rope", "activation=relu2"} & set(ran)
    assert st.get_meta("corpus")["mode"] == "live_search" and st.get_meta("niche")["title"]
    st.close()
    manifest = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert "corpus/claims.json" in manifest and any(k.startswith("corpus/raw/") for k in manifest)
    assert verify_manifest(d) == [] and rederive(d) == []

    import urllib.request

    def no_network(*a, **k):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert main(["replay", "--session", session_name]) == 0
    assert "REPLAY OK" in capsys.readouterr().out


def test_frozen_corpus_can_be_reused_by_a_later_session(corpus, session_name):
    out, _ = corpus
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True, corpus=str(out)))
    assert s.snap.live and (s.out / "corpus" / "raw" / "index.json").exists()
    s.store.close()
    shutil.rmtree(s.out, ignore_errors=True)


def test_dashboard_renders_the_literature_map_for_a_search_session(session_name, monkeypatch):

    from streamlit.testing.v1 import AppTest
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True,
                            niche=str(ROOT / "niche.yaml"), fetch=fake_fetch, sleep=lambda x: None))
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", lambda cfg: 0.0)
    s.run()
    monkeypatch.setenv("NOESIS_NOTEBOOK", str(ROOT / "results" / session_name / "notebook.sqlite"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [h.value for h in at.subheader]
    assert "Coverage grid" in heads and "Paper map" in heads
    assert any("chosen by the PI" in m.value for m in at.markdown)
    assert any("retrieved on" in m.value for m in at.markdown)
    assert not any("novel" in m.value.replace("“novel”", "") for m in at.markdown)
