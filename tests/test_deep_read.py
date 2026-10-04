"""DEEP_READ: full-text reading of the closest papers (CPU, no network, mocked LLM)."""
import json

import pytest

from noesis_lab import gaps, stats
from noesis_lab.cli import main
from noesis_lab.config import ROOT, load_config
from noesis_lab.deep import apply as deep_apply
from noesis_lab.deep.html import classify, parse_paper, quote_in
from noesis_lab.deep.read import (
    build_deep,
    map_recipe,
    numbers_in,
    propose_stated,
    scale_of,
    verify_deep,
    verify_findings,
)
from noesis_lab.deep.select import select_papers
from noesis_lab.llm import LLM
from noesis_lab.mock_llm import make_mock
from noesis_lab.rederive import rederive
from noesis_lab.schemas import DeepFindings, ExperimentConfig, RetrievedPaper, StatedGapProposal
from noesis_lab.store import Store

from .deep_fixtures import DATA, fetch_factory
from .test_gaps import DROP, MECH, RMS, ROPE, SWIGLU, claim
from .test_search import FakeRunner, fake_fetch

CFG = load_config()


def _llm():
    return LLM("mock", CFG["llm"], mock_fn=make_mock({}))


# --------------------------------------------------------------------------- parsing
def test_native_arxiv_html_maps_to_canonical_sections():
    p = parse_paper((DATA / "arxiv_html_lion.html").read_text(), "2302.06675")
    assert p.version == "2302.06675v4"                               # from the page, not from a <script>
    assert set(p.sections) == {"abstract", "introduction", "method", "experimental_setup", "results",
                               "limitations", "conclusion", "appendix_setup"}
    setup = p.sections["experimental_setup"]
    assert "1.3B parameters" in setup                                # a subsection inherits "Experiments"
    assert "AdamW | 3e-4 | 0.1 |" in setup                           # table text is kept
    assert "Loss curves that must be dropped" not in setup           # figures are dropped
    assert "Kingma" not in " ".join(p.sections.values())             # references are dropped
    assert "\\eta" in p.sections["method"]                            # math keeps its LaTeX alttext
    assert "3-10x smaller" in p.sections["appendix_setup"] and "More plots" not in " ".join(p.sections.values())


def test_ar5iv_page_parses_the_same_way():
    p = parse_paper((DATA / "ar5iv_rmsnorm.html").read_text(), "1910.07467")
    assert p.version == "unknown"
    assert p.sections["experimental_setup"].startswith("We use an RNN-based translation model with 60M parameters")
    assert p.sections["results"].startswith("RMSNorm achieves comparable performance")


@pytest.mark.parametrize("heading,appendix,expected", [
    ("5 Limitations", False, "limitations"), ("Conclusion and Future Work", False, "conclusion"),
    ("4.2 Experimental Results", False, "results"), ("Implementation details", False, "experimental_setup"),
    ("1 Introduction", False, "introduction"), ("Related Work", False, "other"),
    ("A Hyperparameters", True, "appendix_setup"), ("B Proofs", True, "other")])
def test_heading_classification(heading, appendix, expected):
    assert classify(heading, appendix=appendix) == expected


# --------------------------------------------------------------------------- quote check
def _item(**kw):
    return {"method": "optimizer=lion", "direction": "improves", "setting_note": "s", "section": "results", **kw}


def test_items_whose_quote_is_not_verbatim_in_that_section_are_dropped_and_logged():
    secs = parse_paper((DATA / "arxiv_html_lion.html").read_text(), "2302.06675").sections
    f = DeepFindings(setting=[], recipes=[], mechanisms=[], limitations=[], small_scale_evidence=[], results=[
        _item(quote="Lion improves validation perplexity over AdamW on every model size."),
        _item(quote="Lion improves validation perplexity over AdamW on every  model size."),   # whitespace: fine
        _item(quote="Lion is three times better than AdamW."),                                 # not in the paper
        _item(quote="Lion is more memory-efficient than AdamW."),                              # wrong section
        _item(quote="Lion improves validation perplexity over AdamW on every model size.", method="optimizer=adam")])
    kept, dropped = verify_findings(f, secs)
    assert len(kept["results"]) == 3 and kept["results"][2]["method"] == ""      # unknown method -> context only
    assert [d["reason"] for d in dropped] == ["quote not verbatim in that section"] * 2
    assert quote_in("  Lion improves validation\nperplexity ", secs["results"])


# --------------------------------------------------------------------------- selection
def _toy():
    cs = [claim("a", SWIGLU, "improves", "T3", "none", mech=MECH, paper="2401.00001"),
          claim("b", ROPE, "improves", "T3", "partial", mech=MECH, paper="2401.00002"),
          claim("c", RMS, "worse", "T2", "none", mech=MECH, paper="2401.00003"),
          claim("d", DROP, "improves", "T1", covers=False, paper="2401.00004"),
          claim("e", SWIGLU, "worse", "T3", "none", paper="W12345")]                  # not an arXiv id: unreadable
    papers = {c.paper_id: RetrievedPaper(paper_id=c.paper_id, title=c.paper_id, abstract="x", url="u",
                                         abstract_sha256="h", source="arxiv", cited_by=i) for i, c in enumerate(cs)}
    return cs, papers


def test_selection_is_deterministic_capped_and_only_arxiv():
    cs, papers = _toy()
    g, found = gaps.find_gaps(cs)
    q = gaps.gap_queue(found, g, cs, top_n=8)
    claims = {c.claim_id: c for c in cs}
    a = select_papers(q, found, g, claims, papers, top_n=8, per_hypothesis=2, max_papers=3)
    b = select_papers(q, found, g, claims, papers, top_n=8, per_hypothesis=2, max_papers=3)
    assert a == b and len(a["selected"]) <= 3 and "W12345" not in a["selected"]
    for t in a["targets"]:
        assert len(t["papers"]) <= 2
        assert [p["proximity"] for p in t["papers"]] == sorted((p["proximity"] for p in t["papers"]), reverse=True)
    swiglu = next(t for t in a["targets"] if t["key"] == "activation=swiglu")
    top = swiglu["papers"][0]
    assert top["paper_id"] == "2401.00001" and {r["relation"] for r in top["relations"]} >= {"path", "method"}


# --------------------------------------------------------------------------- recipes
def _recipe(hp, lo=None, hi=None, quote="", value=None, adamw=None):
    return {"method": "optimizer=lion", "hyperparameter": hp, "value_as_stated": "x", "ratio_to_adamw_min": lo,
            "ratio_to_adamw_max": hi, "value": value, "adamw_value": adamw, "quote": quote, "section": "appendix_setup"}


def test_recipe_maps_to_the_nearest_allowed_multiplier():
    q = "The learning rate for Lion is typically 3-10x smaller than that for AdamW, and the weight decay 3-10x larger."
    r = map_recipe([_recipe("lr", 0.1, 1 / 3, q), _recipe("weight_decay", 3, 10, q)])
    assert (r["lr_mult"], r["wd_mult"]) == (0.2, 5.0)
    r = map_recipe([_recipe("lr", quote="AdamW uses 3e-4 and Lion 3e-5.", value=3e-5, adamw=3e-4)])
    assert (r["lr_mult"], r["wd_mult"]) == (0.1, 1.0)                     # absolute values against its own AdamW
    nums = numbers_in("3 \\times 10^{-4}, 1e-3 and 1.3B")
    assert any(x == pytest.approx(3e-4) for x in nums) and 1e-3 in nums and 1.3 in nums


def test_a_recipe_number_that_is_not_in_the_quote_is_never_used():
    assert map_recipe([_recipe("lr", 0.2, 0.2, "Lion needs a smaller learning rate.")]) is None
    assert map_recipe([_recipe("warmup", 0.1, 0.1, "warmup 0.1")]) is None          # no typed field for it


def test_recipe_multipliers_are_typed_and_left_out_of_the_hash_at_defaults():
    base = ExperimentConfig()
    assert "lr_mult" not in base.model_dump() and base.config_hash() == ExperimentConfig(lr_mult=1.0).config_hash()
    with pytest.raises(ValueError):
        ExperimentConfig(lr_mult=0.3)
    assert ExperimentConfig(optimizer="lion", lr_mult=0.2, wd_mult=5).config_hash() != \
        ExperimentConfig(optimizer="lion").config_hash()


def test_harness_applies_lr_and_wd_multipliers_once():
    from noesis_lab.testbed.harness import build_optimizer
    from noesis_lab.testbed.model import GPT
    small = dict(n_layer=1, n_head=2, n_embd=16)
    for cfg, lr, wd in [(ExperimentConfig(optimizer="lion", **small), 2e-3, 0.01),          # no hidden scaling
                        (ExperimentConfig(optimizer="lion", lr_mult=0.1, wd_mult=10, **small), 2e-4, 0.1),
                        (ExperimentConfig(lr_mult=0.5, **small), 1e-3, 0.01)]:
        opt = build_optimizer(GPT(cfg, vocab_size=10), cfg)
        assert opt.param_groups[0]["base_lr"] == pytest.approx(lr) and opt.param_groups[0]["weight_decay"] == pytest.approx(wd)


def test_scale_from_the_stated_parameter_count():
    assert [scale_of(x) for x in ("5M", "60 million", "125M", "1.3B", "unspecified")] == \
        ["tiny", "small", "medium", "large", "unspecified"]


# --------------------------------------------------------------------------- stated gaps
def test_stated_gap_proposal_is_validated_by_code():
    lim = claim("deep_2401.00009_l1", DROP, "context", "T3", "none", paper="2401.00009")
    lim = lim.model_copy(update={"source_span": "We did not test dropout at small scale.", "span_source": "full_text",
                                 "section": "limitations"})
    ok = propose_stated(_llm(), lim)
    assert ok["delta"] == {"dropout": "0.1"} and ok["concrete"] is True
    bad = LLM("mock", CFG["llm"], mock_fn=lambda r, s, u, sch: StatedGapProposal(
        config_changes=[{"field": "norm", "value": "rmsnorm"}], rationale="r"))
    assert "dropped" in propose_stated(bad, lim)                         # must include the limitation's method
    cs = [lim, claim("x", DROP, "improves", "T3", "none")]
    g, found = gaps.find_gaps(cs, stated=[ok])
    st_ = next(x for x in found if x.gap_type == "stated")
    assert st_.novelty_type == "author_stated" and st_.delta_key == "dropout=0.1"
    w = gaps.TIER_WEIGHT["T3"] * gaps.setting_similarity(lim)
    assert st_.plausibility == pytest.approx(w)                          # concrete: × 1.0
    vague = gaps.find_gaps(cs, stated=[{**ok, "concrete": False}])[1]
    assert next(x for x in vague if x.gap_type == "stated").plausibility == pytest.approx(w * 0.5)
    q = gaps.gap_queue(found, g, cs, top_n=8)
    assert any("stated:dropout=0.1:" + lim.claim_id in i.gap_ids for i in q)


# --------------------------------------------------------------------------- end to end
@pytest.fixture
def deep_corpus(tmp_path, monkeypatch):
    from noesis_lab.search.corpus import build_corpus, load_niche
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    c = tmp_path / "corpus"
    build_corpus(c, load_niche(), _llm(), today="2026-10-04", fetch=fake_fetch, sleep=lambda s: None)
    calls: list = []
    meta = build_deep(c, _llm(), fetch=fetch_factory(small={"2402.00002"}, missing={"1608.03983"}, calls=calls),
                      sleep=lambda s: None, html_cache=tmp_path / "html")
    return c, meta, calls


def test_deep_read_freezes_quotes_hashes_and_never_the_full_text(deep_corpus):
    c, meta, calls = deep_corpus
    d = c / "deep"
    assert {p.name for p in d.iterdir()} >= {"selection.json", "claims.json", "overrides.json", "recipes.json",
                                             "stated.json", "validation.json", "meta.json", "papers", "llm.jsonl"}
    assert "1608.03983" in meta["unavailable"] and meta["read"] and meta["recipes"] == ["optimizer=lion"]
    assert any("ar5iv" in u for u in calls)                              # arXiv HTML first, then ar5iv
    rec = json.loads((d / "papers" / f"{meta['read'][0]}.json").read_text())
    assert rec["html_sha256"] and rec["version"].endswith("v2") and rec["url"].startswith("https://arxiv.org/html/")
    blob = "".join(p.read_text() for p in d.rglob("*.json"))
    assert "We train Transformer language models with 125M parameters" in blob     # a quoted finding
    assert "<html" not in blob and "ltx_section" not in blob                        # never the page itself
    v = meta["validation"]
    assert v["n"] == 9 and set(v["abstract_only"]) == {"config_mapping", "direction", "setting", "setting_of_matched"}
    assert v["with_full_text"]["setting_of_matched"] <= v["with_full_text"]["config_mapping"]


def test_full_text_overrides_settings_but_never_a_curated_label(deep_corpus):
    from noesis_lab.literature import Snapshot
    c, _, _ = deep_corpus
    before, after = Snapshot.from_corpus(c, deep=False), Snapshot.from_corpus(c)
    assert after.deep and after.deep.changes
    ch = after.deep.changes[0]
    assert after.claims[ch["claim_id"]].coverage == ch["after"] and after.claims[ch["claim_id"]].setting_source == "full_text"
    for cid, cl in before.claims.items():
        if cl.tier == "T1":
            assert after.claims[cid] == cl                               # curated claims untouched
    ft = [cl for cl in after.claims.values() if cl.span_source == "full_text"]
    assert ft and all(cl.tier in ("T2", "T3") for cl in ft)              # LLM-transcribed: never T1
    small = next(cl for cl in ft if cl.claim_id.startswith("deep_2402.00002_s"))
    assert small.coverage == "covers" and small.config_change == {"activation": "swiglu"}
    k = "activation=swiglu"
    s0 = stats.literature_status(k, [x for x in before.claims.values() if gaps.method_key(x) == k], contested_open=True)
    s1 = stats.literature_status(k, [x for x in after.claims.values() if gaps.method_key(x) == k], contested_open=True)
    assert s1[0] == "soft_rejected" and small.claim_id in s1[1] and s0 != s1
    assert after.superseded and all(x.superseded_by.startswith("deep_") for x in after.superseded.values())


def test_a_tampered_full_text_claim_is_refused(deep_corpus):
    from noesis_lab.literature import Snapshot
    c, _, _ = deep_corpus
    doc = json.loads((c / "deep" / "claims.json").read_text())
    doc["claims"][0]["source_span"] = "A sentence the paper never wrote."
    (c / "deep" / "claims.json").write_text(json.dumps(doc))
    with pytest.raises(deep_apply.DeepReadError):
        Snapshot.from_corpus(c)


def test_a_deep_read_session_records_updates_and_replays_offline(deep_corpus, session_name, monkeypatch, capsys):
    import urllib.request

    from noesis_lab.orchestrator import Session, SessionOpts
    c, _, _ = deep_corpus
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True, corpus=str(c),
                            fetch=fake_fetch, sleep=lambda x: None))
    s.runner = FakeRunner(s.store, s.profile, s.out / "runs.jsonl", lambda cfg: 0.0)
    assert s.snap.deep and any(g.gap_type == "stated" for g in s.gaps)
    lion_cfg, rc = s._with_recipe(ExperimentConfig(), {"optimizer": "lion"})
    assert (lion_cfg.lr_mult, lion_cfg.wd_mult) == (0.2, 5.0) and rc["paper_id"]
    sgd_cfg, none = s._with_recipe(lion_cfg, {"optimizer": "sgd_momentum"})
    assert none is None and (sgd_cfg.lr_mult, sgd_cfg.wd_mult) == (1.0, 1.0)     # recipes belong to their method
    assert s._with_recipe(ExperimentConfig(), {"norm": "rmsnorm"}) == (ExperimentConfig(), None)
    out = s.run()
    assert out["status"] == "complete"
    d = ROOT / "results" / session_name
    st = Store(d / "notebook.sqlite", readonly=True)
    ups = [x for x in st.decisions() if x["kind"] == "deep_read_update"]
    whats = {x["what"] for x in ups}
    assert {"coverage", "recipe", "stated_gap", "gap_score"} <= whats
    rev = st.get_meta("deep_read")["revisions"]
    g1 = {g["gap_id"]: g for g in st.get_meta("gaps")}
    for key, r in rev.items():                                           # predicted direction is code's
        if r["r1"]:
            assert r["r1"]["predicted_direction"] == g1[r["r1"]["gap_ids"][0]]["predicted_direction"]
    hyp = next(h for h in st.hypotheses() if h.get("kind") == "gap" and h.get("revision"))
    assert "r0" in hyp["revision"] and "deep_findings" in hyp
    events = [e for e in st.events() if e["role"] == "scientist_gap"]
    assert any("FULL-TEXT FINDINGS" in e["user"] for e in events)
    st.close()
    assert rederive(d) == []
    import os

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    os.environ["NOESIS_NOTEBOOK"] = str(d / "notebook.sqlite")
    try:
        at.run()
    finally:
        os.environ.pop("NOESIS_NOTEBOOK", None)
    assert not at.exception, [e.value for e in at.exception]
    assert any("What the full text changed" in m.value for m in at.markdown)
    assert any("Recipe from arXiv:" in m.value or "recipe" in m.value.lower() for m in at.markdown)

    def no_network(*a, **k):
        raise AssertionError("replay must not touch the network")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert main(["replay", "--session", session_name]) == 0 and "REPLAY OK" in capsys.readouterr().out


def test_verify_deep_rechecks_hash_and_quotes(deep_corpus):
    c, meta, _ = deep_corpus
    assert verify_deep(c, fetch_factory(small={"2402.00002"}, missing={"1608.03983"})) == []
    changed = verify_deep(c, lambda url: "<html><body><p>a different page</p></body></html>")
    assert changed and any("SHA256 differs" in p for p in changed) and any("quote not found" in p for p in changed)


def test_recorded_bundles_have_no_deep_read_and_still_verify():
    from noesis_lab.config import bundle_config, protocol
    for s in ("golden", "explore1", "explore2"):
        assert not (ROOT / "results" / s / "corpus" / "deep").exists()
        assert protocol(bundle_config(ROOT / "results" / s))["deep_read_version"] == 0
    assert protocol(load_config())["deep_read_version"] == 1
