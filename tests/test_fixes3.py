"""FIXES3: arXiv backoff + cache resume + abort-on-degraded, the deterministic gate constraint,
Fixture B dropped in explore sessions, and Lion's scaled hyper-parameters."""
import json

import pytest
import torch

from noesis_lab.config import ROOT, baseline_config, get_profile
from noesis_lab.literature import LiteratureAgent, _sub_change
from noesis_lab.llm import LLM
from noesis_lab.mock_llm import make_mock
from noesis_lab.schemas import (
    Claim,
    ExperimentConfig,
    Fixture,
    LiteratureOutputV2,
    PriorArtVerdictKind,
)
from noesis_lab.search import openalex
from noesis_lab.search.corpus import build_corpus, load_niche
from noesis_lab.search.http import RawCache
from noesis_lab.testbed.harness import build_optimizer
from noesis_lab.testbed.model import GPT

from .test_search import CFG, fake_fetch, paper


# --------------------------------------------------------------------------- P0-1 HTTP
class _HTTPError(Exception):
    def __init__(self, code, retry_after=None):
        super().__init__(f"HTTP {code}")
        self.code = code
        self.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}


def test_429_backs_off_then_succeeds_and_honors_retry_after(tmp_path):
    waited, attempts = [], []

    def fetch(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise _HTTPError(429, retry_after=30)      # server told us to wait 30s
        if len(attempts) == 2:
            raise _HTTPError(503)                      # transient, no header -> exponential backoff
        return "<ok/>"

    c = RawCache(tmp_path, fetch, min_interval_s=0.0, sleep=waited.append, max_retries=3, backoff=(15.0, 45.0, 120.0))
    assert c.get("arxiv", "http://x", "xml") == "<ok/>" and len(attempts) == 3 and not c.failures
    assert waited == [30.0, 45.0]                      # Retry-After, then backoff[1] (second retry)


def test_retries_are_bounded_and_a_persistent_429_is_recorded_with_its_status(tmp_path):
    def always429(url):
        raise _HTTPError(429)

    c = RawCache(tmp_path, always429, min_interval_s=0.0, sleep=lambda s: None, max_retries=3)
    assert c.get("arxiv", "http://x", "xml") is None
    assert len(c.failures) == 1 and c.failures[0]["status"] == 429 and c.failure_fraction == 1.0


def test_cache_resumes_on_a_rerun_and_never_caches_failures(tmp_path):
    calls, b_seen = [], []

    def flaky(url):
        calls.append(url)
        if url.endswith("b") and not b_seen:
            b_seen.append(url)
            raise _HTTPError(429)                      # "b" fails on the very first attempt ever, then succeeds
        return f"body for {url}"

    c1 = RawCache(tmp_path, flaky, min_interval_s=0.0, sleep=lambda s: None, max_retries=0)
    assert c1.get("arxiv", "http://a", "xml") == "body for http://a"
    assert c1.get("arxiv", "http://b", "xml") is None and len(c1.failures) == 1
    # a fresh cache over the same dir reuses "a" (no fetch) and only re-fetches the failed "b"
    calls.clear()
    c2 = RawCache(tmp_path, flaky, min_interval_s=0.0, sleep=lambda s: None, max_retries=0)
    assert c2.get("arxiv", "http://a", "xml") == "body for http://a" and calls == []       # served from cache
    assert c2.get("arxiv", "http://b", "xml") == "body for http://b" and calls == ["http://b"]
    assert c2.reused == 1 and c2.attempted == 1 and not c2.failures


def test_failure_fraction_ignores_cache_hits(tmp_path):
    (tmp_path / "raw").mkdir()
    c = RawCache(tmp_path / "raw", lambda u: "<ok/>", min_interval_s=0.0, sleep=lambda s: None)
    c.get("arxiv", "http://a", "xml")
    assert c.get("arxiv", "http://a", "xml") == "<ok/>"       # cache hit
    assert c.attempted == 1 and c.reused == 1 and c.failure_fraction == 0.0


def test_openalex_abstract_is_reconstructed_from_the_inverted_index():
    inv = {"RMSNorm": [0], "lowers": [1], "loss": [2]}
    assert openalex._abstract(inv) == "RMSNorm lowers loss"
    body = json.dumps({"results": [
        {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.48550/arxiv.2402.00002",
         "title": "X", "publication_date": "2024-02-01", "cited_by_count": 5,
         "abstract_inverted_index": inv, "primary_location": {"source": {"type": "conference", "display_name": "V"}}}]})
    (p,) = openalex.parse_search(body, "tag")
    assert p.paper_id == "2402.00002" and p.source == "openalex" and p.abstract == "RMSNorm lowers loss"
    assert p.peer_reviewed is True and p.tier == "T4" and p.queries == ["tag"]


def test_a_badly_degraded_build_aborts_but_a_good_one_does_not(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    monkeypatch.setitem(CFG["lit_search"], "openalex_fallback", False)      # so a dead arXiv really fails
    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    meta = build_corpus(tmp_path / "dead", load_niche(), llm, today="2026-10-03",
                        fetch=lambda u: (_ for _ in ()).throw(OSError("down")), sleep=lambda s: None)
    assert meta["mode"] == "aborted_degraded"
    ok = build_corpus(tmp_path / "ok", load_niche(), llm, today="2026-10-03", fetch=fake_fetch, sleep=lambda s: None)
    assert ok["mode"] == "live_search" and ok["explore"]["requests_failed"] == 0


# --------------------------------------------------------------------------- P0-2 deterministic gate
def _c(cid, change, tier="T3", coverage="covers", outcome="improves"):
    return Claim(claim_id=cid, paper_id="p", dimension="d", method="m", setting="s", claim="c", source_span="x",
                 expected_outcome=outcome, config_change=change, tier=tier,
                 coverage=None if tier == "T1" else coverage,
                 covers_our_setting=(tier == "T1" and coverage == "covers"))


def test_sub_change_is_a_subset_match_on_field_and_value():
    assert _sub_change({"norm_position": "post"}, {"norm_position": "post"})          # single-field: exact
    assert _sub_change({"norm": "rmsnorm"}, {"norm": "rmsnorm", "dropout": "0.1"})     # part of a combination
    assert not _sub_change({"norm_position": "pre"}, {"norm_position": "post"})        # same field, other value
    assert not _sub_change(None, {"norm_position": "post"}) and not _sub_change({}, {"norm_position": "post"})
    assert not _sub_change({"a": "1", "b": "2"}, {"a": "1"})                           # claim broader than the delta


class _Snap:
    def __init__(self, claims):
        self.claims = {c.claim_id: c for c in claims}
        self.papers = {"p": paper("p", abstract="x")}
        self.size, self.live, self.directions = 1, True, []

    def search(self, text, k):
        return [(c, 0.0) for c in self.claims.values()]

    def not_found_label(self):
        return "not found in 1 retrieved papers"


def _gated(claims, listed, require):
    snap = _Snap(claims)

    def fn(role, system, user, schema):
        return LiteratureOutputV2(same_comparison_claim_ids=listed, rationale="r")

    fx = Fixture(fixture_id="gap_norm_position_post", kind="gap", origin="generated",
                 candidate_key="norm_position=post", title="t", statement="s", allowed_fields=["norm_position"])
    return LiteratureAgent(LLM("mock", CFG["llm"], mock_fn=fn), snap).check(fx, require_change=require)


def test_a_curated_claim_about_a_different_change_cannot_reject_a_generated_candidate():
    # c01 (Pre-LN without warm-up) and a T3 claim that actually tests Post-LN are both "listed".
    claims = [_c("c01", None, tier="T1"),                                   # different comparison, null change
              _c("cPost", {"norm_position": "post"}, tier="T3", coverage="covers")]
    pa = _gated(claims, ["c01", "cPost"], {"norm_position": "post"})
    assert pa.gate == "soft_reject" and pa.claim_id == "cPost"              # the T1 claim is context, not a reject
    # with only c01 listed, the candidate is not prior art at all
    pa2 = _gated([_c("c01", None, tier="T1")], ["c01"], {"norm_position": "post"})
    assert pa2.verdict == PriorArtVerdictKind.not_found and pa2.gate == "run"


def test_a_matching_curated_claim_still_rejects():
    pa = _gated([_c("cT1", {"norm_position": "post"}, tier="T1", coverage="covers")], ["cT1"], {"norm_position": "post"})
    assert pa.gate == "reject" and pa.verdict == PriorArtVerdictKind.known


# --------------------------------------------------------------------------- P0-3 Fixture B dropped in explore
def test_fixture_b_is_dropped_in_an_explore_session(session_name, monkeypatch):
    from noesis_lab.orchestrator import Session, SessionOpts

    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    s = Session(SessionOpts(session=session_name, profile="smoke", llm_mode="mock", force=True,
                            niche=str(ROOT / "niche.yaml"), fetch=fake_fetch, sleep=lambda x: None))
    assert s.explore and "fixture_b_rmsnorm" not in s.fixtures and "fixture_a_preln_nowarmup" in s.fixtures
    s.store.close()


def test_fixture_b_is_kept_without_an_engine_corpus():
    from noesis_lab.orchestrator import Session, SessionOpts
    s = Session(SessionOpts(session="pytest-nofx", profile="smoke", llm_mode="mock", force=True))
    try:
        assert not s.explore and "fixture_b_rmsnorm" in s.fixtures
    finally:
        s.store.close()
        import shutil
        shutil.rmtree(ROOT / "results" / "pytest-nofx", ignore_errors=True)


# --------------------------------------------------------------------------- P1-5 Lion scaling
def test_lion_uses_a_smaller_step_and_a_larger_decay_than_adam():
    # The scaling is carried by lr_mult / wd_mult (stated default 0.2 / 5, or a full-text recipe);
    # the harness has no hidden per-optimizer scaling any more (see test_lion_schedule.py).
    c = ExperimentConfig(optimizer="lion", lr=0.002, weight_decay=0.01, n_layer=1, n_head=2, n_embd=32, seq_len=16,
                         lr_mult=0.2, wd_mult=5.0)
    opt = build_optimizer(GPT(c, 65), c)
    assert type(opt).__name__ == "Lion"
    assert opt.param_groups[0]["base_lr"] == pytest.approx(0.002 * 0.2)          # lr x 0.2
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01 * 5)        # wd x 5 on the decay group
    assert opt.param_groups[1]["weight_decay"] == 0.0                           # none on biases / norms
    adam = build_optimizer(GPT(c.model_copy(update={"optimizer": "adamw", "lr_mult": 1.0, "wd_mult": 1.0}), 65),
                           c.model_copy(update={"optimizer": "adamw", "lr_mult": 1.0, "wd_mult": 1.0}))
    assert adam.param_groups[0]["base_lr"] == pytest.approx(0.002)               # Adam keeps the baseline lr


def test_lion_trains_on_cpu_with_the_scaled_hyperparameters():
    from noesis_lab.config import load_config, path_of
    from noesis_lab.testbed.harness import load_dataset, train_one
    data = load_dataset(path_of("data"), load_config()["paths"]["data_sha256"])
    c = baseline_config(get_profile("smoke")).model_copy(update={"optimizer": "lion", "lr_mult": 0.2, "wd_mult": 5.0})
    r = train_one(c, 0, get_profile("smoke"), data, torch.device("cpu"))
    assert r.status == "ok" and r.val_loss is not None and r.val_loss < 4.5


# --------------------------------------------------------------------------- P0-3 lit-dryrun in explore mode
def test_lit_dryrun_over_an_engine_corpus_drops_fixture_b_and_is_stable(tmp_path, monkeypatch, capsys):
    from noesis_lab.cli import main
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    out = tmp_path / "corpus"
    llm = LLM("mock", CFG["llm"], recordings_path=out / "llm.jsonl", mock_fn=make_mock({}))
    build_corpus(out, load_niche(), llm, today="2026-10-03", fetch=fake_fetch, sleep=lambda s: None)
    rc = main(["lit-dryrun", "--llm", "mock", "--corpus", str(out), "--repeat", "3"])
    report = capsys.readouterr().out
    assert rc == 0 and "every row STABLE" in report and "UNSTABLE" not in report
    assert "fixture_b_rmsnorm" not in report and "fixture_a_preln_nowarmup" in report   # engine owns RMSNorm
    assert "actions:" in report                                                          # reports the gate action


# --------------------------------------------------------------------------- a part cannot reject the whole
def test_a_single_method_claim_cannot_reject_a_combination_but_the_tighten_pass_sees_it():
    # c10 (ReLU^2 / Primer) is curated, covers our setting, and is about activation=relu2 alone.
    relu2 = _c("c10", {"activation": "relu2"}, tier="T1", coverage="covers")
    combo = {"activation": "relu2", "norm": "rmsnorm"}
    snap = _Snap([relu2])

    def fn(role, system, user, schema):
        return LiteratureOutputV2(same_comparison_claim_ids=["c10"], rationale="about relu2 only")

    fx = Fixture(fixture_id="gap_x", kind="gap", origin="generated", candidate_key="activation=relu2+norm=rmsnorm",
                 title="t", statement="s", allowed_fields=["activation", "norm"])
    pa = LiteratureAgent(LLM("mock", CFG["llm"], mock_fn=fn), snap).check(fx, require_change=combo)
    assert pa.gate == "run" and pa.verdict == PriorArtVerdictKind.not_found    # the whole combination is not prior art
    assert pa.claim_id is None and pa.partial_overlap_claim_ids == ["c10"]     # but the part is seen, for narrowing
    # the same curated claim DOES reject the single-method candidate relu2
    fx1 = Fixture(fixture_id="gap_relu2", kind="gap", origin="generated", candidate_key="activation=relu2",
                  title="t", statement="s", allowed_fields=["activation"])
    pa1 = LiteratureAgent(LLM("mock", CFG["llm"], mock_fn=fn), snap).check(fx1, require_change={"activation": "relu2"})
    assert pa1.gate == "reject" and pa1.claim_id == "c10" and pa1.partial_overlap_claim_ids == []


def test_a_few_failed_requests_with_plenty_of_papers_still_proceeds(tmp_path, monkeypatch):
    # One request fails; the rest succeed. Below the 10% abort threshold -> go ahead with what we
    # got (degraded note), do not abort and do not throw the papers away.
    monkeypatch.setitem(CFG["lit_search"], "min_papers", 3)
    monkeypatch.setitem(CFG["lit_search"], "openalex_fallback", False)
    target = []

    def one_bad(url):
        if "export.arxiv.org" in url and "search_query" in url:   # fail exactly one search URL, on every attempt
            if not target:
                target.append(url)
            if url == target[0]:
                raise OSError("one flake")
        return fake_fetch(url)

    llm = LLM("mock", CFG["llm"], mock_fn=make_mock({}))
    meta = build_corpus(tmp_path / "c", load_niche(), llm, today="2026-10-03", fetch=one_bad, sleep=lambda s: None)
    assert meta["mode"] == "live_search" and meta["papers"] >= 10                 # papers kept, not discarded
    assert meta["explore"]["requests_failed"] == 1 and meta["counts"]["kept"] >= 10
    assert any("request failed" in d for d in meta["degraded"])                   # reported, not fatal
