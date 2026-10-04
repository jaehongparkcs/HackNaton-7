"""Tighten pass (EXPLORE §5): one targeted search per hypothesis, built from the hypothesis's own
change, then extraction and the quote check. Runs live while a session is recorded; everything it
returns is frozen under `corpus/tighten/` so replay needs no network.

Sources (FINAL_FIXES B4): OpenAlex first (polite pool, ~10 requests/s with OPENALEX_MAILTO), arXiv
only when the OpenAlex request fails. The quote check is identical for both. Responses are also kept
in a URL-keyed cache under work/cache/ that later sessions reuse; the bundle keeps its own copy."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from ..llm import LLM
from ..schemas import Claim, NicheSpec, RetrievedPaper
from . import arxiv, openalex
from .extract import extract
from .http import RawCache
from .rank import filter_excluded

# Human-written search phrase per runnable change (never LLM-generated).
CHANGE_TERMS = {
    "norm=rmsnorm": "RMSNorm", "norm_position=post": "Post-LN", "activation=swiglu": "SwiGLU",
    "activation=relu2": "squared ReLU", "pos_encoding=rope": "rotary position embedding",
    "schedule=constant": "constant learning rate", "optimizer=sgd_momentum": "SGD with momentum",
    "dropout=0.1": "dropout", "qk_norm=true": "QK normalization", "weight_tying=true": "weight tying",
    "z_loss_coef=0.0001": "z-loss", "label_smoothing=0.1": "label smoothing",
    "warmup_frac=0": "without warmup", "warmup_frac=0.02": "warmup", "warmup_frac=0.1": "warmup",
    "grad_clip=0": "gradient clipping", "init_scale=0.5": "initialization scale",
    "init_scale=2.0": "initialization scale", "optimizer=lion": "Lion optimizer",
}
CONTEXT_TERMS = ("language model", "Transformer")


def tighten_queries(delta: dict[str, str], n: int = 2) -> list[str]:
    """Queries for exactly this change. For a combination every query names BOTH methods, so a hit
    is a paper that mentions them together."""
    terms = [CHANGE_TERMS[f"{f}={delta[f]}"] for f in sorted(delta)]
    core = " AND ".join(f'"{t}"' for t in terms)
    return [f'{core} AND "{c}"' for c in CONTEXT_TERMS][:n]


class Tightener:
    def __init__(self, corpus: Path, niche: NicheSpec, *, replay: bool, llm: LLM | None = None,
                 fetch: Callable[[str], str] | None = None, sleep: Callable[[float], None] | None = None,
                 min_interval_s: float = 3.0, n_queries: int = 2, cap: int = 10, today: str = "",
                 openalex_first: bool = False, openalex_min_interval_s: float = 0.1,
                 shared_cache: Path | None = None):
        self.dir, self.niche, self.replay, self.llm = Path(corpus) / "tighten", niche, replay, llm
        self.n_queries, self.cap, self.today, self.openalex_first = n_queries, cap, today, openalex_first
        if not replay:
            kw = {k: v for k, v in (("fetch", fetch), ("sleep", sleep)) if v}
            shared = {name: RawCache(Path(shared_cache) / f"tighten_{name}", min_interval_s=0, **kw)
                      for name in ("arxiv", "openalex")} if shared_cache else {}
            self.cache = RawCache(self.dir / "raw", min_interval_s=min_interval_s, shared=shared.get("arxiv"), **kw)
            self.oa_cache = RawCache(self.dir / "raw_openalex", min_interval_s=openalex_min_interval_s,
                                     shared=shared.get("openalex"), **kw) if openalex_first else None

    def _file(self, fixture_id: str, rnd: int) -> Path:
        return self.dir / f"{re.sub(r'[^A-Za-z0-9_]+', '_', fixture_id)}_r{rnd}.json"

    def run(self, fixture_id: str, rnd: int, delta: dict[str, str], known_papers: set[str]) -> dict:
        """Returns {queries, papers, claims, dropped, degraded}. New papers only (not in the corpus)."""
        f = self._file(fixture_id, rnd)
        if self.replay:
            if not f.exists():        # bundle without a tighten record for this hypothesis
                return {"queries": [], "papers": [], "claims": [], "dropped": [], "degraded": ["no frozen tighten record"]}
            doc = json.loads(f.read_text())
            return {**doc, "papers": [RetrievedPaper(**p) for p in doc["papers"]],
                    "claims": [Claim(**c) for c in doc["claims"]]}
        queries, found, n_fail = [], {}, len(self.cache.failures)
        date_to = self.niche.date_to or self.today
        for q in tighten_queries(delta, self.n_queries):
            tag, body, source = f"tighten {fixture_id}: {q}", None, "arxiv"
            if self.oa_cache is not None:
                body = self.oa_cache.get("openalex_tighten",
                                         openalex.search_url(q, self.niche, date_to, self.cap, boolean=True), "json")
                source = "openalex"
            if body is not None:
                hits = openalex.parse_search(body, tag)
            else:                         # OpenAlex off, or its request failed: arXiv
                source = "arxiv"
                body = self.cache.get("arxiv_tighten", arxiv.search_url(q, self.niche, date_to, self.cap), "xml")
                hits = arxiv.parse_feed(body, tag) if body else []
            queries.append({"q": q, "n_results": len(hits), **({"source": source} if self.oa_cache else {})})
            for p in hits:
                found.setdefault(p.paper_id, p)
        new = [p for p in found.values() if p.paper_id not in known_papers]
        new, excluded = filter_excluded(new, self.niche)
        new = new[: self.cap]
        claims, dropped, errors = extract(self.llm, new, role="extract_tighten") if new else ([], [], [])
        with_claims = {c.paper_id for c in claims}
        for p in new:
            p.tier = "T3" if p.paper_id in with_claims else "T4"
        degraded = [f"{x['kind']} request failed after retry: {x['error']}" for x in self.cache.failures[n_fail:]] + errors
        doc = {"fixture_id": fixture_id, "round": rnd, "delta": delta, "queries": queries,
               "already_in_corpus": sorted(set(found) & known_papers),
               "papers": [p.model_dump() for p in new], "claims": [c.model_dump() for c in claims],
               "dropped": excluded + dropped, "degraded": degraded}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        self.cache.write_index()
        if self.oa_cache is not None:
            self.oa_cache.write_index()
        return {**doc, "papers": new, "claims": claims}
