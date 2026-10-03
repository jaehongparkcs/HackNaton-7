"""Dedupe, filter and rank retrieved papers. Pure and deterministic; every drop has a reason."""
from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence

from ..schemas import NicheSpec, RetrievedPaper
from .textsim import similarities

W_TEXT, W_CITES, W_RECENCY = 0.6, 0.2, 0.2


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def dedupe(papers: Sequence[RetrievedPaper]) -> tuple[list[RetrievedPaper], list[dict]]:
    """By arXiv id, then DOI, then normalized title. The first occurrence wins (curated papers and
    seeds are passed first); its `queries` list absorbs the duplicates'."""
    kept: list[RetrievedPaper] = []
    by_key: dict[tuple[str, str], RetrievedPaper] = {}
    dropped = []
    for p in papers:
        keys = [("id", p.paper_id)] + ([("doi", p.doi.lower())] if p.doi else []) + [("title", _norm_title(p.title))]
        hit = next((k for k in keys if k in by_key), None)
        if hit:
            first = by_key[hit]
            first.queries.extend(q for q in p.queries if q not in first.queries)
            first.directions.extend(d for d in p.directions if d not in first.directions)
            if p.paper_id != first.paper_id:
                dropped.append({"paper_id": p.paper_id, "title": p.title,
                                "reason": f"duplicate of {first.paper_id} by {hit[0]}"})
            continue
        kept.append(p)
        for k in keys:
            by_key[k] = p
    return kept, dropped


def filter_excluded(papers: Sequence[RetrievedPaper], niche: NicheSpec
                    ) -> tuple[list[RetrievedPaper], list[dict]]:
    """Hard filter on `exclude_terms` in title/abstract. Curated papers are never dropped."""
    kept, dropped = [], []
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        term = next((t for t in niche.exclude_terms if t.lower() in text), None)
        if term and p.source != "curated":
            dropped.append({"paper_id": p.paper_id, "title": p.title, "reason": f"exclude_term: {term}"})
        else:
            kept.append(p)
    return kept, dropped


def _recency(published: str, date_from: str, date_to: str) -> float:
    try:
        d, a, b = (dt.date.fromisoformat(x[:10]) for x in (published, date_from, date_to))
    except ValueError:
        return 0.0
    span = (b - a).days
    return min(max((d - a).days / span, 0.0), 1.0) if span > 0 else 0.0


def rank(papers: Sequence[RetrievedPaper], niche: NicheSpec, *, date_to: str,
         max_papers: int | None = None) -> tuple[list[RetrievedPaper], list[dict]]:
    """relevance = 0.6 × TF-IDF cosine(abstract, niche text) + 0.2 × normalized log-citations
    + 0.2 × recency. Keeps the top `max_papers` (curated and seed papers always kept). Ties are
    broken by arXiv id. The components are stored on each paper."""
    if not papers:
        return [], []
    query = " ".join([niche.title, niche.description, *niche.include_terms])
    sims = similarities(query, [f"{p.title} {p.abstract}" for p in papers])
    max_log = max(math.log1p(p.cited_by) for p in papers) or 1.0
    for p, sim in zip(papers, sims, strict=True):
        comp = {"text": round(sim, 6), "citations": round(math.log1p(p.cited_by) / max_log, 6),
                "recency": round(_recency(p.published, niche.date_from, date_to), 6)}
        p.relevance_components = comp
        p.relevance = round(W_TEXT * comp["text"] + W_CITES * comp["citations"]
                            + W_RECENCY * comp["recency"], 6)
    pinned = [p for p in papers if p.source == "curated" or p.paper_id in niche.seed_papers]
    pin_ids = {p.paper_id for p in pinned}
    rest = sorted((p for p in papers if p.paper_id not in pin_ids), key=lambda p: (-p.relevance, p.paper_id))
    cap = max_papers or niche.max_papers
    room = max(cap - len(pinned), 0)
    dropped = [{"paper_id": p.paper_id, "title": p.title,
                "reason": f"ranked below max_papers ({cap}); relevance {p.relevance}"}
               for p in rest[room:]]
    kept = pinned + rest[:room]
    return sorted(kept, key=lambda p: (-p.relevance, p.paper_id)), dropped
