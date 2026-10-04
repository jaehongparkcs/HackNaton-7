"""OpenAlex enrichment (optional): venue, peer-review status and citation counts, looked up by the
arXiv DOI. The contact email comes from OPENALEX_MAILTO; never hard-coded."""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from collections.abc import Sequence

from ..schemas import RetrievedPaper
from .http import RawCache

API = "https://api.openalex.org/works"
BATCH = 40
_REVIEWED = {"journal", "conference"}


def batch_url(arxiv_ids: Sequence[str]) -> str:
    dois = "|".join(f"10.48550/arxiv.{i}" for i in arxiv_ids)
    params = {"filter": f"doi:{dois}", "per-page": 50,
              "select": "doi,cited_by_count,primary_location,locations"}
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    return API + "?" + urllib.parse.urlencode(params)


def parse_works(body: str) -> dict[str, dict]:
    """arXiv id -> {cited_by, venue, peer_reviewed}. peer_reviewed is True if any location is a
    journal or conference; False only means "no venue on this OpenAlex record"."""
    out = {}
    for w in json.loads(body).get("results", []):
        m = re.search(r"10\.48550/arxiv\.(.+)$", (w.get("doi") or "").lower())
        if not m:
            continue
        venue = None
        for loc in [w.get("primary_location") or {}, *(w.get("locations") or [])]:
            src = loc.get("source") or {}
            if src.get("type") in _REVIEWED:
                venue = src.get("display_name")
                break
        out[m.group(1)] = {"cited_by": int(w.get("cited_by_count") or 0), "venue": venue,
                           "peer_reviewed": venue is not None}
    return out


def enrich(papers: Sequence[RetrievedPaper], cache: RawCache) -> int:
    """Mutates papers in place; returns how many were matched. OpenAlex can only ADD evidence of
    publication (and citation counts); it never marks a paper as unreviewed."""
    todo = [p for p in papers if p.source == "arxiv"]
    matched = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        body = cache.get("openalex", batch_url([p.paper_id for p in chunk]), "json")
        if body is None:
            continue
        found = parse_works(body)
        for p in chunk:
            w = found.get(p.paper_id.lower())
            if not w:
                continue
            matched += 1
            p.cited_by = w["cited_by"]
            if w["peer_reviewed"]:
                p.peer_reviewed, p.venue = True, p.venue or w["venue"]
            # Repository-only is NOT evidence of "not peer-reviewed": OpenAlex often keeps the
            # conference version as a separate work. Status stays None = "review status unknown".
    return matched


SEARCH_API = "https://api.openalex.org/works"


def search_url(query: str, niche, date_to: str, per_page: int = 15, boolean: bool = False) -> str:
    """OpenAlex full-text search: a fallback for scout directions when arXiv throttles, and the
    first source for tighten searches. Higher rate limits; abstracts come back as an inverted index
    and are reconstructed by code. `boolean` keeps the query's quoted phrases and AND operators
    (OpenAlex `search` accepts them), so a tighten query for a combination still needs both methods."""
    terms = query if boolean else " ".join(
        p.strip().strip('"') for p in re.split(r"\s+AND\s+", query) if p.strip().strip('"'))
    params = {"search": terms, "filter": f"from_publication_date:{niche.date_from},to_publication_date:{date_to}",
              "per-page": per_page, "sort": "relevance_score:desc",
              "select": "id,doi,title,publication_date,abstract_inverted_index,primary_location,locations,cited_by_count"}
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    return SEARCH_API + "?" + urllib.parse.urlencode(params)


def _abstract(inv: dict | None) -> str:
    """Reconstruct the abstract text from OpenAlex's {word: [positions]} inverted index."""
    if not inv:
        return ""
    words: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            words[p] = word
    return re.sub(r"\s+", " ", " ".join(words[i] for i in sorted(words))).strip()


def parse_search(body: str, query_tag: str):
    """OpenAlex search results as RetrievedPaper objects (source=openalex). The arXiv id is used
    when the work is an arXiv DOI, else the OpenAlex id; the quote check downstream is unchanged."""
    import hashlib

    from ..schemas import RetrievedPaper
    out = []
    for w in json.loads(body).get("results", []):
        abstract = _abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        doi = (w.get("doi") or "").lower()
        m = re.search(r"10\.48550/arxiv\.(.+)$", doi)
        pid = m.group(1) if m else (w.get("id") or "").rsplit("/", 1)[-1]
        venue = None
        for loc in [w.get("primary_location") or {}, *(w.get("locations") or [])]:
            src = loc.get("source") or {}
            if src.get("type") in _REVIEWED:
                venue = src.get("display_name")
                break
        out.append(RetrievedPaper(
            paper_id=pid, title=re.sub(r"\s+", " ", w.get("title") or "").strip(),
            published=(w.get("publication_date") or "")[:10],
            url=f"https://arxiv.org/abs/{pid}" if m else (w.get("id") or ""),
            abstract=abstract, abstract_sha256=hashlib.sha256(abstract.encode()).hexdigest(),
            source="openalex", venue=venue, peer_reviewed=True if venue else None,
            cited_by=int(w.get("cited_by_count") or 0), doi=doi or None, queries=[query_tag], tier="T4"))
    return out
