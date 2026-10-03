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
