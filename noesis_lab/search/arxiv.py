"""arXiv API client (export.arxiv.org/api/query). Papers come only from API responses."""
from __future__ import annotations

import hashlib
import re
import urllib.parse
import xml.etree.ElementTree as ET

from ..schemas import NicheSpec, RetrievedPaper

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def to_arxiv_query(q: str, niche: NicheSpec, date_to: str) -> str:
    """'"RMSNorm" AND "language model"' -> all:"RMSNorm" AND all:"language model" AND (cat:…) AND
    submittedDate:[…]. Phrases are split on AND; everything else is treated as one phrase."""
    parts = [p.strip().strip('"').replace('"', "") for p in re.split(r"\s+AND\s+", q) if p.strip().strip('"')]
    terms = " AND ".join(f'all:"{p}"' for p in parts)
    if niche.arxiv_categories:
        terms += " AND (" + " OR ".join(f"cat:{c}" for c in niche.arxiv_categories) + ")"
    a, b = niche.date_from.replace("-", ""), date_to.replace("-", "")
    return f"{terms} AND submittedDate:[{a}0000 TO {b}2359]"


def search_url(q: str, niche: NicheSpec, date_to: str, max_results: int) -> str:
    return API + "?" + urllib.parse.urlencode({
        "search_query": to_arxiv_query(q, niche, date_to), "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending"})


def title_url(title: str, max_results: int = 3) -> str:
    """Title search for a scout title hint. No category/date filter: we only want that paper."""
    clean = re.sub(r'[^\w\s-]', " ", title)
    return API + "?" + urllib.parse.urlencode({
        "search_query": f'ti:"{_ws(clean)}"', "start": 0, "max_results": max_results})


def id_list_url(ids: list[str]) -> str:
    return API + "?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})


def parse_feed(xml_text: str, query: str) -> list[RetrievedPaper]:
    out = []
    for e in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        url = e.findtext(f"{ATOM}id", "")
        if "/abs/" not in url:
            continue                                     # arXiv error entries have no abs URL
        pid = re.sub(r"v\d+$", "", url.rsplit("/abs/", 1)[-1])
        abstract = _ws(e.findtext(f"{ATOM}summary"))
        if not abstract:
            continue
        journal = _ws(e.findtext(f"{ARXIV}journal_ref")) or None
        out.append(RetrievedPaper(
            paper_id=pid, title=_ws(e.findtext(f"{ATOM}title")),
            authors=[a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
            published=e.findtext(f"{ATOM}published", "")[:10], url=f"https://arxiv.org/abs/{pid}",
            abstract=abstract, abstract_sha256=hashlib.sha256(abstract.encode()).hexdigest(),
            source="arxiv", venue=journal, peer_reviewed=True if journal else None,
            doi=_ws(e.findtext(f"{ARXIV}doi")) or None,
            categories=[c.get("term", "") for c in e.findall(f"{ATOM}category")],
            queries=[query], tier="T4"))
    return out
