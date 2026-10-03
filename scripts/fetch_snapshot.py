"""Fetch verbatim arXiv metadata for the curated literature snapshot.

Writes data/papers.json. Run once; the output is committed so nothing is fetched at runtime.
Claims in data/claims.json must quote these abstracts verbatim (checked by verify_snapshot.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATOM = "{http://www.w3.org/2005/Atom}"

# arXiv ids of the curated snapshot (one per exposed experiment dimension, see BUILD_PLAN s4).
PAPER_IDS = [
    "2002.04745",  # Xiong et al. - Pre-LN vs Post-LN
    "2004.08249",  # Liu et al. - difficulty of training transformers (warmup)
    "1910.07467",  # Zhang & Sennrich - RMSNorm
    "1608.03983",  # Loshchilov & Hutter - SGDR (cosine schedule)
    "1711.05101",  # Loshchilov & Hutter - decoupled weight decay (AdamW)
    "1606.08415",  # Hendrycks & Gimpel - GELU
    "2002.05202",  # Shazeer - GLU variants
    "2109.08668",  # So et al. - Primer (squared ReLU)
    "2104.09864",  # Su et al. - RoFormer (RoPE)
    "1207.0580",   # Hinton et al. - dropout
]


def fetch(ids: list[str]) -> list[dict]:
    url = "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids)
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    out = []
    for e in ET.fromstring(raw).findall(f"{ATOM}entry"):
        arxiv_url = e.findtext(f"{ATOM}id", "")
        pid = re.sub(r"v\d+$", "", arxiv_url.rsplit("/abs/", 1)[-1])
        abstract = re.sub(r"\s+", " ", e.findtext(f"{ATOM}summary", "")).strip()
        out.append({
            "paper_id": pid,
            "title": re.sub(r"\s+", " ", e.findtext(f"{ATOM}title", "")).strip(),
            "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published", "")[:10],
            "url": f"https://arxiv.org/abs/{pid}",
            "abstract": abstract,
            "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
        })
    got = {p["paper_id"] for p in out}
    missing = set(ids) - got
    if missing:
        sys.exit(f"arXiv did not return: {sorted(missing)}")
    order = {pid: i for i, pid in enumerate(ids)}
    return sorted(out, key=lambda p: order[p["paper_id"]])


if __name__ == "__main__":
    papers = fetch(PAPER_IDS)
    doc = {
        "source": "arXiv API (export.arxiv.org), abstracts only",
        "fetched": dt.date.today().isoformat(),
        "papers": papers,
    }
    (ROOT / "data" / "papers.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(papers)} papers")
