"""Coverage grid: the literature map's functional view. Pure formatting over frozen claims and
stored results; every decision shown here was made elsewhere (queue, gate, statistics)."""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..stats import ALLOWED_CHANGES

OUTSIDE = "outside testbed"
BUCKETS = ("covers our setting", "partial", "other settings")


def claim_key(c: dict) -> str:
    ch = c.get("config_change")
    if ch and len(ch) == 1:
        (f, v), = ch.items()
        return f"{f}={v}"
    return OUTSIDE


def claim_bucket(c: dict) -> str:
    cov = c.get("coverage") or ("covers" if c.get("covers_our_setting") else "none")
    return {"covers": BUCKETS[0], "partial": BUCKETS[1]}.get(cov, BUCKETS[2])


def _cell(claims: list[dict]) -> str:
    if not claims:
        return ""
    tiers = Counter(c.get("tier", "T1") for c in claims)
    direction = Counter(c["expected_outcome"] for c in claims).most_common(1)[0][0]
    return " ".join(f"{t}·{n}" for t, n in sorted(tiers.items())) + f"  ({direction})"


def coverage_grid(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows = runnable config changes + one "outside testbed" row. A gap is a runnable change with
    no directional claim that covers (or partially covers) our setting but an "improves" claim
    elsewhere: exactly what the candidate queue tries first."""
    claims = list(data["claims"].values())
    queue = {q["key"]: q for q in data.get("queue") or []}
    hyp_by_key = {}
    for h in data["hyps"]:
        key = h.get("candidate_key")
        if not key and h.get("config_delta") and len(h["config_delta"]) == 1:
            key = "{}={}".format(*next(iter(h["config_delta"].items())))
        if key:
            hyp_by_key[key] = h
    rows = []
    for key in [*ALLOWED_CHANGES, OUTSIDE]:
        mine = [c for c in claims if claim_key(c) == key]
        cells = {b: [c for c in mine if claim_bucket(c) == b] for b in BUCKETS}
        directional_cover = [c for b in BUCKETS[:2] for c in cells[b] if c["expected_outcome"] != "context"]
        gap = (key != OUTSIDE and not directional_cover
               and any(c["expected_outcome"] == "improves" for c in cells[BUCKETS[2]]))
        h = hyp_by_key.get(key)
        ana = next((a for a in reversed(data["analyses"]) if h and a["hypothesis_id"] == h["hypothesis_id"]), None)
        ev = [e for e in data["evidence"] if ana and e["analysis_id"] == ana["analysis_id"]]
        q = queue.get(key, {})
        rows.append({
            "config change": key, "gap": "GAP" if gap else "",
            **{b: _cell(cells[b]) for b in BUCKETS},
            "queue": q.get("status", "tested" if h else ("—" if key == OUTSIDE else "no claim maps here")),
            "gate verdict": ((h.get("prior_art") or {}).get("verdict", "—") if h else "—"),
            "measured": f"{ana['branch']} ({ana['improvement_in_noise_sd']:+.2f}× SD)" if ana else (h["status"] if h else "—"),
            "relation": ", ".join(f"{e['claim_id']}: {e['relation']}" for e in ev) or "—",
        })
    return rows


def quotes(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The claims behind one grid row, with tier badge, date and review status."""
    out = []
    for c in sorted(data["claims"].values(), key=lambda c: (claim_bucket(c), c["claim_id"])):
        if claim_key(c) != key:
            continue
        p = data["papers"].get(c["paper_id"], {})
        review = ("curated" if c.get("tier", "T1") == "T1" else
                  {True: "peer-reviewed", False: "preprint"}.get(c.get("peer_reviewed"), "review status unknown"))
        out.append({"claim_id": c["claim_id"], "tier": c.get("tier", "T1"), "review": review,
                    "published": p.get("published", ""), "bucket": claim_bucket(c),
                    "direction": c["expected_outcome"], "quote": c["source_span"],
                    "title": p.get("title", ""), "url": p.get("url", "")})
    return out
