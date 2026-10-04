"""Scout → targeted search per direction (EXPLORE §1-2). The scout proposes search directions,
never evidence: everything it names, including remembered paper titles, is used only as a search.
Linking claims to directions and the direction gap status are pure functions."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ..literature import TESTBED_DESCRIPTION
from ..llm import LLM
from ..schemas import Claim, DirectionRecord, NicheSpec, RetrievedPaper, ScoutOutput
from ..stats import QUEUE_FIELDS
from . import arxiv, openalex
from .coverage import claim_coverage
from .http import RawCache
from .textsim import cosine, idf, tokens, vector

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
BLOCKS = tuple(sorted(QUEUE_FIELDS))

# Human-written search terms for the deterministic direction of each building block.
BLOCK_TERMS = {
    "activation": ["activation function", "SwiGLU", "squared ReLU"],
    "dropout": ["dropout", "regularization"],
    "norm": ["RMSNorm", "layer normalization"],
    "norm_position": ["Pre-LN", "Post-LN", "normalization placement"],
    "optimizer": ["SGD momentum", "Adam optimizer", "Lion optimizer"],
    "pos_encoding": ["rotary position embedding", "positional encoding"],
    "schedule": ["learning rate schedule", "cosine schedule"],
    "qk_norm": ["QK normalization", "query-key normalization", "cosine attention"],
    "weight_tying": ["weight tying", "tied embeddings"],
    "z_loss_coef": ["z-loss", "output logit regularization"],
    "label_smoothing": ["label smoothing"],
    "warmup_frac": ["learning rate warmup", "warm-up stage"],
    "grad_clip": ["gradient clipping"],
    "init_scale": ["weight initialization", "initialization scale"],
}


def norm_title(t: str) -> str:
    return " ".join(tokens(t))


def title_matches(hint: str, title: str, min_jaccard: float = 0.9) -> bool:
    """A remembered title counts as found only if the API returned (nearly) that exact title."""
    a, b = set(tokens(hint)), set(tokens(title))
    if not a or not b:
        return False
    return norm_title(hint) == norm_title(title) or len(a & b) / len(a | b) >= min_jaccard


def to_records(out: ScoutOutput | None) -> list[DirectionRecord]:
    """Assign ids, drop unknown block names (logged on the record), and add one deterministic
    direction per building block the scout did not mention, so the map always covers what the
    testbed can run."""
    recs: list[DirectionRecord] = []
    for d in (out.directions if out else []):
        good = [b for b in dict.fromkeys(d.building_blocks) if b in BLOCKS]
        recs.append(DirectionRecord(
            direction_id=f"dir_{len(recs) + 1:02d}", origin="scout", title=d.title.strip(), idea=d.idea,
            mechanism=d.mechanism, search_terms=[t.strip() for t in d.search_terms if t.strip()][:6],
            building_blocks=good, dropped_blocks=[b for b in d.building_blocks if b not in BLOCKS],
            title_hints=[{"title": t.strip(), "status": "not_searched", "paper_id": None}
                         for t in d.possibly_related_titles[:3] if t.strip()]))
    mentioned = {b for r in recs for b in r.building_blocks}
    for b in BLOCKS:
        if b not in mentioned:
            recs.append(DirectionRecord(
                direction_id=f"dir_{len(recs) + 1:02d}", origin="deterministic",
                title=f"{b} (building block not mentioned by the scout)",
                idea=f"Change the testbed's {b} setting.", search_terms=BLOCK_TERMS[b],
                building_blocks=[b]))
    return recs


def run_scout(llm: LLM, niche: NicheSpec, n_range: Sequence[int] = (8, 15)
              ) -> tuple[list[DirectionRecord], str]:
    """One LLM call. On failure the deterministic per-block directions still exist."""
    lo, hi = n_range
    user = (f"{TESTBED_DESCRIPTION}\n\nNICHE (chosen by the PI)\nTitle: {niche.title}\n"
            f"Description: {niche.description}\nInclude terms: {', '.join(niche.include_terms)}\n"
            f"Exclude terms: {', '.join(niche.exclude_terms)}\n\n"
            "BUILDING BLOCKS (config fields the testbed can change; up to two together):\n"
            + "\n".join(f"- {b}" for b in BLOCKS) + "\n")

    def validate(o: ScoutOutput) -> None:
        if not lo <= len(o.directions) <= hi:
            raise ValueError(f"return {lo} to {hi} directions")
        for d in o.directions:
            if not 1 <= len([t for t in d.search_terms if t.strip()]):
                raise ValueError(f"direction {d.title!r} needs search_terms")

    out, err = None, ""
    try:
        out, _ = llm.call("scout", (PROMPTS / "scout.md").read_text(), user, ScoutOutput, validate=validate)
    except Exception as e:  # noqa: BLE001 - the search degrades instead of aborting
        err = f"{type(e).__name__}: {e}"[:300]
    return to_records(out), err


def direction_queries(rec: DirectionRecord, n: int) -> list[str]:
    """Up to n arXiv queries from the direction's search terms (one quoted phrase each)."""
    seen, out = set(), []
    for t in rec.search_terms:
        q = '"' + re.sub(r'["\s]+', " ", t).strip() + '"'
        if q.lower() not in seen and len(q) > 2:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


def search_directions(cache: RawCache, niche: NicheSpec, recs: list[DirectionRecord], *,
                      date_to: str, n_queries: int, cap: int, block_cap: int | None = None,
                      openalex_fallback: bool = True) -> list[RetrievedPaper]:
    """Targeted retrieval per direction, with provenance on every paper. Mutates the records
    (queries with counts, title-hint outcomes, paper ids). Directions tied to a building block
    get `block_cap` papers instead of `cap`: those are the methods the testbed can run, and the
    combination gaps need claims about them. When an arXiv direction query fails after retries and
    `openalex_fallback` is on, the same terms are tried on OpenAlex (higher limits); those papers
    are tagged `source: openalex` and go through the same quote check."""
    found: list[RetrievedPaper] = []
    base_cap = cap
    for rec in recs:
        cap = (block_cap or base_cap) if rec.building_blocks else base_cap
        mine: dict[str, RetrievedPaper] = {}
        qs = direction_queries(rec, n_queries)
        per_q = max(cap // max(len(qs), 1), 1)
        for q in qs:
            body = cache.get("arxiv_dir", arxiv.search_url(q, niche, date_to, per_q), "xml")
            if body is not None:
                hits = arxiv.parse_feed(body, f"{rec.direction_id}: {q}")
                source = "arxiv"
            elif openalex_fallback:          # arXiv threw after retries: fall back to OpenAlex search
                oa = cache.get("openalex_dir", openalex.search_url(q, niche, date_to, per_q), "json")
                hits = openalex.parse_search(oa, f"{rec.direction_id}: {q}") if oa else []
                source = "openalex"
            else:
                hits, source = [], "arxiv"
            rec.queries.append({"q": q, "n_results": len(hits), "source": source})
            for p in hits:
                mine.setdefault(p.paper_id, p)
        for hint in rec.title_hints:      # remembered titles: search hints only
            body = cache.get("arxiv_title", arxiv.title_url(hint["title"]), "xml")
            if body is None:              # a failed lookup is NOT evidence the paper does not exist
                hint["status"], hint["paper_id"] = "request_failed", None
                continue
            hits = arxiv.parse_feed(body, f"{rec.direction_id}: title hint")
            match = next((p for p in hits if title_matches(hint["title"], p.title)), None)
            hint["status"], hint["paper_id"] = ("found", match.paper_id) if match else ("not_found", None)
            if match:
                mine.setdefault(match.paper_id, match)
        kept = list(mine.values())[:cap]
        for p in kept:
            p.directions = [rec.direction_id]
        rec.paper_ids = [p.paper_id for p in kept]
        found += kept
    return found


def direction_text(rec: DirectionRecord) -> str:
    return " ".join([rec.title, rec.idea, rec.mechanism, *rec.search_terms])


def link_claims(recs: Sequence[DirectionRecord], claims: Sequence[Claim],
                papers: dict[str, RetrievedPaper], threshold: float = 0.25) -> dict[str, list[str]]:
    """direction_id -> claim ids. A claim belongs to a direction if its paper was found by that
    direction's search AND (the claim maps to one of the direction's building blocks OR TF-IDF
    cosine(claim text, direction text) >= threshold). Hybrid addition: a claim whose paper came
    only from the niche search or the curated set (no direction provenance at all) is linked by
    the building-block mapping alone, so both sources feed one graph."""
    idf_ = idf([f"{c.method} {c.claim}" for c in claims] + [direction_text(r) for r in recs])
    out: dict[str, list[str]] = {}
    for r in recs:
        dv = vector(direction_text(r), idf_)
        ids = []
        for c in claims:
            p = papers.get(c.paper_id)
            block = bool(c.config_change) and next(iter(c.config_change)) in r.building_blocks
            found_by = bool(p) and r.direction_id in p.directions
            unscouted = not p or not p.directions
            sim = cosine(vector(f"{c.method} {c.claim}", idf_), dv)
            if (found_by and (block or sim >= threshold)) or (unscouted and block):
                ids.append(c.claim_id)
        out[r.direction_id] = sorted(ids)
    return out


# Shown verbatim in the UI.
DIRECTION_STATUS_RULE = (
    "covered: a T1/T2 claim with coverage `covers` reports this direction's method in our setting.\n"
    "contested: at least 2 linked claims with opposite direction (improves vs worse / no_worse).\n"
    "partial: covered only by `partial`-coverage claims or only by T3 preprints.\n"
    "open: linked claims exist for the method elsewhere, none covers our setting.\n"
    "unexplored: no linked claims at all (only papers, or nothing found).\n"
    "outside_testbed: the direction touches no building block (shown next to the status above; never run).\n"
    "Context-only claims report no result and do not count as covering."
)


def direction_status(rec: DirectionRecord, linked: Sequence[Claim]) -> dict:
    """Gap status of one direction (pure). Returns {status, outside_testbed, n_claims, n_papers}."""
    directional = [c for c in linked if c.expected_outcome != "context"]
    cov = [(c, claim_coverage(c)) for c in directional]
    signs = {c.expected_outcome for c in directional}
    if not linked:
        status = "unexplored"
    elif any(k == "covers" and c.tier in ("T1", "T2") for c, k in cov):
        status = "covered"
    elif "improves" in signs and signs & {"worse", "no_worse"}:
        status = "contested"
    elif any(k in ("covers", "partial") for _, k in cov):
        status = "partial"
    else:
        status = "open"
    return {"status": status, "outside_testbed": not rec.building_blocks,
            "n_claims": len(linked), "n_papers": len(rec.paper_ids)}
