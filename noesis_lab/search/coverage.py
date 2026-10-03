"""Coverage rule, trust tiers and the T4 overlap check. Pure code: one rule for every paper."""
from __future__ import annotations

from collections.abc import Sequence

from ..schemas import Claim, Coverage, RetrievedPaper, SettingFields, Tier
from .textsim import cosine, idf, vector

OUR_SETTING = "transformer, char_language_modeling, tiny, step-budgeted training"

# Shown verbatim in the UI.
COVERAGE_RULE_TEXT = (
    f"Our setting: {OUR_SETTING}.\n"
    "covers: model_family ∈ {transformer, general} AND task ∈ {char_language_modeling, "
    "language_modeling, general} AND scale ∈ {tiny, small}.\n"
    "partially covers: same model_family and task condition AND scale ∈ {medium, large, unspecified}.\n"
    "does not cover: anything else.\n"
    "For overlap / prior art, covers OR partially covers counts as covered (we avoid rediscovery). "
    "For evidence relations (agrees / contradicts), only covers counts.\n"
    "Curated (T1) claims carry a human covers / does-not-cover flag instead of extracted fields."
)

_FAMILY = {"transformer", "general"}
_TASK = {"char_language_modeling", "language_modeling", "general"}


def coverage_of(s: SettingFields) -> Coverage:
    if s.model_family in _FAMILY and s.task in _TASK:
        return "covers" if s.scale in ("tiny", "small") else "partial"
    return "none"


def claim_coverage(c: Claim) -> Coverage:
    """Stored coverage for auto claims; the human flag for curated ones."""
    if c.coverage is not None:
        return c.coverage
    return "covers" if c.covers_our_setting else "none"


def counts_as_covered(c: Claim) -> bool:
    """Overlap / novelty question. Errors lean toward 'covered'."""
    return claim_coverage(c) in ("covers", "partial")


def tier_of(*, curated: bool, quote_verified: bool, peer_reviewed: bool | None) -> Tier:
    if curated:
        return "T1"
    if not quote_verified:
        return "T4"
    return "T2" if peer_reviewed else "T3"       # unknown review status is treated as a preprint


def review_label(peer_reviewed: bool | None) -> str:
    return {True: "peer-reviewed", False: "preprint"}.get(peer_reviewed, "review status unknown")


T4_THRESHOLD = 0.35


def candidate_text(field: str, value: str, supporting: Sequence[Claim]) -> str:
    """The candidate-specific text for the overlap check: the change plus what its supporting
    claims say. The templated hypothesis sentence is NOT used: its boilerplate ("tiny
    character-level Transformer ...") matches every on-niche abstract equally."""
    return " ".join([field, value, *(f"{c.method} {c.claim}" for c in supporting)])


def t4_overlaps(text: str, papers: Sequence[RetrievedPaper], threshold: float = T4_THRESHOLD
                ) -> list[tuple[str, float]]:
    """Unextracted (T4) papers whose abstract is textually close to a candidate. IDF is computed
    over the whole corpus; only T4 papers are scored. A hit holds the candidate for PI review; it
    is never silently treated as untested. Threshold tuned on the curated set (README)."""
    docs = [f"{p.title} {p.abstract}" for p in papers]
    idf_ = idf(docs)
    q = vector(text, idf_)
    hits = [(p.paper_id, cosine(q, vector(d, idf_))) for p, d in zip(papers, docs, strict=True)
            if p.tier == "T4"]
    return sorted(((pid, s) for pid, s in hits if s >= threshold), key=lambda h: (-h[1], h[0]))
