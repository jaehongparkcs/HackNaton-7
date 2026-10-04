"""Extraction: the LLM transcribes fields from abstracts; code verifies the quote verbatim, maps
the method onto the allowed config space, computes coverage and assigns the tier."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ..llm import LLM
from ..schemas import MECHANISMS, Claim, ExtractedClaim, ExtractionOutput, RetrievedPaper
from ..stats import ALLOWED_CHANGES
from .coverage import coverage_of, tier_of

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
BATCH = 5
MAX_PER_PAPER = 3


def _prompt(batch: Sequence[RetrievedPaper]) -> str:
    return ("ALLOWED CONFIG CHANGES (field=value):\n" + "\n".join(f"- {k}" for k in ALLOWED_CHANGES)
            + "\n\nMECHANISMS (pick one name, or none):\n" + "\n".join(f"- {k}: {v}" for k, v in MECHANISMS.items())
            + "\n\n" + "\n\n".join(f"### PAPER {p.paper_id}\nTitle: {p.title}\nAbstract: {p.abstract}"
                                   for p in batch) + "\n")


def to_claim(ec: ExtractedClaim, paper: RetrievedPaper, n: int, event: str,
             dropped: list[dict]) -> Claim | None:
    """Code-side checks on one extracted claim. Returns None (and logs why) if the quote fails."""
    if not ec.source_span.strip() or ec.source_span not in paper.abstract:
        dropped.append({"paper_id": paper.paper_id, "title": paper.title,
                        "reason": "non-verbatim source_span", "source_span": ec.source_span})
        return None
    change = ec.config_change.strip()
    if change and change not in ALLOWED_CHANGES:
        dropped.append({"paper_id": paper.paper_id, "title": paper.title,
                        "reason": f"config_change {change!r} not in the allowlist: set to null (claim kept as context)"})
        change = ""
    mech, category = ec.mechanism.strip(), ec.mechanism_category
    if mech and mech not in paper.abstract:          # same verbatim rule as the quote; the claim is kept
        dropped.append({"paper_id": paper.paper_id, "title": paper.title,
                        "reason": "non-verbatim mechanism: set to empty (claim kept)", "mechanism": mech})
        mech = ""
    if category == "none" or not mech:               # a category without a verbatim quote is not kept
        if category != "none":
            dropped.append({"paper_id": paper.paper_id, "title": paper.title,
                            "reason": f"mechanism category {category!r} without a verbatim quote: dropped (claim kept)"})
        category, mech = "", ""
    cov = coverage_of(ec.setting)
    s = ec.setting
    field, _, value = change.partition("=")
    return Claim(
        claim_id="x" + re.sub(r"\W+", "_", paper.paper_id) + f"_{n}", paper_id=paper.paper_id,
        dimension=field or "outside_testbed", method=ec.method,
        setting=f"{s.model_family}, {s.task}, scale {s.scale}, {s.evidence} (extracted from the abstract)",
        claim=ec.claim, source_span=ec.source_span, expected_outcome=ec.direction,
        covers_our_setting=cov == "covers",
        coverage_note=f"computed by the coverage rule from the extracted setting fields: {cov}",
        config_change={field: value} if change else None, tier=tier_of(
            curated=False, quote_verified=True, peer_reviewed=paper.peer_reviewed),
        coverage=cov, setting_fields=s, peer_reviewed=paper.peer_reviewed, published=paper.published,
        extraction_event=event, mechanism=mech, mechanism_category=category)


def extract(llm: LLM, papers: Sequence[RetrievedPaper], *, role: str = "extract"
            ) -> tuple[list[Claim], list[dict], list[str]]:
    """Returns (claims, dropped, errors). A failed batch leaves its papers unextracted (T4).

    Batches are independent LLM calls, issued concurrently (`llm.workers`); their results are
    applied in batch order, so the output never depends on which call returned first."""
    system = (PROMPTS / "extract.md").read_text()
    claims: list[Claim] = []
    dropped: list[dict] = []
    errors: list[str] = []
    batches = [list(papers[i:i + BATCH]) for i in range(0, len(papers), BATCH)]

    def run(batch: list[RetrievedPaper]):
        by_id = {p.paper_id: p for p in batch}

        def validate(o: ExtractionOutput) -> None:
            bad = sorted({c.paper_id for c in o.claims} - set(by_id))
            if bad:
                raise ValueError(f"paper_id {bad} not in this batch; use only {sorted(by_id)}")

        try:
            return llm.call(role, system, _prompt(batch), ExtractionOutput, validate=validate), None
        except Exception as e:  # noqa: BLE001 - the search degrades instead of aborting
            return None, e

    workers = min(getattr(llm, "workers", 1), len(batches)) or 1
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(run, batches))
    else:
        results = [run(b) for b in batches]
    for i, (batch, (res, exc)) in enumerate(zip(batches, results)):
        if exc is not None:
            errors.append(f"extraction batch {i}: {type(exc).__name__}: {exc}"[:300])
            continue
        out, event = res
        by_id = {p.paper_id: p for p in batch}
        count: dict[str, int] = {}
        for ec in out.claims:
            paper = by_id[ec.paper_id]
            if count.get(paper.paper_id, 0) >= MAX_PER_PAPER:
                dropped.append({"paper_id": paper.paper_id, "title": paper.title,
                                "reason": f"more than {MAX_PER_PAPER} claims for one paper"})
                continue
            c = to_claim(ec, paper, count.get(paper.paper_id, 0) + 1, event, dropped)
            if c:
                count[paper.paper_id] = count.get(paper.paper_id, 0) + 1
                claims.append(c)
    return claims, dropped, errors


def agreement_with_curated(extracted: Sequence[Claim], curated: Sequence[Claim]) -> dict:
    """Validation: extraction on the curated papers vs the human labels. A curated claim with a
    config_change is matched to an extracted claim of the same paper with the same config_change."""
    from .coverage import counts_as_covered
    mapped = [c for c in curated if c.config_change]
    by_paper: dict[str, list[Claim]] = {}
    for c in extracted:
        by_paper.setdefault(c.paper_id, []).append(c)
    cfg = direction = setting = 0
    rows = []
    for h in mapped:
        m = next((x for x in by_paper.get(h.paper_id, []) if x.config_change == h.config_change), None)
        d_ok = bool(m) and m.expected_outcome == h.expected_outcome
        s_ok = bool(m) and counts_as_covered(m) == counts_as_covered(h)
        cfg, direction, setting = cfg + bool(m), direction + d_ok, setting + s_ok
        from .coverage import claim_coverage
        rows.append({"claim_id": h.claim_id, "config_mapping": bool(m), "direction": d_ok, "setting": s_ok,
                     # the human label is abstract-scoped (data/claims.json note); the model's is computed
                     # by the coverage rule from whatever text it read. Shown so a disagreement can be read.
                     "label_coverage": claim_coverage(h), "model_coverage": claim_coverage(m) if m else None})
    # setting is only comparable where the change was matched: an unmatched label counts as a setting miss
    # in "setting", so "setting_of_matched" is reported beside it (denominator = config_mapping).
    setting_m = sum(r["setting"] for r in rows if r["config_mapping"])
    return {"n": len(mapped), "config_mapping": cfg, "direction": direction, "setting": setting,
            "setting_of_matched": setting_m, "rows": rows}
