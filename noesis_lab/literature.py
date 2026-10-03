"""Curated literature snapshot + the Literature Agent.

The snapshot is tiny on purpose (BUILD_PLAN s4). The UI calls it a *curated literature snapshot*
and never says "novel": the strongest negative statement is "not found in curated snapshot (N papers)".
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .llm import LLM
from .schemas import Claim, Fixture, LiteratureOutput, Paper, PriorArtResult, PriorArtVerdictKind
from .stats import prior_art_verdict

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

TESTBED_DESCRIPTION = (
    "Our testbed: a tiny character-level Transformer language model (<1M parameters) trained on "
    "TinyShakespeare for a fixed 2,000-step budget (~0.5-1 min on Apple Silicon); "
    "metric = validation cross-entropy."
)


class SnapshotError(RuntimeError):
    pass


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class Snapshot:
    """Papers + claims. Loading FAILS if any claim span is not a verbatim substring of its abstract."""

    def __init__(self, papers_path: Path, claims_path: Path):
        doc = json.loads(Path(papers_path).read_text())
        self.source, self.fetched = doc["source"], doc["fetched"]
        self.papers: dict[str, Paper] = {p["paper_id"]: Paper(**p) for p in doc["papers"]}
        self.claims: dict[str, Claim] = {}
        for raw in json.loads(Path(claims_path).read_text())["claims"]:
            c = Claim(**raw)
            if c.paper_id not in self.papers:
                raise SnapshotError(f"{c.claim_id}: unknown paper {c.paper_id}")
            if c.source_span not in self.papers[c.paper_id].abstract:
                raise SnapshotError(f"{c.claim_id}: source_span is not verbatim in "
                                    f"{c.paper_id}'s abstract")
            self.claims[c.claim_id] = c
        self._build_index()

    @property
    def size(self) -> int:
        return len(self.papers)

    def _doc_text(self, c: Claim) -> str:
        return " ".join([c.method, c.dimension, c.setting, c.claim, self.papers[c.paper_id].title])

    def _build_index(self) -> None:
        self._tf = {cid: Counter(_tokens(self._doc_text(c))) for cid, c in self.claims.items()}
        df = Counter(t for tf in self._tf.values() for t in tf)
        n = len(self.claims)
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Claim, float]]:
        """Deterministic TF-IDF cosine. Ties broken by claim_id. Retrieval only proposes
        candidates; it never decides a verdict."""
        q = Counter(_tokens(query))
        qv = {t: c * self._idf[t] for t, c in q.items() if t in self._idf}
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scored = []
        for cid, tf in self._tf.items():
            dv = {t: c * self._idf[t] for t, c in tf.items()}
            dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            dot = sum(v * dv.get(t, 0.0) for t, v in qv.items())
            scored.append((round(dot / (qn * dn), 6), cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(self.claims[cid], s) for s, cid in scored[:k]]

    def dimensions(self) -> list[str]:
        return sorted({c.dimension for c in self.claims.values()})


def _claims_block(hits: list[tuple[Claim, float]], snap: Snapshot) -> str:
    out = []
    for c, _ in hits:
        p = snap.papers[c.paper_id]
        out.append(f"[{c.claim_id}] {p.title} (arXiv:{c.paper_id})\n"
                   f"  method: {c.method}\n  setting (per abstract): {c.setting}\n"
                   f"  claim: {c.claim}\n  passage: \"{c.source_span}\"")
    return "\n".join(out)


class LiteratureAgent:
    def __init__(self, llm: LLM, snap: Snapshot, top_k: int = 5):
        self.llm, self.snap, self.top_k = llm, snap, top_k
        self.system = (PROMPTS / "literature.md").read_text()

    def check(self, fixture: Fixture, *, question: str | None = None,
              role: str = "literature") -> PriorArtResult:
        """Prior-art verdict. `question` overrides the text checked (targeted post-hoc checks).

        The LLM judges one thing: does a retrieved claim test the same change? Setting coverage is
        a human-curated flag on the claim; the verdict is derived by `stats.prior_art_verdict`."""
        text = question or fixture.statement
        hits = self.snap.search(text, self.top_k)
        have = {c.claim_id for c, _ in hits}
        # the claims a candidate was generated from are always shown to the agent
        hits += [(self.snap.claims[cid], 0.0) for cid in fixture.cited_claim_ids
                 if cid not in have and cid in self.snap.claims and fixture.origin == "generated"]
        valid_ids = {c.claim_id for c, _ in hits}
        user = (f"{TESTBED_DESCRIPTION}\n\nHypothesis id: {fixture.fixture_id}\n"
                f"Text to check:\n{text}\n\n"
                f"Curated snapshot: {self.snap.size} papers. Retrieved claims:\n"
                f"{_claims_block(hits, self.snap)}\n")

        def validate(o: LiteratureOutput) -> None:
            if o.same_comparison and o.claim_id not in valid_ids:
                raise ValueError(f"claim_id {o.claim_id!r} must be one of {sorted(valid_ids)} "
                                 "when same_comparison is true")
            if not o.same_comparison and o.claim_id:
                raise ValueError("claim_id must be \"\" when same_comparison is false")

        out, eid = self.llm.call(role, self.system, user, LiteratureOutput, validate=validate)
        claim = self.snap.claims.get(out.claim_id) if out.claim_id else None
        paper = self.snap.papers[claim.paper_id] if claim else None
        verdict = prior_art_verdict(out.same_comparison, claim.covers_our_setting if claim else None)
        n = self.snap.size
        label = {
            PriorArtVerdictKind.known: f"KNOWN IN CORPUS (arXiv:{claim.paper_id})" if claim else "",
            PriorArtVerdictKind.setting_untested:
                f"method found (arXiv:{claim.paper_id}); this setting not found in curated snapshot "
                f"({n} papers)" if claim else "",
            PriorArtVerdictKind.not_found: f"not found in curated snapshot ({n} papers)",
        }[verdict]
        return PriorArtResult(
            verdict=verdict, claim_id=claim.claim_id if claim else None,
            paper_id=paper.paper_id if paper else None, paper_title=paper.title if paper else None,
            paper_url=paper.url if paper else None,
            passage=claim.source_span if claim else None,    # from the snapshot, never the LLM
            rationale=out.rationale, snapshot_size=n, label=label, event_id=eid,
            same_comparison=out.same_comparison,
            covers_our_setting=claim.covers_our_setting if claim else None,
            coverage_note=claim.coverage_note if claim else None)
