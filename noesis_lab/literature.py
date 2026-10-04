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
from .schemas import (
    Claim,
    DirectionRecord,
    Fixture,
    LiteratureOutput,
    LiteratureOutputV2,
    PriorArtResult,
    PriorArtVerdictKind,
    RetrievedPaper,
)
from .search.coverage import claim_coverage, counts_as_covered, review_label
from .stats import gate_decision, gate_outcome, prior_art_verdict

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

    def __init__(self, papers_path: Path, claims_path: Path, meta: dict | None = None):
        doc = json.loads(Path(papers_path).read_text())
        self.source, self.fetched = doc["source"], doc["fetched"]
        self.meta = meta or {"mode": "curated_only"}      # corpus/meta.json when frozen from a search
        self.directions: list[DirectionRecord] = []       # scout + deterministic directions (scout.json)
        self.papers: dict[str, RetrievedPaper] = {p["paper_id"]: RetrievedPaper(**p) for p in doc["papers"]}
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

    @classmethod
    def from_corpus(cls, corpus: Path) -> Snapshot:
        """A frozen session corpus (results/<session>/corpus). No network."""
        mp = Path(corpus) / "meta.json"
        snap = cls(Path(corpus) / "papers.json", Path(corpus) / "claims.json",
                   json.loads(mp.read_text()) if mp.exists() else None)
        sp = Path(corpus) / "scout.json"
        if sp.exists():
            snap.directions = [DirectionRecord(**d) for d in json.loads(sp.read_text())["directions"]]
        return snap

    def add(self, papers: list[RetrievedPaper], claims: list[Claim]) -> None:
        """Papers and claims found by a hypothesis's own targeted search join the session corpus."""
        for p in papers:
            self.papers.setdefault(p.paper_id, p)
        for c in claims:
            if c.source_span not in self.papers[c.paper_id].abstract:
                raise SnapshotError(f"{c.claim_id}: source_span is not verbatim in {c.paper_id}'s abstract")
            self.claims.setdefault(c.claim_id, c)
        self._build_index()

    @property
    def size(self) -> int:
        return len(self.papers)

    @property
    def live(self) -> bool:
        return self.meta.get("mode") == "live_search"

    def not_found_label(self) -> str:
        """The strongest negative statement we make. Never "novel"."""
        if self.live:
            return (f"not found in {self.size} papers retrieved on {self.meta.get('search_date')} "
                    f"for {len(self.meta.get('queries', []))} queries")
        return f"not found in curated snapshot ({self.size} papers)"

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


def _sub_change(claim_change: dict | None, required: dict) -> bool:
    """A claim's `config_change` is part of the candidate's delta: non-empty and every one of its
    field=value pairs appears in the delta. For a single-field candidate this is exact match; for a
    combination it also admits a claim about one of the two fields (which is what lets the tighten
    pass narrow the hypothesis to the untested part). A claim about a different change, or about the
    same field with a different value, cannot decide the verdict."""
    return bool(claim_change) and all(required.get(f) == v for f, v in claim_change.items())


def _claims_block(hits: list[tuple[Claim, float]], snap: Snapshot) -> str:
    out = []
    for c, _ in hits:
        p = snap.papers[c.paper_id]
        out.append(f"[{c.claim_id}] {p.title} (arXiv:{c.paper_id}, {p.published})\n"
                   f"  method: {c.method}\n  setting (per abstract): {c.setting}\n"
                   f"  claim: {c.claim}\n  passage: \"{c.source_span}\"")
    return "\n".join(out)


class LiteratureAgent:
    def __init__(self, llm: LLM, snap: Snapshot, top_k: int = 5, overrides: tuple[str, ...] = (),
                 version: int = 2):
        """`version` 1 = the gate older bundles were recorded with (one nearest claim, kept for
        replay); 2 = the LLM lists all same-comparison claims and code picks the verdict."""
        self.llm, self.snap, self.top_k, self.overrides, self.version = llm, snap, top_k, set(overrides), version
        self.system = (PROMPTS / ("literature_gate.md" if version >= 2 else "literature.md")).read_text()

    def check(self, fixture: Fixture, *, question: str | None = None,
              role: str = "literature", extra_claim_ids: tuple[str, ...] = (),
              contested_ids: tuple[str, ...] = (), require_change: dict[str, str] | None = None) -> PriorArtResult:
        """Prior-art verdict. `question` overrides the text checked (targeted post-hoc checks).

        The LLM judges one thing: does a retrieved claim test the same change? Setting coverage is
        a human-curated flag on the claim; the verdict is derived by `stats.prior_art_verdict`.

        `require_change` (generated candidates only): a claim may decide the verdict only if its
        `config_change` equals this delta. A claim the LLM lists whose change differs or is null
        stays as context, so the gate cannot flip on which "nearest" claim the LLM happens to pick
        (e.g. c01, Pre-LN without warm-up, can no longer reject a Post-LN candidate)."""
        text = question or fixture.statement
        hits = self.snap.search(text, self.top_k)
        have = {c.claim_id for c, _ in hits}
        # the claims a candidate was generated from are always shown to the agent
        hits += [(self.snap.claims[cid], 0.0) for cid in fixture.cited_claim_ids
                 if cid not in have and cid in self.snap.claims and fixture.origin == "generated"]
        have = {c.claim_id for c, _ in hits}        # claims found by this hypothesis's own targeted search
        hits += [(self.snap.claims[cid], 0.0) for cid in extra_claim_ids
                 if cid not in have and cid in self.snap.claims]
        valid_ids = {c.claim_id for c, _ in hits}
        user = (f"{TESTBED_DESCRIPTION}\n\nHypothesis id: {fixture.fixture_id}\n"
                f"Text to check:\n{text}\n\n"
                f"Curated snapshot: {self.snap.size} papers. Retrieved claims:\n"
                f"{_claims_block(hits, self.snap)}\n")

        contested, partial = False, []
        if self.version >= 2:
            def validate2(o: LiteratureOutputV2) -> None:
                bad = sorted(set(o.same_comparison_claim_ids) - valid_ids)
                if bad:
                    raise ValueError(f"claim ids {bad} are not in the retrieved list {sorted(valid_ids)}")

            out, eid = self.llm.call(role, self.system, user, LiteratureOutputV2, validate=validate2)
            ids = sorted(set(out.same_comparison_claim_ids))
            deciding, partial = ids, []
            if require_change is not None:
                # Only a claim about the WHOLE change decides the verdict: a claim about one method
                # is not prior art for two methods tested together. Claims about a part (strict
                # subset) that would otherwise be prior art are returned separately; the tighten
                # pass uses them to narrow the combination to its untested field.
                deciding = [i for i in ids if self.snap.claims[i].config_change == require_change]
                partial = sorted(i for i in ids if i not in deciding
                                 and _sub_change(self.snap.claims[i].config_change, require_change)
                                 and counts_as_covered(self.snap.claims[i]))
            verdict, claim, gate, contested = gate_decision(
                [self.snap.claims[i] for i in deciding], contested_ids=contested_ids,
                overridden=fixture.candidate_key in self.overrides)
            same = bool(deciding)
        else:
            def validate(o: LiteratureOutput) -> None:
                if o.same_comparison and o.claim_id not in valid_ids:
                    raise ValueError(f"claim_id {o.claim_id!r} must be one of {sorted(valid_ids)} "
                                     "when same_comparison is true")
                if not o.same_comparison and o.claim_id:
                    raise ValueError("claim_id must be \"\" when same_comparison is false")

            out, eid = self.llm.call(role, self.system, user, LiteratureOutput, validate=validate)
            ids, same = [], out.same_comparison
            claim = self.snap.claims.get(out.claim_id) if out.claim_id else None
            verdict = prior_art_verdict(same, claim_coverage(claim) if claim else None)
            gate = gate_outcome(verdict, claim.tier if claim else None,
                                overridden=fixture.candidate_key in self.overrides)
        paper = self.snap.papers[claim.paper_id] if claim else None
        cov = claim_coverage(claim) if claim else None
        n = self.snap.size
        where = "curated snapshot" if not self.snap.live else f"{n} retrieved papers"
        badge = (f"{claim.tier}, {review_label(paper.peer_reviewed) if claim.tier != 'T1' else 'curated'}, "
                 f"{paper.published}") if claim else ""
        label = {
            PriorArtVerdictKind.known: f"KNOWN IN CORPUS (arXiv:{claim.paper_id}; {badge})" if claim else "",
            PriorArtVerdictKind.setting_untested:
                f"method found (arXiv:{claim.paper_id}; {badge}); this setting not found in {where}"
                + (f" ({n} papers)" if not self.snap.live else "") if claim else "",
            PriorArtVerdictKind.not_found: self.snap.not_found_label(),
        }[verdict]
        return PriorArtResult(
            verdict=verdict, claim_id=claim.claim_id if claim else None,
            paper_id=paper.paper_id if paper else None, paper_title=paper.title if paper else None,
            paper_url=paper.url if paper else None,
            passage=claim.source_span if claim else None,    # from the snapshot, never the LLM
            rationale=out.rationale, snapshot_size=n, label=label, event_id=eid,
            same_comparison=same,
            covers_our_setting=claim.covers_our_setting if claim else None,
            coverage_note=claim.coverage_note if claim else None,
            coverage=cov, tier=claim.tier if claim else None, gate=gate,
            same_comparison_claim_ids=ids, partial_overlap_claim_ids=partial, contested=contested)
