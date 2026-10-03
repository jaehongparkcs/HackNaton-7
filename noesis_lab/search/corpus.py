"""Search live, then freeze. `build_corpus` runs the whole pipeline once and writes
`<bundle>/corpus/`; sessions, replay and rederive read only that folder (no network)."""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml

from ..config import load_config, path_of
from ..llm import LLM
from ..schemas import Claim, NicheSpec, RetrievedPaper
from . import arxiv, openalex
from .coverage import COVERAGE_RULE_TEXT
from .extract import agreement_with_curated, extract
from .http import RawCache, urllib_fetch
from .plan import plan_queries
from .rank import dedupe, filter_excluded, rank
from .scout import run_scout, search_directions
from .textsim import pca_2d

CURATED_ONLY = "curated snapshot only"


def load_niche(path: str | Path | None = None) -> NicheSpec:
    return NicheSpec(**yaml.safe_load(Path(path or path_of("niche")).read_text()))


def _curated() -> tuple[list[RetrievedPaper], list[Claim], dict]:
    doc = json.loads(path_of("papers").read_text())
    papers = [RetrievedPaper(**p, source="curated", tier="T1", peer_reviewed=None) for p in doc["papers"]]
    claims = [Claim(**c) for c in json.loads(path_of("claims").read_text())["claims"]]
    return papers, claims, doc


def _write(out: Path, niche: NicheSpec, papers: list[RetrievedPaper], claims: list[Claim],
           dropped: list[dict], meta: dict, *, source: str, fetched: str,
           query_plan: dict | None = None, scout: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    xy = pca_2d([f"{p.title} {p.abstract}" for p in papers])
    for p, pos in zip(papers, xy, strict=True):
        p.map_xy = pos
    tiers = Counter(c.tier for c in claims)
    tiers["T4"] = sum(1 for p in papers if p.tier == "T4")
    meta = {**meta, "niche": niche.model_dump(), "papers": len(papers), "claims": len(claims),
            "tiers": {t: tiers.get(t, 0) for t in ("T1", "T2", "T3", "T4")},
            "coverage_rule": COVERAGE_RULE_TEXT}
    (out / "niche.yaml").write_text(yaml.safe_dump(niche.model_dump(), sort_keys=False))
    (out / "query_plan.json").write_text(json.dumps(query_plan or {"plan": None, "queries": []}, indent=2) + "\n")
    (out / "papers.json").write_text(json.dumps(
        {"source": source, "fetched": fetched, "papers": [p.model_dump() for p in papers]},
        indent=2, ensure_ascii=False) + "\n")
    (out / "claims.json").write_text(json.dumps(
        {"note": "T1 = human-curated. T2/T3 = auto-extracted: quote verified verbatim by code, coverage "
                 "computed by the written rule. Tiers and coverage are never set by an LLM.",
         "claims": [c.model_dump() for c in claims]}, indent=2, ensure_ascii=False) + "\n")
    (out / "scout.json").write_text(json.dumps(scout or {"directions": [], "error": ""}, indent=2,
                                               ensure_ascii=False) + "\n")
    (out / "dropped.json").write_text(json.dumps(dropped, indent=2, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def freeze_curated(out: Path, niche: NicheSpec, *, degraded: list[str] | None = None,
                   extra: dict | None = None, query_plan: dict | None = None) -> dict:
    """No live search (or a failed one): the corpus is the curated snapshot, labeled as such."""
    papers, claims, doc = _curated()
    meta = {"mode": "curated_only", "label": CURATED_ONLY, "search_date": doc["fetched"],
            "queries": [], "degraded": degraded or [], **(extra or {})}
    return _write(out, niche, papers, claims, [], meta, source=doc["source"], fetched=doc["fetched"],
                  query_plan=query_plan)


def build_corpus(out: Path, niche: NicheSpec, llm: LLM, *, today: str,
                 fetch: Callable[[str], str] = urllib_fetch,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    """NicheSpec → query plan → retrieve → dedupe/filter/rank → extract → verify/map/cover/tier →
    freeze. Failures degrade the search and are recorded; they never abort the session."""
    cfg = load_config()["lit_search"]
    out.mkdir(parents=True, exist_ok=True)
    cache = RawCache(out / "raw", fetch, cfg["arxiv_min_interval_s"], sleep)
    date_to = niche.date_to or today
    degraded: list[str] = []
    cur_papers, cur_claims, _ = _curated()

    plan, queries, err = plan_queries(llm, niche, cfg["max_queries"])
    if err:
        degraded.append(f"query plan failed ({err}); deterministic queries only")
    found: list[RetrievedPaper] = []
    if niche.seed_papers:
        body = cache.get("arxiv_seeds", arxiv.id_list_url(niche.seed_papers), "xml")
        found += arxiv.parse_feed(body, "seed_papers") if body else []
    for q in queries:
        body = cache.get("arxiv", arxiv.search_url(q["q"], niche, date_to, cfg["per_query"]), "xml")
        hits = arxiv.parse_feed(body, q["q"]) if body else []
        q["n_results"] = len(hits)
        found += hits
    # Hybrid ideation: the scout's per-direction searches feed the same corpus as the niche search.
    ecfg = load_config().get("explore", {})
    recs, scout_err = [], ""
    if ecfg.get("enabled"):
        recs, scout_err = run_scout(llm, niche, ecfg["scout_directions"])
        if scout_err:
            degraded.append(f"scout failed ({scout_err}); deterministic per-block directions only")
        found += search_directions(cache, niche, recs, date_to=date_to,
                                   n_queries=ecfg["per_direction_queries"], cap=ecfg["per_direction_cap"],
                                   block_cap=ecfg.get("per_block_direction_cap"))
    degraded += [f"{f['kind']} request failed after retry: {f['error']}" for f in cache.failures]
    plan_doc = {"plan": plan.model_dump() if plan else None, "queries": queries}
    n_retrieved = len({p.paper_id for p in found})

    def stats_(**kw) -> dict:
        return {"llm_calls": llm.n_calls, "llm_cost_usd": round(llm.cost_usd, 4), **kw}

    if n_retrieved < cfg["min_papers"]:
        degraded.append(f"only {n_retrieved} papers retrieved (< {cfg['min_papers']}): "
                        "fell back to the curated snapshot")
        cache.write_index()
        return freeze_curated(out, niche, degraded=degraded, query_plan=plan_doc,
                              extra=stats_(queries=queries, search_attempted=today))

    papers, dup = dedupe([*cur_papers, *found])
    papers, excl = filter_excluded(papers, niche)
    oa = {"enabled": bool(cfg["openalex"]), "matched": 0}
    if cfg["openalex"]:
        n_fail = len(cache.failures)
        oa["matched"] = openalex.enrich(papers, cache)
        if len(cache.failures) > n_fail:
            degraded.append("OpenAlex request failed: review status unknown for some papers")
    papers, ranked_out = rank(papers, niche, date_to=date_to,
                              max_papers=ecfg["max_total_papers"] if recs else None)
    kept_ids = {p.paper_id for p in papers}
    for r in recs:                      # provenance survives dedupe: read it back from the papers
        r.paper_ids = sorted(p.paper_id for p in papers if r.direction_id in p.directions)
    hints = [h for r in recs for h in r.title_hints]
    explore = {"enabled": bool(recs), "directions": len(recs),
               "scout_directions": sum(r.origin == "scout" for r in recs),
               "title_hints_found": sum(h["status"] == "found" for h in hints),
               "title_hints_not_found": sum(h["status"] == "not_found" for h in hints),
               "papers_from_directions": sum(1 for p in papers if p.directions and p.paper_id in kept_ids)}

    auto = [p for p in papers if p.source != "curated"]
    claims, ext_dropped, errors = extract(llm, auto)
    degraded += errors
    with_claims = {c.paper_id for c in claims}
    for p in auto:
        p.tier = ("T2" if p.peer_reviewed else "T3") if p.paper_id in with_claims else "T4"

    if recs:      # health check of the graph the gap formulas will run on (ABC, link and bridge gaps
        # need mechanism nodes shared by two or more runnable methods)
        from .. import gaps
        shared = gaps.shared_mechanisms(gaps.build_graph([*cur_claims, *claims]))
        explore["shared_mechanisms"] = shared
        explore["claims_with_mechanism"] = sum(1 for c in claims if c.mechanism_category)
        if len(shared) < ecfg["min_shared_mechanisms"]:
            degraded.append(f"thin mechanism graph: only {len(shared)} mechanism node(s) are linked to 2 or more "
                            f"runnable methods (want >= {ecfg['min_shared_mechanisms']}); ABC, combination and "
                            "bridge gaps will be sparse. Fix before recording.")

    validation = None
    if cfg.get("validate_extraction"):
        v_claims, _, v_err = extract(llm, cur_papers, role="extract_validation")
        validation = agreement_with_curated(v_claims, cur_claims) if not v_err else {"error": v_err}

    cache.write_index()
    dropped = ([{**d, "stage": "dedupe"} for d in dup] + [{**d, "stage": "filter"} for d in excl]
               + [{**d, "stage": "rank"} for d in ranked_out] + [{**d, "stage": "extract"} for d in ext_dropped])
    meta = stats_(
        explore=explore, mode="live_search", label=f"live search, {len(papers)} papers retrieved on {today} for "
                                  f"{len(queries)} queries",
        search_date=today, queries=queries, degraded=degraded, openalex=oa, validation=validation,
        counts={"retrieved": n_retrieved, "duplicates": len(dup), "excluded": len(excl),
                "ranked_out": len(ranked_out), "kept": len(papers), "curated": len(cur_papers)},
        extraction={"papers": len(auto), "claims_kept": len(claims),
                    "dropped_non_verbatim": sum(1 for d in ext_dropped if d["reason"].startswith("non-verbatim")),
                    "nulled_config_change": sum(1 for d in ext_dropped if "allowlist" in d["reason"]),
                    "papers_without_claims": sum(1 for p in auto if p.tier == "T4")})
    return _write(out, niche, papers, [*cur_claims, *claims], dropped, meta,
                  source="curated snapshot + live arXiv search (abstracts only)", fetched=today,
                  query_plan=plan_doc,
                  scout={"directions": [r.model_dump() for r in recs], "error": scout_err})


def load_meta(corpus: Path) -> dict:
    p = corpus / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {"mode": "curated_only", "label": CURATED_ONLY}


def copy_corpus(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
