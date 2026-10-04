"""Deep read pipeline (DEEP_READ §2–§4, §6): select → fetch → parse → extract → verify → derive →
freeze into <corpus>/deep/. Runs once, before the session; the session and replay read only the
frozen folder. LLMs transcribe facts (each with a verbatim quote and its section) and propose a typed
delta for an author-stated limitation; code checks every quote, maps settings to coverage, maps
recipes to the allowed multipliers and decides everything else.

The bundle stores URL, served version, the HTML's SHA256, section names with lengths and the
quoted spans. Never the full text (licensing): the raw HTML stays in work/cache/html/."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .. import gaps, stats
from ..config import ROOT, load_config, protocol
from ..literature import TESTBED_DESCRIPTION, Snapshot
from ..llm import LLM
from ..schemas import (
    MECHANISMS,
    Claim,
    DeepFindings,
    ExperimentConfig,
    RetrievedPaper,
    SettingFields,
    StatedGapProposal,
)
from ..search.coverage import coverage_of, is_derived, tier_of
from ..search.extract import agreement_with_curated, extract
from ..search.http import RawCache, urllib_fetch
from .html import SECTIONS, norm, parse_paper, quote_in
from .select import select_papers

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
PRIORITY = ("experimental_setup", "appendix_setup", "results", "limitations", "conclusion", "method",
            "introduction", "abstract")
FIELDS = ("setting", "recipes", "results", "mechanisms", "limitations", "small_scale_evidence")
CONCRETE = re.compile(r"\d|small|large|scale|character|char-level|batch|longer|shorter|dataset|task|step|"
                      r"token|width|depth|layer|size", re.I)
CONCRETE_RULE = ("A limitation 'names a concrete untested condition' (plausibility × 1.0, else × 0.5) when its "
                 "quote mentions a number or one of: small, large, scale, character, char-level, batch, longer, "
                 "shorter, dataset, task, step, token, width, depth, layer, size.")
OPTIMIZER_METHODS = ("optimizer=lion", "optimizer=sgd_momentum")


# ----------------------------------------------------------------------------- numbers + scale
def numbers_in(text: str) -> list[float]:
    """Numbers written in a quote: 3e-4, 3 × 10^{-4} (also LaTeX \\times), 1.3B, 3-10x, 0.1."""
    t = text.replace("−", "-")
    out = []
    for a, e in re.findall(r"(\d+(?:\.\d+)?)\s*(?:×|\\times|\*)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?", t):
        out.append(float(a) * 10 ** int(e))
    out += [float(x) for x in re.findall(r"\d+(?:\.\d+)?[eE][-+]?\d+", t)]
    out += [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", t)]
    return out


def _supported(v: float | None, nums: list[float]) -> bool:
    if v is None or v <= 0:
        return False
    return any(abs(v - n) <= 0.01 * n or abs(1 / v - n) <= 0.01 * n for n in nums if n > 0)


def nearest(v: float, allowed: tuple[float, ...]) -> float:
    return min(allowed, key=lambda a: (abs(math.log(v) - math.log(a)), a))


_UNITS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def params_of(stated: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|thousand|m|million|b|billion)\b", stated or "", re.I)
    return float(m.group(1)) * _UNITS[m.group(2).lower()] if m else None


def scale_of(stated: str) -> str:
    """The extraction prompt's buckets: tiny < 10M, small < 100M, medium < 1B, large otherwise."""
    n = params_of(stated)
    if n is None:
        return "unspecified"
    return "tiny" if n < 1e7 else "small" if n < 1e8 else "medium" if n < 1e9 else "large"


# ----------------------------------------------------------------------------- fetch
def html_sources(paper_id: str) -> list[tuple[str, str]]:
    return [("arxiv_html", f"https://arxiv.org/html/{paper_id}"),
            ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{paper_id}")]


def fetch_paper(paper_id: str, cache: RawCache) -> dict:
    """arXiv native HTML, then ar5iv; otherwise unavailable (no PDF parsing)."""
    for source, url in html_sources(paper_id):
        body = cache.get(f"html_{source}", url, "html")
        if not body:
            continue
        parsed = parse_paper(body, paper_id)
        if len([s for s in parsed.sections if s != "abstract"]) >= 1:
            return {"source": source, "url": url, "html_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "version": parsed.version, "parsed": parsed}
    return {"source": "unavailable", "url": None, "html_sha256": None, "version": None, "parsed": None}


# ----------------------------------------------------------------------------- extraction
def _prompt(p: RetrievedPaper, sections: dict[str, str], abstract_claims: list[Claim], budget_chars: int) -> str:
    head = ("ALLOWED CONFIG CHANGES (field=value):\n" + "\n".join(f"- {k}" for k in stats.ALLOWED_CHANGES)
            + "\n\nMECHANISMS (pick one name, or none):\n" + "\n".join(f"- {k}: {v}" for k, v in MECHANISMS.items())
            + f"\n\nPAPER {p.paper_id}: {p.title}\nClaims already extracted from its abstract:\n"
            + ("\n".join(f"- [{c.claim_id}] {c.claim} (config {c.config_change or '-'}, {c.expected_outcome})"
                         for c in abstract_claims) or "- none") + "\n")
    body, left = [], budget_chars
    for name in PRIORITY:
        text = sections.get(name, "")
        if not text or left <= 0:
            continue
        chunk = text[:left]
        body.append(f"## SECTION {name}\n{chunk}")
        left -= len(chunk)
    return head + "\n" + "\n\n".join(body) + "\n"


def verify_findings(f: DeepFindings, sections: dict[str, str]) -> tuple[dict, list[dict]]:
    """Keep only items whose quote is verbatim in the named section; normalize method fields."""
    kept: dict[str, list[dict]] = {k: [] for k in FIELDS}
    dropped: list[dict] = []
    for fname in FIELDS:
        for item in getattr(f, fname):
            d = item.model_dump()
            why = ""
            if d["section"] not in sections:
                why = f"section {d['section']} not found in the full text"
            elif not quote_in(d["quote"], sections[d["section"]]):
                why = "quote not verbatim in that section"
            elif "method" in d and d["method"] and d["method"] not in stats.ALLOWED_CHANGES:
                if fname == "recipes":
                    why = f"method {d['method']} is not an allowed config change"
                else:
                    d["method"] = ""
            if not why and fname == "mechanisms" and d["mechanism_category"] not in MECHANISMS:
                why = "no listed mechanism"
            if why:
                dropped.append({"field": fname, "quote": d["quote"], "section": d["section"], "reason": why})
            else:
                d["quote"] = norm(d["quote"])
                kept[fname].append(d)
    return kept, dropped


# ----------------------------------------------------------------------------- deriving facts (code)
def setting_override(findings: dict) -> dict | None:
    s = (findings.get("setting") or [None])[0]
    if not s:
        return None
    sf = SettingFields(model_family=s["model_family"], task=s["task"], scale=scale_of(s["parameter_count"]),
                       evidence="empirical")
    return {"setting_fields": sf.model_dump(), "parameter_count": s["parameter_count"], "dataset": s["dataset"],
            "training_steps": s["training_steps"], "batch_size": s["batch_size"], "quote": s["quote"],
            "section": s["section"]}


def merge_setting(old: SettingFields | None, new: dict) -> SettingFields:
    """Full-text fields override abstract fields, field by field, unless the full text says unspecified."""
    n = SettingFields(**new)
    if old is None:
        return n
    pick = {k: (getattr(n, k) if getattr(n, k) != "unspecified" else getattr(old, k))
            for k in ("model_family", "task", "scale")}
    return SettingFields(**pick, evidence=old.evidence)


def derive_claims(p: RetrievedPaper, findings: dict, override: dict | None, abstract: list[Claim],
                  small_max: float) -> tuple[list[Claim], list[dict]]:
    """Full-text claims: directional results, small-scale evidence, mechanism and limitation edges.
    Tier is auto-extracted (T2/T3) even for a curated paper: the quote is LLM-transcribed."""
    tier = tier_of(curated=False, quote_verified=True, peer_reviewed=p.peer_reviewed)
    base_sf = SettingFields(**override["setting_fields"]) if override else next(
        (c.setting_fields for c in abstract if c.setting_fields), None) or SettingFields(
        model_family="unspecified", task="unspecified", scale="unspecified", evidence="empirical")
    mech = {m["method"]: m for m in findings.get("mechanisms", []) if m["method"]}
    out, dropped, used_mech = [], [], set()

    def mk(cid: str, method: str, outcome: str, sf: SettingFields, item: dict, claim: str, setting: str,
           mech_item: dict | None = None) -> Claim:
        f, v = method.split("=", 1) if method else ("context", "")
        return Claim(claim_id=cid, paper_id=p.paper_id, dimension=f, method=method or "context", setting=setting,
                     claim=claim, source_span=item["quote"], expected_outcome=outcome,
                     config_change={f: v} if method else None, tier=tier, coverage=coverage_of(sf),
                     setting_fields=sf, peer_reviewed=p.peer_reviewed, published=p.published,
                     mechanism=mech_item["quote"] if mech_item else "",
                     mechanism_category=mech_item["mechanism_category"] if mech_item else "",
                     span_source="full_text", setting_source="full_text" if override else "",
                     section=item["section"],
                     coverage_note=f"full text ({item['section']}): computed by the coverage rule from the "
                                   f"{'full-text' if override else 'abstract'} setting")
    for i, r in enumerate(findings.get("results", []), 1):
        if r["method"] and r["direction"] != "context":
            m = mech.get(r["method"])
            if m:
                used_mech.add(r["method"])
            out.append(mk(f"deep_{p.paper_id}_r{i}", r["method"], r["direction"], base_sf, r,
                          f"full text: {r['direction']} ({r['section']})", r["setting_note"] or "as in the paper", m))
    for i, s in enumerate(findings.get("small_scale_evidence", []), 1):
        n = params_of(s["parameter_count"])
        if not s["method"] or not (s["char_level"] or (n is not None and n <= small_max)):
            dropped.append({"field": "small_scale_evidence", "quote": s["quote"], "section": s["section"],
                            "reason": "not about an allowed change, or not at <= the small-scale size / character level"})
            continue
        sf = SettingFields(model_family=base_sf.model_family if base_sf.model_family != "unspecified" else "transformer",
                           task="char_language_modeling" if s["char_level"] else base_sf.task,
                           scale=scale_of(s["parameter_count"]) if n is not None else "tiny", evidence="empirical")
        out.append(mk(f"deep_{p.paper_id}_s{i}", s["method"], s["direction"], sf, s,
                      f"full text, small-scale experiment: {s['direction']} ({s['parameter_count']})",
                      f"{s['parameter_count']} parameters" + (", character-level" if s["char_level"] else "")))
    for i, m in enumerate(findings.get("mechanisms", []), 1):
        if m["method"] and m["method"] not in used_mech:
            out.append(mk(f"deep_{p.paper_id}_m{i}", m["method"], "context", base_sf, m,
                          f"full text: mechanism {m['mechanism_category']}", "as in the paper", m))
    for i, lim in enumerate(findings.get("limitations", []), 1):
        out.append(mk(f"deep_{p.paper_id}_l{i}", lim["method"], "context", base_sf, lim,
                      "full text: author-stated limitation", "as in the paper"))
    return out, dropped


def map_recipe(items: list[dict]) -> dict | None:
    """lr / weight-decay recipe for one method from one paper → nearest allowed multipliers. A ratio
    stated as a range takes its geometric middle. Every number used must appear in the quote."""
    out: dict = {}
    for it in items:
        if it["hyperparameter"] not in ("lr", "weight_decay"):
            continue
        nums = numbers_in(it["quote"])
        lo, hi = it["ratio_to_adamw_min"], it["ratio_to_adamw_max"]
        ratio, how = None, ""
        if lo and hi and _supported(lo, nums) and _supported(hi, nums):
            ratio, how = math.sqrt(lo * hi), f"ratio {lo:g}–{hi:g} to AdamW (geometric middle)"
        elif lo and not hi and _supported(lo, nums):
            ratio, how = lo, f"ratio {lo:g} to AdamW"
        elif it["value"] and it["adamw_value"] and _supported(it["value"], nums) and _supported(it["adamw_value"], nums):
            ratio, how = it["value"] / it["adamw_value"], f"{it['value']:g} vs AdamW {it['adamw_value']:g}"
        if ratio is None or ratio <= 0:
            continue
        key = "lr_mult" if it["hyperparameter"] == "lr" else "wd_mult"
        allowed = ExperimentConfig.LR_MULTS if key == "lr_mult" else ExperimentConfig.WD_MULTS
        if key not in out:
            out[key] = nearest(ratio, allowed)
            out[f"{key}_source"] = {"stated": it["value_as_stated"], "ratio": round(ratio, 6), "how": how,
                                    "quote": it["quote"], "section": it["section"]}
    if not out:
        return None
    out.setdefault("lr_mult", 1.0)
    out.setdefault("wd_mult", 1.0)
    return out


# ----------------------------------------------------------------------------- stated gaps
def _stated_prompt(lim: Claim, method: str) -> str:
    return (f"{TESTBED_DESCRIPTION}\n\nALLOWED CONFIG CHANGES (field=value):\n"
            + "\n".join(f"- {k}" for k in stats.ALLOWED_CHANGES)
            + f"\n\nLIMITATION (arXiv:{lim.paper_id}, section {lim.section}): \"{lim.source_span}\"\n"
            f"METHOD: {method}\n")


def propose_stated(llm: LLM, lim: Claim) -> dict | None:
    method = gaps.method_key(lim)
    if not method:
        return None
    f, v = method.split("=", 1)
    holder: dict = {}

    def validate(o: StatedGapProposal) -> None:
        delta = {c.field: c.value for c in o.config_changes}
        if len(delta) != len(o.config_changes) or not stats.valid_delta(delta):
            raise ValueError("config_changes must be 1-2 allowed changes on different fields")
        if delta.get(f) != v:
            raise ValueError(f"config_changes must include {method}, the method the limitation is about")
        holder["delta"] = delta

    try:
        out, eid = llm.call("deep_stated_gap", (PROMPTS / "stated_gap.md").read_text(), _stated_prompt(lim, method),
                            StatedGapProposal, validate=validate)
    except Exception as e:  # noqa: BLE001 - an invalid proposal is dropped and logged
        return {"limitation_claim_id": lim.claim_id, "dropped": f"{type(e).__name__}: {e}"[:300]}
    return {"limitation_claim_id": lim.claim_id, "paper_id": lim.paper_id, "method": method,
            "quote": lim.source_span, "section": lim.section, "delta": holder["delta"],
            "delta_key": stats.delta_key(holder["delta"]), "concrete": bool(CONCRETE.search(lim.source_span)),
            "rationale": out.rationale, "event": eid}


# ----------------------------------------------------------------------------- pipeline
def _pool_map(fn, items, workers: int) -> list:
    if workers > 1 and len(items) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(items))) as pool:
            return list(pool.map(fn, items))
    return [fn(x) for x in items]


def build_deep(corpus: Path, llm: LLM, *, fetch: Callable[[str], str] = urllib_fetch,
               sleep: Callable[[float], None] = time.sleep, html_cache: Path | None = None) -> dict:
    cfg = load_config()
    dcfg, lcfg = cfg["deep_read"], cfg["lit_search"]
    corpus = Path(corpus)
    out = corpus / "deep"
    if out.exists():
        shutil.rmtree(out)
    (out / "papers").mkdir(parents=True)
    llm.recordings_path = out / "llm.jsonl"
    snap = Snapshot.from_corpus(corpus, deep=False)
    frozen = sorted(snap.claims.values(), key=lambda c: c.claim_id)
    graph, found = gaps.find_gaps(frozen, protocol(cfg)["prediction_version"] >= 2)
    queue = gaps.gap_queue(found, graph, frozen, overrides=snap_overrides(corpus),
                           top_n=dcfg["top_hypotheses"])
    selection = select_papers(queue, found, graph, snap.claims, snap.papers, snap.directions,
                              top_n=dcfg["top_hypotheses"], per_hypothesis=dcfg["papers_per_hypothesis"],
                              max_papers=dcfg["max_papers"])
    curated_labeled = sorted({c.paper_id for c in snap.claims.values() if c.tier == "T1" and c.config_change})
    to_read = list(selection["selected"]) + ([p for p in curated_labeled if p not in selection["selected"]]
                                             if dcfg.get("validate", True) else [])
    cache = RawCache(Path(html_cache) if html_cache else ROOT / cfg["paths"]["work"] / "cache" / "html", fetch,
                     max(lcfg["arxiv_min_interval_s"], 5.0), sleep, max_retries=lcfg.get("arxiv_max_retries", 3))
    fetched = {pid: fetch_paper(pid, cache) for pid in to_read}      # one request in flight (arXiv limits)

    by_paper: dict[str, list[Claim]] = {}
    for c in frozen:
        by_paper.setdefault(c.paper_id, []).append(c)
    budget = int(dcfg["section_token_budget"]) * 4

    def read(pid: str) -> dict:
        f = fetched[pid]
        rec = {"paper_id": pid, "title": snap.papers[pid].title, "source": f["source"], "url": f["url"],
               "version": f["version"], "html_sha256": f["html_sha256"],
               "validation_only": pid not in selection["selected"]}
        if not f["parsed"]:
            return {**rec, "full_text": "unavailable", "sections": [], "findings": {k: [] for k in FIELDS},
                    "dropped": []}
        secs = f["parsed"].sections
        try:
            out_, eid = llm.call("deep_read", (PROMPTS / "deep_read.md").read_text(),
                                 _prompt(snap.papers[pid], secs, by_paper.get(pid, []), budget), DeepFindings)
        except Exception as e:  # noqa: BLE001 - a failed read keeps the abstract-only claims
            return {**rec, "full_text": "extraction_failed", "error": f"{type(e).__name__}: {e}"[:300],
                    "sections": f["parsed"].headings, "findings": {k: [] for k in FIELDS}, "dropped": []}
        kept, dropped = verify_findings(out_, secs)
        return {**rec, "full_text": "read", "event": eid, "sections": f["parsed"].headings,
                "section_chars": {k: len(v) for k, v in secs.items() if k in SECTIONS},
                "findings": kept, "dropped": dropped}
    records = dict(zip(to_read, _pool_map(read, to_read, llm.workers)))

    claims, overrides, recipes, extra_dropped = [], {}, {}, []
    for pid in selection["selected"]:
        r = records[pid]
        if r["full_text"] != "read":
            continue
        ov = setting_override(r["findings"])
        if ov:
            overrides[pid] = ov
        new, dr = derive_claims(snap.papers[pid], r["findings"], ov, by_paper.get(pid, []),
                                float(dcfg["small_scale_max_params"]))
        claims += new
        extra_dropped += [{**d, "paper_id": pid} for d in dr]
        for method in OPTIMIZER_METHODS:                     # one recipe per method: the closest paper's
            items = [x for x in r["findings"]["recipes"] if x["method"] == method]
            if method not in recipes and items and (rc := map_recipe(items)):
                recipes[method] = {**rc, "paper_id": pid, "version": r["version"]}
    lims = [c for c in claims if is_limitation(c) and gaps.method_key(c)]
    stated = [s for s in _pool_map(lambda c: propose_stated(llm, c), lims, llm.workers) if s]

    validation = None
    if dcfg.get("validate", True) and curated_labeled:
        cur = [snap.papers[p] for p in curated_labeled]
        v_claims, _, v_err = extract(llm, cur, role="extract_validation")
        labels = [c for c in snap.claims.values() if c.tier == "T1"]
        before = agreement_with_curated(v_claims, labels)
        ovs = {p: setting_override(records[p]["findings"]) for p in curated_labeled
               if records[p]["full_text"] == "read"}
        after_claims = []
        for c in v_claims:
            ov = ovs.get(c.paper_id)
            if ov:
                sf = merge_setting(c.setting_fields, ov["setting_fields"])
                c = c.model_copy(update={"setting_fields": sf, "coverage": coverage_of(sf)})
            after_claims.append(c)
        after = agreement_with_curated(after_claims, labels)
        validation = {"n": before["n"], "abstract_only": {k: before[k] for k in ("config_mapping", "direction", "setting")},
                      "with_full_text": {k: after[k] for k in ("config_mapping", "direction", "setting")},
                      "papers_read": sorted(ovs), "rows_before": before["rows"], "rows_after": after["rows"],
                      "errors": v_err}

    for pid, r in records.items():
        (out / "papers" / f"{pid}.json").write_text(json.dumps(r, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (out / "claims.json").write_text(json.dumps({"claims": [c.model_dump() for c in claims]}, indent=2,
                                                sort_keys=True, ensure_ascii=False) + "\n")
    (out / "overrides.json").write_text(json.dumps(overrides, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "recipes.json").write_text(json.dumps(recipes, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "stated.json").write_text(json.dumps(stated, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    meta = {"deep_read_version": 1, "config": dcfg, "selected": selection["selected"],
            "read": sorted(p for p, r in records.items() if r["full_text"] == "read"),
            "unavailable": sorted(p for p, r in records.items() if r["full_text"] != "read"),
            "claims": len(claims), "overrides": len(overrides), "recipes": sorted(recipes),
            "stated_gaps": len([s for s in stated if "delta" in s]), "dropped": extra_dropped,
            "fetch_failures": cache.failures, "llm_calls": llm.n_calls, "llm_cost_usd": round(llm.cost_usd, 4),
            "concrete_rule": CONCRETE_RULE, "validation": validation and {
                k: validation[k] for k in ("n", "abstract_only", "with_full_text")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    return meta


def snap_overrides(corpus: Path) -> list[str]:
    import yaml
    p = Path(corpus) / "niche.yaml"
    return list((yaml.safe_load(p.read_text()) or {}).get("pi_overrides", [])) if p.exists() else []


# ----------------------------------------------------------------------------- verify-deep
def verify_deep(corpus: Path, fetch: Callable[[str], str] = urllib_fetch) -> list[str]:
    """Re-fetch every read paper (needs network), check the HTML's SHA256 and every stored quote.
    Anyone can check the quotes without us redistributing the papers."""
    problems = []
    for f in sorted((Path(corpus) / "deep" / "papers").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("full_text") != "read":
            continue
        try:
            body = fetch(r["url"])
        except Exception as e:  # noqa: BLE001
            problems.append(f"{r['paper_id']}: fetch failed: {type(e).__name__}: {e}"[:200])
            continue
        if hashlib.sha256(body.encode()).hexdigest() != r["html_sha256"]:
            problems.append(f"{r['paper_id']}: SHA256 differs from the recorded {r['version']} (page changed?)")
        secs = parse_paper(body, r["paper_id"]).sections
        for fname, items in r["findings"].items():
            for it in items:
                if not quote_in(it["quote"], secs.get(it["section"], "")):
                    problems.append(f"{r['paper_id']}: {fname} quote not found in {it['section']}: {it['quote'][:80]}")
    return problems


def is_limitation(c: Claim) -> bool:
    return c.span_source == "full_text" and re.search(r"_l\d+$", c.claim_id) is not None


def is_full_text(c: Claim) -> bool:
    return c.span_source == "full_text"


__all__ = ["build_deep", "verify_deep", "numbers_in", "map_recipe", "scale_of", "params_of", "is_derived",
           "is_full_text", "merge_setting"]
