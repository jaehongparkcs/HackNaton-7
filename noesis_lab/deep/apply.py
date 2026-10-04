"""Apply a frozen deep read (<corpus>/deep/) to a Snapshot (DEEP_READ §4). Pure code, no network.

- Settings: for each claim of a read paper, full-text setting fields override abstract-extracted ones
  (field by field; "unspecified" never overrides). Coverage is recomputed by the same rule. A curated
  (T1) claim keeps its human flag, always.
- Claims: full-text claims (results, small-scale evidence, mechanisms, limitations) join the corpus.
  Their quote was checked against the fetched section when it was read; here it must match a stored
  finding of that paper. A full-text result supersedes the same paper's abstract claim about the same
  change (non-T1 only), so a comparison is never counted twice.
- Recipes and stated-gap proposals are carried on the snapshot for the session to use."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import gaps
from ..schemas import Claim, SettingFields
from ..search.coverage import coverage_of, is_derived
from .read import merge_setting


@dataclass
class DeepRead:
    meta: dict
    selection: dict
    papers: dict[str, dict]
    claims: list[Claim]
    overrides: dict[str, dict]
    recipes: dict[str, dict]
    stated: list[dict]
    validation: dict | None
    changes: list[dict] = field(default_factory=list)     # coverage overrides applied to this snapshot


def load(corpus: Path) -> DeepRead | None:
    d = Path(corpus) / "deep"
    if not (d / "meta.json").exists():
        return None
    rd = lambda n: json.loads((d / n).read_text())  # noqa: E731
    return DeepRead(meta=rd("meta.json"), selection=rd("selection.json"),
                    papers={p.stem: json.loads(p.read_text()) for p in sorted((d / "papers").glob("*.json"))},
                    claims=[Claim(**c) for c in rd("claims.json")["claims"]], overrides=rd("overrides.json"),
                    recipes=rd("recipes.json"), stated=rd("stated.json"), validation=rd("validation.json"))


class DeepReadError(ValueError):
    pass


def apply(snap, dr: DeepRead) -> None:
    """Mutates the snapshot: overridden settings, added full-text claims, superseded abstract claims."""
    for cid, c in sorted(snap.claims.items()):
        ov = dr.overrides.get(c.paper_id)
        if not ov or c.tier == "T1" or is_derived(c):
            continue
        sf = merge_setting(c.setting_fields, ov["setting_fields"])
        new_cov = coverage_of(sf)
        if sf != c.setting_fields or new_cov != c.coverage:
            dr.changes.append({"claim_id": cid, "paper_id": c.paper_id, "before": c.coverage, "after": new_cov,
                               "setting_before": c.setting_fields.model_dump() if c.setting_fields else None,
                               "setting_after": sf.model_dump(), "quote": ov["quote"], "section": ov["section"],
                               "parameter_count": ov["parameter_count"]})
            snap.claims[cid] = c.model_copy(update={
                "setting_fields": sf, "coverage": new_cov, "setting_source": "full_text",
                "coverage_note": f"computed by the coverage rule from the FULL-TEXT setting ({ov['section']}: "
                                 f"{ov['parameter_count']} parameters)"})
    for c in dr.claims:
        rec = dr.papers.get(c.paper_id, {})
        quotes = {it["quote"] for items in rec.get("findings", {}).values() for it in items}
        if c.paper_id not in snap.papers or c.source_span not in quotes:
            raise DeepReadError(f"{c.claim_id}: quote is not a verified finding of {c.paper_id}")
        snap.claims[c.claim_id] = c
    superseded = {}
    for c in dr.claims:
        if c.expected_outcome == "context" or not c.claim_id.split("_")[-1].startswith("r"):
            continue
        key = gaps.method_key(c)
        for cid, old in list(snap.claims.items()):
            if (old.paper_id == c.paper_id and old.span_source != "full_text" and old.tier != "T1"
                    and not is_derived(old) and gaps.method_key(old) == key):
                superseded[cid] = old.model_copy(update={"superseded_by": c.claim_id})
                del snap.claims[cid]
    snap.superseded = superseded
    snap.deep = dr
    snap.recipes = dr.recipes
    snap.stated = [s for s in dr.stated if "delta" in s]
    snap._build_index()


def full_text_setting(sf: dict) -> SettingFields:
    return SettingFields(**sf)
