"""Streamlit dashboard: the evidence chain in under 60 seconds (BUILD_PLAN s7).

Read-only over a notebook.sqlite. Every number shown is read from a stored run / analysis /
decision; nothing here computes a statistic. Run: `make app` (SESSION=golden).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from noesis_lab import stats
from noesis_lab.evidence_chain import chain_dot, evidence_grid
from noesis_lab.gap_views import (
    DIRECTION_WORD,
    direction_map_dot,
    direction_view_dot,
    exploration_rows,
    gap_card_dot,
    gap_rows,
    gap_type_summary,
    our_results,
    overview_dot,
    prediction_counts,
    prediction_rows,
    shrinkage_rows,
)
from noesis_lab.schemas import ExperimentConfig, NicheSpec, RunResult
from noesis_lab.search.grid import coverage_grid, quotes
from noesis_lab.store import Store

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"      # baseline / candidate / neutral

st.set_page_config(page_title="Noesis Lab", page_icon="🔬", layout="wide")


# ----------------------------------------------------------------------------- source
def find_notebooks() -> dict[str, Path]:
    out = {}
    for pat in ("results/*/notebook.sqlite", "work/replay-*/notebook.sqlite"):
        for p in sorted(ROOT.glob(pat)):
            out[str(p.relative_to(ROOT))] = p
    return out


books = find_notebooks()
env_nb = os.environ.get("NOESIS_NOTEBOOK")
if env_nb:
    books = {env_nb: Path(env_nb), **books}
if not books:
    st.error("No notebook found. Run `make golden` (needs an API key and a Mac), or `make smoke`.")
    st.stop()
choice = st.sidebar.selectbox("Session notebook", list(books), index=0)
S = Store(books[choice], readonly=True)
meta = S.get_meta("session") or {"status": "replayed", "llm_mode": "replay"}
profile = S.get_meta("profile") or {}


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float):
    s = Store(path, readonly=True)
    return dict(hyps=s.hypotheses(), analyses=s.analyses(), decisions=s.decisions(), events=s.events(),
                runs={r["run_id"]: r for r in s.runs()}, claims={c["claim_id"]: c for c in s.claims()},
                papers={p["paper_id"]: p for p in s.papers()}, evidence=s.derived_evidence(),
                noise=s.noise_floors().get("noise_baseline"), rule=s.get_meta("rule_text"),
                snapshot=s.get_meta("snapshot"), objective=s.get_meta("objective"),
                queue=s.get_meta("candidate_queue") or [], baseline=s.get_meta("baseline_config"),
                niche=s.get_meta("niche") or {}, corpus=s.get_meta("corpus") or {},
                holds=s.get_meta("t4_holds") or {}, gaps=s.get_meta("gaps") or [],
                gap_graph=s.get_meta("gap_graph") or {}, directions=s.get_meta("directions") or [],
                explore=s.get_meta("explore") or {})


D = load(str(books[choice]), books[choice].stat().st_mtime)
analyses = {a["analysis_id"]: a for a in D["analyses"]}
decisions = D["decisions"]
STEPS = profile.get("budget_mode") == "steps"        # equal-token delta is redundant under a step budget

BADGE_LABEL = {"coverage": "Coverage gap (missing cell)", "abc": "ABC closure (hidden connection)",
               "link": "Combination (link prediction)", "contradiction": "Contradiction (signed edges)",
               "bridge": "Structural hole (bridge between communities)"}
STATUS_LABEL = {
    "rejected_prior_art": ("REJECTED: PRIOR ART", "🟥"),
    "rejected_critic": ("REJECTED BY CRITIC", "🟥"),
    "screened_promising": ("SCREENED: PROMISING", "🟩"),
    "screened_no_improvement": ("SCREENED: NO IMPROVEMENT", "🟨"),
    "screened_harmful": ("SCREENED: HARMFUL", "🟧"),
    "run_failed": ("RUN FAILED", "⬛"),
    "explored": ("EXPLORATION — NOT A RESULT (1 seed; not selected for paired confirmation)", "⬜"),
    "held_pi_review": ("HELD FOR PI REVIEW: POSSIBLE OVERLAP WITH AN UNEXTRACTED PAPER", "🟫"),
    "soft_rejected_prior_art": ("SOFT-REJECTED: COVERED BY AUTO-EXTRACTED / PREPRINT CLAIM (PI MAY OVERRIDE)", "🟪"),
}

# ----------------------------------------------------------------------------- header
st.title("Noesis Lab")
st.caption("LLMs propose and interpret. Deterministic code measures, decides and records. "
           "An LLM mistake can waste compute or skip an idea; it cannot produce a measured result or a decision.")
if meta.get("llm_mode") == "mock" or profile.get("name") == "smoke":
    st.warning("MOCK LLM / SMOKE PROFILE: pipeline check only. These numbers are not results.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("LLM mode", meta.get("llm_mode", "?"))
c2.metric("Device / profile", f"{(D['runs'] and next(iter(D['runs'].values()))['env'].get('device', '?'))} / {profile.get('name', '?')}")
c3.metric("Runs executed", len(D["runs"]))
c4.metric("Runs avoided (prior art)", meta.get("runs_avoided", 0))
c5.metric("LLM spend", f"${meta.get('llm_cost_usd', 0):.2f}", f"{meta.get('llm_calls', len(D['events']))} calls")
st.write(f"**Objective.** {D['objective']}")
snap = D["snapshot"] or {}
corpus, niche = D["corpus"], D["niche"]
LIVE = corpus.get("mode") == "live_search"
if LIVE:
    st.write(f"**Literature corpus:** {corpus['label']}. {snap.get('claims', '?')} claims. "
             "Searched live once, then frozen into this bundle; replay reads only the frozen copy. "
             "The strongest negative statement is “not found in these papers for these queries”; the UI never says “novel”.")
else:
    st.write(f"**Curated literature snapshot only:** {snap.get('papers', '?')} papers, {snap.get('claims', '?')} claims "
             f"({', '.join(snap.get('dimensions', []))}). Fetched {snap.get('fetched', '?')} from arXiv. "
             "Verdicts are scoped to this snapshot; the UI never says “novel”.")
for reason in corpus.get("degraded", []):
    st.warning(f"Search degraded: {reason}")

# ----------------------------------------------------------------------------- niche form (sidebar)
with st.sidebar.form("niche"):
    st.markdown("### Define the niche")
    st.caption("Chosen by the PI. The lab chooses the experiments, not the niche. This form only writes a file; "
               "it does not start a search or training.")
    n_title = st.text_input("Title", niche.get("title", ""))
    n_desc = st.text_area("Description (2–4 sentences)", niche.get("description", ""))
    n_inc = st.text_area("Include terms (one per line)", "\n".join(niche.get("include_terms", [])))
    n_exc = st.text_area("Exclude terms (one per line)", "\n".join(niche.get("exclude_terms", [])))
    n_cat = st.text_input("arXiv categories", ", ".join(niche.get("arxiv_categories", ["cs.LG", "cs.CL", "cs.NE"])))
    n_from = st.text_input("Date from", niche.get("date_from", "2017-01-01"))
    n_seed = st.text_input("Seed papers (arXiv ids)", ", ".join(niche.get("seed_papers", [])))
    n_max = st.number_input("Max papers", 1, 300, int(niche.get("max_papers", 80)))
    if st.form_submit_button("Write niche file") and n_title.strip():
        def _lines(t: str, sep: str = "\n") -> list[str]:
            return [x.strip() for x in t.split(sep) if x.strip()]
        spec = NicheSpec(title=n_title.strip(), description=n_desc.strip(), include_terms=_lines(n_inc),
                         exclude_terms=_lines(n_exc), arxiv_categories=_lines(n_cat, ","), date_from=n_from.strip(),
                         seed_papers=_lines(n_seed, ","), max_papers=int(n_max))
        fname = "niche_" + re.sub(r"[^a-z0-9]+", "_", spec.title.lower()).strip("_")[:40] + ".yaml"
        (ROOT / fname).write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False))
        st.success(f"Wrote {fname}")
        st.code(f"make search NICHE={fname}\nmake golden NICHE={fname} SESSION=<name>", language="bash")

# ----------------------------------------------------------------------------- literature map
st.header("Literature: the niche, the search and the map")
if niche:
    with st.container(border=True):
        st.markdown(f"**Niche — chosen by the PI:** {niche['title']}")
        st.write(niche["description"])
        st.caption(f"Include: {', '.join(niche['include_terms']) or '—'}  ·  Exclude: {', '.join(niche['exclude_terms']) or '—'}  ·  "
                   f"arXiv: {', '.join(niche['arxiv_categories'])}  ·  from {niche['date_from']}  ·  seeds: "
                   f"{', '.join(niche['seed_papers']) or '—'}  ·  testbed: {niche['testbed']}")

with st.expander("Search transparency: queries, what was kept or dropped, extraction yield, coverage rule, tiers",
                 expanded=LIVE):
    if not LIVE:
        st.info("No live search in this session: the corpus is the curated snapshot only (13 human-checked claims, tier T1). "
                "Run `make search NICHE=niche.yaml` or `make golden NICHE=niche.yaml`.")
    else:
        cnt, ext, tiers = corpus.get("counts", {}), corpus.get("extraction", {}), corpus.get("tiers", {})
        m = st.columns(5)
        m[0].metric("Search date", corpus.get("search_date", "?"))
        m[1].metric("Retrieved → kept", f"{cnt.get('retrieved', '?')} → {cnt.get('kept', '?')}",
                    f"-{cnt.get('duplicates', 0)} dup, -{cnt.get('excluded', 0)} excluded, -{cnt.get('ranked_out', 0)} ranked out",
                    delta_color="off")
        m[2].metric("Claims kept", ext.get("claims_kept", "?"),
                    f"{ext.get('dropped_non_verbatim', 0)} dropped: quote not verbatim", delta_color="off")
        m[3].metric("Tiers (claims)", f"T1·{tiers.get('T1', 0)} T2·{tiers.get('T2', 0)} T3·{tiers.get('T3', 0)}",
                    f"T4·{tiers.get('T4', 0)} unextracted papers", delta_color="off")
        m[4].metric("Search LLM spend", f"${corpus.get('llm_cost_usd', 0):.2f}", f"{corpus.get('llm_calls', 0)} calls")
        st.dataframe(pd.DataFrame([{"query": q["q"], "origin": "code (one per runnable change)" if q["kind"] == "deterministic"
                                    else f"LLM-proposed ({q['kind']})", "dimension": q["dimension"],
                                    "papers returned": q.get("n_results")} for q in corpus.get("queries", [])]),
                     use_container_width=True, hide_index=True)
        v = corpus.get("validation")
        if v and "n" in v:
            st.markdown(f"**Extraction check against the curated set** ({v['n']} human-labeled claims with a config change): "
                        f"config mapping **{v['config_mapping']}/{v['n']}**, direction **{v['direction']}/{v['n']}**, "
                        f"setting (covered or not) **{v['setting']}/{v['n']}**.")
        oa = corpus.get("openalex", {})
        st.caption(f"OpenAlex: {'on' if oa.get('enabled') else 'off'}, {oa.get('matched', 0)} papers matched. It can only add evidence "
                   "of publication; without it a claim is labeled “review status unknown” and treated as a preprint (T3).")
    st.markdown("**Coverage rule** (code; one rule for every paper)")
    st.text(corpus.get("coverage_rule", ""))
    st.markdown("**Trust tiers.** T1 curated (human-checked; can hard-reject) · T2 auto-extracted, quote verbatim, published at a venue "
                "(soft-reject) · T3 auto-extracted, quote verbatim, preprint or unknown review status (soft-reject) · "
                "T4 retrieved but no claim passed the quote check (text-similarity hold for PI review). "
                "Unreviewed work still counts as covering a topic.")

MAPD = dict(D, queue=D["queue"])
grid = coverage_grid(MAPD)
st.subheader("Coverage grid")
st.caption("Rows: the config changes this testbed can run. Cells: claims by tier and dominant direction. "
           "GAP = no directional claim covers our setting, but an “improves” claim exists elsewhere: what the queue tries first.")
gdf = pd.DataFrame(grid)
st.dataframe(gdf.style.apply(lambda r: ["background-color: rgba(235,104,52,0.18)" if r["gap"] else "" for _ in r], axis=1),
             use_container_width=True, hide_index=True)
if not any(q["status"] == "open" for q in D["queue"]) and not any(h.get("origin") == "GENERATED" for h in D["hyps"]):
    st.info("No runnable candidates in this testbed for this niche: the map and overlap analysis are still shown, "
            "but nothing in the retrieved literature maps onto a config change the testbed can run.")
row_key = st.selectbox("Show the quotes behind a row", [r["config change"] for r in grid])
for qt in quotes(MAPD, row_key)[:40]:
    st.markdown(f"`{qt['tier']}` · {qt['review']} · {qt['published']} · **{qt['bucket']}** · {qt['direction']}  \n"
                f"> “{qt['quote']}”  \n> — [{qt['title']}]({qt['url']}) (`{qt['claim_id']}`)")
if row_key in D["holds"]:
    st.warning("Possible overlap with unextracted paper(s): "
               + ", ".join(f"arXiv:{pid} (similarity {score:.2f})" for pid, score in D["holds"][row_key])
               + ". Held for PI review unless dismissed in niche.yaml (`pi_dismissed_holds`).")

papers_xy = [p for p in D["papers"].values() if p.get("map_xy")]
if len(papers_xy) >= 3:
    st.subheader("Paper map")
    DIM = {"norm": "norm", "norm_position": "norm", "activation": "activation", "pos_encoding": "positional",
           "optimizer": "optimizer", "schedule": "schedule"}
    by_paper: dict = {}
    for c in D["claims"].values():
        by_paper.setdefault(c["paper_id"], []).append(DIM.get(c["dimension"], "other"))
    pm = pd.DataFrame([{"x": p["map_xy"][0], "y": p["map_xy"][1], "title": p["title"], "paper": p["paper_id"],
                        "tier": p.get("tier", "T1"), "relevance": p.get("relevance") or 0.3, "published": p.get("published", ""),
                        "dimension": max(set(by_paper.get(p["paper_id"], ["other"])), key=by_paper.get(p["paper_id"], ["other"]).count)}
                       for p in papers_xy])
    pos = {p["paper_id"]: p["map_xy"] for p in papers_xy}
    hyp_by_key = {h.get("candidate_key"): h for h in D["hyps"] if h.get("candidate_key")}
    stars = []
    for q in D["queue"]:
        pts = [pos[D["claims"][c]["paper_id"]] for c in q["supporting_claim_ids"] if D["claims"][c]["paper_id"] in pos]
        if pts:
            h = hyp_by_key.get(q["key"])
            stars.append({"x": sum(a for a, _ in pts) / len(pts), "y": sum(b for _, b in pts) / len(pts),
                          "candidate": q["key"], "outcome": h["status"] if h else q["status"]})
    base = alt.Chart(pm).mark_point(filled=True, opacity=0.75).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), color=alt.Color("dimension:N"),
        shape=alt.Shape("tier:N", scale=alt.Scale(domain=["T1", "T2", "T3", "T4"],
                                                  range=["circle", "square", "triangle-up", "cross"])),
        size=alt.Size("relevance:Q", legend=None, scale=alt.Scale(range=[40, 260])),
        tooltip=["paper", "title", "tier", "published", "dimension", "relevance"])
    layers = base
    if stars:
        layers = base + alt.Chart(pd.DataFrame(stars)).mark_point(shape="M0,-1L0.29,-0.4L0.95,-0.31L0.47,0.15L0.59,0.81L0,0.5L-0.59,0.81L-0.47,0.15L-0.95,-0.31L-0.29,-0.4Z",
                                                                  size=420, filled=True, stroke="black", strokeWidth=1).encode(
            x="x:Q", y="y:Q", fill=alt.Fill("outcome:N", legend=alt.Legend(title="candidate ★")), tooltip=["candidate", "outcome"])
    st.altair_chart(layers.properties(height=380).resolve_scale(color="independent", fill="independent"), use_container_width=True)
    st.caption("Positions show text similarity only (TF-IDF → PCA); use the coverage grid for decisions. "
               "★ = our candidates, at the centroid of their supporting claims' papers.")

# ----------------------------------------------------------------------------- hypothesis engine
if D["gaps"]:
    st.header("Hypothesis engine: gaps in the literature graph")
    st.caption("Code builds a graph from the frozen, quote-verified claims (methods, mechanisms, settings, outcome) and scores "
               "where it is silent or disagrees for our setting. Scores are **structural hints, not probabilities**: the "
               "prior-art gate and the measured result still decide. Hypotheses are transfer / combination / resolution "
               f"questions “not found in {snap.get('papers', '?')} retrieved papers”, never “novel”.")
    shared = D["gap_graph"].get("shared_mechanisms", {})
    need = D["explore"].get("config", {}).get("min_shared_mechanisms", 3)
    line = (f"Mechanism nodes linked to 2 or more runnable methods: **{len(shared)}**"
            + (" — " + "; ".join(f"{k.replace('_', ' ')} ({', '.join(v)})" for k, v in shared.items()) if shared else ""))
    if len(shared) < need:
        st.warning(line + f". Thin graph (want ≥ {need}): ABC, combination and bridge gaps will be sparse; "
                   "coverage and contradiction gaps do not depend on mechanisms.")
    else:
        st.caption(line)
    with st.expander("How a gap is scored (code; shown verbatim)"):
        st.text(D["explore"].get("gap_score_rule", ""))
    rows = gap_rows(D["gaps"])
    queued = {q["key"]: q for q in D["queue"]}
    hyp_by_key = {h.get("candidate_key"): h for h in D["hyps"] if h.get("candidate_key")}
    for r in rows:
        key = r["config change"].replace(" + ", "+")
        r["hypothesis"] = (STATUS_LABEL.get(hyp_by_key[key]["status"], (hyp_by_key[key]["status"],))[0] if key in hyp_by_key
                           else queued[key]["status"].replace("_", " ") if key in queued else "below the top gaps")
    for fam in gap_type_summary(D["gaps"], D["gap_graph"]):
        if fam["gaps"]:
            st.markdown(f"**{fam['family']}:** {fam['gaps']} gap(s)")
        else:
            st.info(f"**{fam['family']}:** {fam['note']}")
    st.subheader("Gap list")
    st.dataframe(pd.DataFrame(rows).drop(columns=["gap_id"]), use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")
                                for c in ("gap score", "plausibility", "1 − coverage", "testability", "evidence quality")})
    st.subheader("Gap card")
    gid = st.selectbox("Gap", [g["gap_id"] for g in D["gaps"]],
                       format_func=lambda x: next(f"{r['rank']}. {r['type']} · {r['config change']} · score {r['gap score']:.2f}"
                                                  for r in rows if r["gap_id"] == x))
    gap = next(g for g in D["gaps"] if g["gap_id"] == gid)
    g1, g2 = st.columns([3, 2])
    g1.graphviz_chart(gap_card_dot(gap, D["claims"]), use_container_width=True)
    with g2:
        st.markdown(f"**{BADGE_LABEL.get(gap['gap_type'], gap['gap_type'])}** · label `{gap['novelty_type']}` (computed by code)")
        st.write(f"gap score **{gap['score']:.3f}** = plausibility {gap['plausibility']:.2f} × (1 − coverage {gap['coverage']:.2f}) "
                 f"× testability {gap['testability']:.1f} × evidence quality {gap['evidence_quality']:.2f}")
        st.markdown(f"Predicted direction along the path (code, from the signs of the claims): "
                    f"**{DIRECTION_WORD.get(gap.get('predicted_direction', ''), '?')}**")
        measured = [r for r in our_results(D["hyps"], analyses) if r["delta_key"] == gap["delta_key"]]
        for r in measured:
            st.markdown(f"Measured ({r['seeds']} paired seeds): **{r['branch']}** ({r['sd']:+.2f}× noise SD)"
                        + (f" → prediction **{r['outcome'].upper()}**" if r["outcome"] not in ("", "no_prediction") else ""))
        if not measured:
            st.caption("Not measured in this session.")
        st.json(gap["detail"], expanded=False)
        st.caption(f"Not found in {snap.get('papers', '?')} papers retrieved on {corpus.get('search_date', '?')} for "
                   f"{len(corpus.get('queries', []))} niche queries plus {len(D['directions'])} direction searches.")
    for cid in gap["claim_ids"]:
        c = D["claims"].get(cid)
        if c:
            p = D["papers"].get(c["paper_id"], {})
            st.markdown(f"`{cid}` · `{c.get('tier', 'T1')}` · {p.get('published', '')}  \n> “{c['source_span']}”  \n"
                        f"> — [{p.get('title', '')}]({p.get('url', '')})"
                        + (f"  \n> mechanism quoted: “{c['mechanism']}”" if c.get("mechanism") else ""))
    ex_rows = exploration_rows(decisions)
    if ex_rows:
        st.subheader("Explore cheaply, confirm rigorously")
        st.caption("Exploration runs ONE seed per hypothesis against the incumbent's run on the same seed. It is "
                   "**EXPLORATION — NOT A RESULT**: it only decides which hypotheses earn a paired confirmation. "
                   "Finalists are picked by code (best per mechanism cell, then top by single-seed improvement).")
        st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
        sh = shrinkage_rows(decisions)
        if sh:
            st.markdown("**Shrinkage** — what a single run suggested vs what the paired protocol measured "
                        "(positive = the one-run number was optimistic). This is the single-run-loop comparison, "
                        "measured inside one session.")
            st.dataframe(pd.DataFrame(sh), use_container_width=True, hide_index=True)
        for d in [d for d in decisions if d["kind"] == "promotion"]:
            st.success(f"Promotion (cycle {d['cycle']}): `{d['candidate_key']}` stayed promising on every seed "
                       f"({d['improvement_in_noise_sd']:+.2f}× SD) and became the incumbent. Its own runs are the new noise "
                       f"floor (SD {d['noise_sd']:.4f}); later candidates are deltas on it. Headline numbers stay relative "
                       "to the original baseline.")
        if not [d for d in decisions if d["kind"] == "promotion"]:
            st.caption("No promotion in this session: no finalist stayed promising on every seed. The baseline is still the incumbent.")
    st.subheader("How good were the gaps?")
    prows = prediction_rows(D["hyps"], analyses)
    if not prows:
        st.info("No gap hypothesis has been screened in this session yet, so there is nothing to score.")
    else:
        st.caption("Each gap predicts a direction from the signs of the claims on its path. After screening, code compares it "
                   "with the measured branch: hit = same sign, miss = opposite sign, null = no improvement. Counts, not "
                   "percentages: the numbers are small. Misses are reported, not hidden.")
        pc1, pc2 = st.columns(2)
        pc1.dataframe(pd.DataFrame(prediction_counts(prows, "gap type")), use_container_width=True, hide_index=True)
        pc2.dataframe(pd.DataFrame(prediction_counts(prows, "label")), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)
    st.subheader("Overview map")
    st.graphviz_chart(overview_dot(D["gap_graph"], D["gaps"], our_results(D["hyps"], analyses)), use_container_width=True)
    st.caption("Boxes = runnable methods, ellipses = mechanisms quoted from abstracts, colors = graph communities. "
               "Claim edges: green +, red −, grey 0 (×count). Thick edges = our measured results. Dashed orange = gaps with their score.")

    st.subheader("Direction map")
    st.graphviz_chart(direction_map_dot(D["directions"]), use_container_width=True)
    st.caption("Directions colored by gap status: covered grey · partial amber · contested purple · open green · unexplored white · "
               "dashed border = outside the testbed (shown, never run). Dots are papers; a shared paper connects two directions.")
    with st.expander("Direction gap status rule (code; shown verbatim)"):
        st.text(D["explore"].get("direction_status_rule", ""))
    did_ = st.selectbox("Direction view", [d["direction_id"] for d in D["directions"]],
                        format_func=lambda x: next(f"{d['direction_id']} · {d['title']} [{d['status']}]" for d in D["directions"]
                                                   if d["direction_id"] == x))
    dsel = next(d for d in D["directions"] if d["direction_id"] == did_)
    st.graphviz_chart(direction_view_dot(dsel, D["claims"], D["papers"], D["hyps"], analyses), use_container_width=True)
    with st.expander("Scout panel: the scout's directions are searches, never evidence"):
        st.dataframe(pd.DataFrame([{
            "id": d["direction_id"], "origin": "scout (LLM)" if d["origin"] == "scout" else "code (block not mentioned)",
            "title": d["title"], "status (code)": d["status"] + (" · outside testbed" if d["outside_testbed"] else ""),
            "building blocks": ", ".join(d["building_blocks"]) or "—",
            "queries (papers returned)": "; ".join(f"{q['q']} ({q['n_results']})" for q in d["queries"]),
            "papers kept": d["n_papers"], "claims linked": d["n_claims"],
            "remembered titles": "; ".join(f"{h['title'][:40]} → {h['status']}" + (f" ({h['paper_id']})" if h["paper_id"] else "")
                                           for h in d["title_hints"]) or "—"} for d in D["directions"]]),
            use_container_width=True, hide_index=True)
        st.caption("A remembered title is only a search: found → the paper enters with its real arXiv id; not found → logged and dropped.")

# ----------------------------------------------------------------------------- noise floor
st.header("1. Noise floor")
n = D["noise"]
if n:
    nf = pd.DataFrame({"seed": n["seeds"], "val_loss": n["val_losses"], "run_id": n["run_ids"]})
    band = pd.DataFrame({"lo": [n["mean"] - n["sd"]], "hi": [n["mean"] + n["sd"]]})
    base = alt.Chart(nf).encode(x=alt.X("seed:O", title="baseline seed"))
    chart = (alt.Chart(band).mark_rect(color=GRAY, opacity=0.18).encode(y="lo:Q", y2="hi:Q")
             + alt.Chart(pd.DataFrame({"m": [n["mean"]]})).mark_rule(color=GRAY, strokeDash=[4, 3]).encode(y="m:Q")
             + base.mark_circle(size=90, color=BLUE).encode(
                 y=alt.Y("val_loss:Q", scale=alt.Scale(zero=False), title="validation loss (nats/char)"),
                 tooltip=["seed", "val_loss", "run_id"]))
    left, right = st.columns([2, 1])
    left.altair_chart(chart.properties(height=240), use_container_width=True)
    iv = stats.sd_interval(n["val_losses"])
    right.markdown(f"**{n['n']} baseline seeds**\n\nmean {n['mean']:.4f}\n\n**seed-to-seed SD {n['sd']:.4f}**\n\n"
                   "Every later delta is shown against this band (shaded = ±1 SD).")
    right.caption(f"The SD is itself an estimate from {n['n']} seeds"
                  + (f": exhaustive-bootstrap 95% interval {iv[0]:.4f}–{iv[1]:.4f}" if iv else "")
                  + ". It differs between sessions and machines; never pool it across sessions.")

# ----------------------------------------------------------------------------- hypotheses
st.header("2. Hypotheses")
for h in D["hyps"]:
    label, icon = STATUS_LABEL.get(h["status"], (h["status"].upper(), "⬜"))
    with st.container(border=True):
        st.markdown(f"#### {icon} {label}  ·  `{h.get('origin', 'FIXTURE')}` {h.get('title', h['fixture_id'])}")
        st.write(h.get("statement", ""))
        if h.get("kind") == "gap":
            g = h.get("gap") or {}
            st.markdown(f"**`{(h.get('novelty_type') or '?').upper()}`** · from gap `{', '.join(h.get('gap_ids', []))}` · gap score "
                        f"**{g.get('score', 0):.2f}** (plausibility {g.get('plausibility', 0):.2f}, 1 − coverage {1 - g.get('coverage', 0):.2f}, "
                        f"testability {g.get('testability', 0):.1f}, evidence quality {g.get('evidence_quality', 0):.2f}) — a structural hint, "
                        "not a probability. Label and score computed by code.")
            for cid in h.get("grounded_in", []):
                c = D["claims"].get(cid)
                if c:
                    st.markdown(f"> grounded in `{cid}` (`{c.get('tier', 'T1')}`): “{c['source_span']}”")
            t = h.get("tighten")
            if t:
                word = {"passed": "PASSED", "narrowed": "NARROWED after one revision", "rejected": "REJECTED",
                        "held": "HELD for PI review"}[t["outcome"]]
                st.markdown(f"**Tighten pass: {word}**")
                for r in t["rounds"]:
                    st.caption(f"round {r['round']} · `{r['delta_key']}` · searches: "
                               + "; ".join(f"{q['q']} ({q['n_results']})" for q in r["queries"])
                               + f" · {r['new_papers']} new papers, {r['new_claims']} new claims · gate: {r['gate']}"
                               + (f" · overlap `{r['claim_id']}`: “{r['passage']}”" if r["gate"] != "run" and r.get("passage") else ""))
        elif h.get("origin") == "GENERATED":
            st.caption(f"Chosen by code from the literature queue. Priority tuple {h.get('queue_priority')}; "
                       f"supporting claims {', '.join(h.get('supporting_claim_ids', []))}.")
        pa = h.get("prior_art")
        if pa:
            st.markdown(f"**Prior-art gate:** `{pa['verdict']}` — {pa['label']}  ·  event `{pa['event_id']}`")
            if pa.get("passage"):
                st.markdown(f"> “{pa['passage']}”  \n> — [{pa['paper_title']}]({pa['paper_url']}) (arXiv:{pa['paper_id']}, claim `{pa['claim_id']}`)")
            if pa.get("claim_id"):
                cov = {True: "YES", False: "NO"}.get(pa.get("covers_our_setting"), "n/a")
                how = "human-curated" if pa.get("tier") in (None, "T1") else f"computed by the coverage rule ({pa.get('coverage')})"
                st.markdown(f"**Covers our setting: {cov}** · _{how}_ · tier `{pa.get('tier') or 'T1'}` — {pa.get('coverage_note') or ''}")
            st.caption(f"LLM judged only “same comparison: {pa.get('same_comparison')}”. {pa['rationale']}")
        if h["status"] == "soft_rejected_prior_art":
            st.info("Not run: covered by an auto-extracted / preprint claim. The PI may override by adding this "
                    f"candidate (`{h.get('candidate_key')}`) to `pi_overrides` in the niche file.")
        if h["status"] == "rejected_prior_art":
            d = next(x for x in decisions if x["kind"] == "prior_art_rejection" and x["hypothesis_id"] == h["hypothesis_id"])
            st.success(f"No compute spent: {d['runs_avoided']} paired runs avoided.")
        if h.get("config_delta"):
            st.markdown(f"**Typed experiment config (delta vs baseline):** `{json.dumps(h['config_delta'])}` · config hash `{h['config_hash']}`")
            with st.expander("Scientist proposal and Critic pre-run review"):
                st.json({"scientist": h.get("proposal"), "critic_pre": h.get("critic_pre")})
        # latest analysis for this hypothesis
        aids = [a for a in D["analyses"] if a["hypothesis_id"] == h["hypothesis_id"]]
        for a in aids:
            st.markdown(f"##### {a['label']}  ·  analysis `{a['analysis_id']}`  ·  {len(a['pairs'])} paired seeds")
            rows = pd.DataFrame([{"seed": p["seed"], "delta": p["delta"], "baseline": p["baseline_val_loss"],
                                  "candidate": p["candidate_val_loss"], "candidate_run": p["candidate_run_id"],
                                  "baseline_run": p["baseline_run_id"],
                                  **({} if STEPS else {"equal-token Δ": p["equal_token_delta"]})}
                                 for p in a["pairs"]])
            sd = a["noise"]["sd"]
            band = alt.Chart(pd.DataFrame({"lo": [-sd], "hi": [sd]})).mark_rect(color=GRAY, opacity=0.18).encode(y="lo:Q", y2="hi:Q")
            zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRAY).encode(y="z:Q")
            pts = alt.Chart(rows).mark_circle(size=110, color=ORANGE).encode(
                x=alt.X("seed:O", title="seed"), y=alt.Y("delta:Q", title="Δ val loss (candidate − baseline); below 0 = better"),
                tooltip=["seed", "delta", "candidate_run", "baseline_run"])
            cc1, cc2 = st.columns([3, 2])
            cc1.altair_chart((band + zero + pts).properties(height=260), use_container_width=True)
            with cc2:
                st.metric("Mean Δ", f"{a['mean_delta']:+.4f}", f"{a['improvement_in_noise_sd']:+.2f}× noise SD (+ = better)", delta_color="off")
                st.write(f"Branch (pre-registered rule): **{a['branch']}**" + ("  ·  ⚠ seeds disagree" if a["seed_disagreement"] else ""))
                if not STEPS:
                    st.write(f"Tokens seen, candidate/baseline: **{a['token_ratio']:.2f}**"
                             + (f"  ·  mean Δ at equal tokens **{a['mean_equal_token_delta']:+.4f}**" if a["mean_equal_token_delta"] is not None else ""))
                    if abs(a["token_ratio"] - 1) > 0.05:
                        st.caption("Throughput differs by >5%: compare the equal-token delta before claiming it “learns better”.")
                if a.get("throughput_ratio") is not None:
                    st.caption(f"Speed cost/benefit (secondary; never part of the decision): candidate/baseline throughput "
                               f"**{a['throughput_ratio']:.2f}×** ({a['candidate_tokens_per_s']:,.0f} vs {a['baseline_tokens_per_s']:,.0f} tokens/s).")
            cf = a["counterfactual"]
            with st.container(border=True):
                st.markdown("**Single-run counterfactual** — what an autoresearch-style keep/revert loop would have decided")
                st.write(cf.get("same_seed_summary") or cf["summary"])
                if cf.get("cross_seed_summary"):
                    st.caption(cf["cross_seed_summary"] + f" (keep rate {cf['n_keep']}/{cf['n_pairings']} pairings.) "
                               "Computed from runs already made; zero extra compute.")
            with st.expander("Per-seed table (click-through to run ids)"):
                st.dataframe(rows, use_container_width=True)
            for e in [e for e in D["evidence"] if e["analysis_id"] == a["analysis_id"]]:
                cl = D["claims"][e["claim_id"]]
                st.markdown(f"**Derived evidence stored next to the cited claim:** this measurement **{e['relation'].replace('_', ' ').upper()}** "
                            f"claim `{cl['claim_id']}` (arXiv:{cl['paper_id']}) — {e['note']}.  \n> “{cl['source_span']}”")
            sr = [x for x in decisions if x["kind"] in ("screening_result", "extra_seeds_result")
                  and x.get("analysis_id") == a["analysis_id"]]
            if sr:
                src = "Critic (LLM, number-checked)" if sr[0]["reading_source"] == "llm" else "deterministic template (number guard fired)"
                st.markdown(f"**Reading** — _{src}_: {sr[0]['reading']}")

# ----------------------------------------------------------------------------- rule + decisions
st.header("3. Pre-registered rule and the decisions it made")
with st.expander("Rule text (written before the first run; implemented in stats.py)", expanded=False):
    st.text(D["rule"])
st.subheader("Why this experiment next")
st.caption("The PI chooses the niche, the objective and the testbed. The lab chooses the experiments: this queue is generated "
           "and ordered by code from the frozen literature corpus (no LLM). Covered by a curated T1 claim → rejected; covered by "
           "an auto-extracted / preprint claim → end of the queue (PI may override); possible overlap with an unextracted paper → "
           "held for PI review; the rest by (best expected outcome among supporting claims, more supporting claims first, alphabetical).")
if D["queue"]:
    seen = {}
    for h in D["hyps"]:
        for d in decisions:
            if d["kind"] == "followup_candidate" and d["hypothesis_id"] == h["hypothesis_id"]:
                seen[d["candidate_key"]] = h["status"]
        if h.get("config_delta") and len(h["config_delta"]) == 1:
            k, v = next(iter(h["config_delta"].items()))
            seen.setdefault(f"{k}={v}", h["status"])
    st.dataframe(pd.DataFrame([{
        "rank": i + 1, "candidate": q["key"], "priority (outcome rank, −#claims, key)": str(q["priority"]),
        "supporting claims": ", ".join(q["supporting_claim_ids"]),
        "literature status": q.get("status", "open").replace("_", " "),
        "covered by": ", ".join(q.get("covering_claim_ids", [])) or "—",
        "this session": STATUS_LABEL.get(seen.get(q["key"], ""), (seen.get(q["key"], "not reached"), ""))[0]}
        for i, q in enumerate(D["queue"])]), use_container_width=True, hide_index=True)
for d in decisions:
    if d["kind"] in ("next_action", "extra_seeds_result", "literature_check", "followup_candidate", "budget_exhausted",
                     "cycle_start", "exploration_result", "finalists_selected", "confirmation", "promotion",
                     "headline_vs_original_baseline",
                     "queue_rejection", "queue_soft_rejection", "pi_review_hold", "prior_art_soft_rejection",
                     "search_degraded"):
        with st.container(border=True):
            if d["kind"] == "next_action":
                na = d["next_action"]
                st.markdown(f"**Decision `{d['decision_id']}` → {na['action'].replace('_', ' ').upper()}** (branch **{na['branch']}**, from analysis `{d['analysis_id']}`)  ←  triggered by `{d.get('triggered_by', '')}`")
                st.write(na["rationale"])
                st.caption("Queue at this decision: " + (", ".join(f"{q['key']} {q['priority']} ← {','.join(q['supporting_claim_ids'])}"
                                                               for q in d["inputs"]["queue"]) or "empty"))
            else:
                st.markdown(f"**{d['kind'].replace('_', ' ').title()}** `{d['decision_id']}`  ←  triggered by `{d.get('triggered_by', '')}`")
                st.json({k: v for k, v in d.items() if k not in ("decision_id", "kind")}, expanded=False)
chain = [d for d in decisions if d["kind"] in ("screening_result", "extra_seeds_result", "next_action", "followup_candidate",
                                               "literature_check", "budget_exhausted")]
st.caption(f"{len(chain)} chained decisions this session; each stores `triggered_by` → the previous decision.")

# ----------------------------------------------------------------------------- learning curves
st.header("4. Learning curves (by tokens seen)")
if analyses:
    aid = st.selectbox("Analysis", list(analyses), format_func=lambda a: f"{a} · {analyses[a]['hypothesis_id']}")
    rows = []
    for p in analyses[aid]["pairs"]:
        for role, rid in (("baseline", p["baseline_run_id"]), ("candidate", p["candidate_run_id"])):
            for pt in D["runs"][rid]["curve"]:
                rows.append({"tokens_seen": pt["tokens_seen"], "val_loss": pt["val_loss"], "role": role,
                             "seed": p["seed"], "line": f"{role}-{p['seed']}", "run_id": rid})
    cdf = pd.DataFrame(rows)
    cdf = cdf[cdf.tokens_seen > 0]
    st.altair_chart(alt.Chart(cdf).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("tokens_seen:Q", title="tokens seen"), y=alt.Y("val_loss:Q", scale=alt.Scale(zero=False), title="validation loss"),
        color=alt.Color("role:N", scale=alt.Scale(domain=["baseline", "candidate"], range=[BLUE, ORANGE])),
        detail="line:N", tooltip=["role", "seed", "tokens_seen", "val_loss", "run_id"]).properties(height=320),
        use_container_width=True)

# ----------------------------------------------------------------------------- throughput / drift
st.header("5. Throughput (secondary metric)")
st.caption("Tokens per second per run, in run order. Speed cost/benefit is shown separately and is never mixed into a decision. "
           "The budget is a fixed number of steps, so throughput only changes wall-clock, not what each run learns.")
ok_runs = [r for r in D["runs"].values() if r["status"] == "ok" and r["train_seconds"] > 0]
if ok_runs and D["baseline"]:
    bh = ExperimentConfig(**D["baseline"]).config_hash()
    tp = pd.DataFrame([{"run_order": r["run_order"], "tokens_per_s": r["tokens_seen"] / r["train_seconds"],
                        "arm": "baseline" if r["config_hash"] == bh else "candidate", "run_id": r["run_id"],
                        "seed": r["seed"]} for r in ok_runs])
    st.altair_chart(alt.Chart(tp).mark_circle(size=80).encode(
        x=alt.X("run_order:Q", title="run order"), y=alt.Y("tokens_per_s:Q", scale=alt.Scale(zero=False), title="tokens / s"),
        color=alt.Color("arm:N", scale=alt.Scale(domain=["baseline", "candidate"], range=[BLUE, ORANGE])),
        tooltip=["run_id", "seed", "run_order", "tokens_per_s"]).properties(height=240), use_container_width=True)
    drift = stats.throughput_drift([RunResult(**r) for r in ok_runs], bh)
    if drift["flag"]:
        st.warning(f"Throughput drift: median baseline-era {drift['early_median']:,.0f} tokens/s vs late-session "
                   f"{drift['late_median']:,.0f} ({drift['relative_change']:+.1%}). Candidates differ in cost, so this is a prompt "
                   "to look at the plot, not a verdict; it cannot change a decision under a step budget.")
    elif drift["relative_change"] is not None:
        st.caption(f"Baseline-era vs late-session median throughput differs by {drift['relative_change']:+.1%} (flag threshold 10%).")

# ----------------------------------------------------------------------------- provenance
st.header("6. Click any number → run id → config, curve, environment")
rid = st.selectbox("Run", list(D["runs"]), format_func=lambda r: f"{r} · seed {D['runs'][r]['seed']} · "
                   f"config {D['runs'][r]['config_hash'][:8]}")
r = D["runs"][rid]
left, right = st.columns(2)
left.json({k: r[k] for k in ("run_id", "status", "seed", "val_loss", "tokens_seen", "steps", "train_seconds", "n_params", "error", "run_order", "started_at")})
right.json({"config": r["config"], "environment": r["env"]}, expanded=False)

st.subheader("Evidence chain")
st.caption("paper → claim → verdict → hypothesis → config → runs → analysis → decision → next decision. "
           "Solid = produced by code or curated by a human; dashed = carries LLM output.")
hid = st.selectbox("Hypothesis", [h["hypothesis_id"] for h in D["hyps"]])
st.graphviz_chart(chain_dot(D, hid), use_container_width=True)
with st.expander("All candidates at a glance (paper claim, coverage, verdict, measured result, relation)"):
    st.dataframe(pd.DataFrame(evidence_grid(D)), use_container_width=True, hide_index=True)
st.caption(f"{len(S.edges())} stored provenance edges")

# ----------------------------------------------------------------------------- LLM log
st.header("7. Every LLM call (recorded)")
ev = pd.DataFrame([{"event": e["event_id"], "role": e["role"], "attempt": e["attempt"], "valid": e["valid"],
                    "model": e["model_served"], "in_tok": e["usage"].get("input_tokens"),
                    "out_tok": e["usage"].get("output_tokens"), "cost_usd": e["cost_usd"],
                    "request_hash": e["request_hash"][:12]} for e in D["events"]])
st.dataframe(ev, use_container_width=True)
with st.expander("Prompts and raw responses"):
    for e in D["events"]:
        st.markdown(f"**{e['event_id']} · {e['role']}**")
        st.code(e["user"][:4000], language="text")
        st.code(e["response_text"], language="json")
