"""Streamlit dashboard: the evidence chain in under 60 seconds (BUILD_PLAN s7).

Read-only over a notebook.sqlite. Every number shown is read from a stored run / analysis /
decision; nothing here computes a statistic. Run: `make app` (SESSION=golden).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from prior_art.store import Store

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"      # baseline / candidate / neutral

st.set_page_config(page_title="Prior Art", page_icon="🔬", layout="wide")


# ----------------------------------------------------------------------------- source
def find_notebooks() -> dict[str, Path]:
    out = {}
    for pat in ("results/*/notebook.sqlite", "work/replay-*/notebook.sqlite"):
        for p in sorted(ROOT.glob(pat)):
            out[str(p.relative_to(ROOT))] = p
    return out


books = find_notebooks()
env_nb = os.environ.get("PRIOR_ART_NOTEBOOK")
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
                snapshot=s.get_meta("snapshot"), objective=s.get_meta("objective"))


D = load(str(books[choice]), books[choice].stat().st_mtime)
analyses = {a["analysis_id"]: a for a in D["analyses"]}
decisions = D["decisions"]

STATUS_LABEL = {
    "rejected_prior_art": ("REJECTED: PRIOR ART", "🟥"),
    "rejected_critic": ("REJECTED BY CRITIC", "🟥"),
    "screened_promising": ("SCREENED: PROMISING", "🟩"),
    "screened_no_improvement": ("SCREENED: NO IMPROVEMENT", "🟨"),
    "screened_harmful": ("SCREENED: HARMFUL", "🟧"),
    "run_failed": ("RUN FAILED", "⬛"),
}

# ----------------------------------------------------------------------------- header
st.title("Prior Art")
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
st.write(f"**Curated literature snapshot:** {snap.get('papers', '?')} papers, {snap.get('claims', '?')} claims "
         f"({', '.join(snap.get('dimensions', []))}). Fetched {snap.get('fetched', '?')} from arXiv. "
         "Verdicts are scoped to this snapshot; the UI never says “novel”.")

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
    right.markdown(f"**{n['n']} baseline seeds**\n\nmean {n['mean']:.4f}\n\n**seed-to-seed SD {n['sd']:.4f}**\n\n"
                   "Every later delta is shown against this band (shaded = ±1 SD).")

# ----------------------------------------------------------------------------- hypotheses
st.header("2. Hypotheses")
for h in D["hyps"]:
    label, icon = STATUS_LABEL.get(h["status"], (h["status"].upper(), "⬜"))
    with st.container(border=True):
        st.markdown(f"#### {icon} {label}  ·  `FIXTURE` {h.get('title', h['fixture_id'])}")
        st.write(h.get("statement", ""))
        pa = h.get("prior_art")
        if pa:
            st.markdown(f"**Prior-art gate:** `{pa['verdict']}` — {pa['label']}  ·  event `{pa['event_id']}`")
            if pa.get("passage"):
                st.markdown(f"> “{pa['passage']}”  \n> — [{pa['paper_title']}]({pa['paper_url']}) (arXiv:{pa['paper_id']}, claim `{pa['claim_id']}`)")
            st.caption(pa["rationale"])
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
                                  "baseline_run": p["baseline_run_id"], "equal-token Δ": p["equal_token_delta"]}
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
                st.write(f"Tokens seen, candidate/baseline: **{a['token_ratio']:.2f}**"
                         + (f"  ·  mean Δ at equal tokens **{a['mean_equal_token_delta']:+.4f}**" if a["mean_equal_token_delta"] is not None else ""))
                if abs(a["token_ratio"] - 1) > 0.05:
                    st.caption("Throughput differs by >5%: compare the equal-token delta before claiming it “learns better”.")
            cf = a["counterfactual"]
            with st.container(border=True):
                st.markdown("**Single-run counterfactual** — what an autoresearch-style keep/revert loop would have decided")
                st.write(cf["summary"])
                st.caption("Same-seed decisions: " + ", ".join(f"seed {k}: {v}" for k, v in cf["same_seed_decisions"].items())
                           + f"  ·  keep rate {cf['n_keep']}/{cf['n_pairings']}. Computed from runs already made; zero extra compute.")
            with st.expander("Per-seed table (click-through to run ids)"):
                st.dataframe(rows, use_container_width=True)
            for e in [e for e in D["evidence"] if e["analysis_id"] == a["analysis_id"]]:
                cl = D["claims"][e["claim_id"]]
                st.markdown(f"**Derived evidence stored next to the cited claim:** this measurement **{e['relation'].upper()}** "
                            f"claim `{cl['claim_id']}` (arXiv:{cl['paper_id']}) — {e['note']}.  \n> “{cl['source_span']}”")
            sr = [x for x in decisions if x["kind"] == "screening_result" and x.get("analysis_id") == a["analysis_id"]]
            if sr:
                src = "Critic (LLM, number-checked)" if sr[0]["reading_source"] == "llm" else "deterministic template (number guard fired)"
                st.markdown(f"**Reading** — _{src}_: {sr[0]['reading']}")

# ----------------------------------------------------------------------------- rule + decisions
st.header("3. Pre-registered rule and the decision it made")
with st.expander("Rule text (written before the first run; implemented in stats.py)", expanded=False):
    st.text(D["rule"])
for d in decisions:
    if d["kind"] in ("next_action", "extra_seeds_result", "literature_check", "followup_candidate", "budget_exhausted"):
        with st.container(border=True):
            if d["kind"] == "next_action":
                na = d["next_action"]
                st.markdown(f"**Decision `{d['decision_id']}` → {na['action'].replace('_', ' ').upper()}** (branch **{na['branch']}**, from analysis `{d['analysis_id']}`)")
                st.write(na["rationale"])
            else:
                st.markdown(f"**{d['kind'].replace('_', ' ').title()}** `{d['decision_id']}`  ←  triggered by `{d.get('triggered_by', '')}`")
                st.json({k: v for k, v in d.items() if k not in ("decision_id", "kind")}, expanded=False)

# ----------------------------------------------------------------------------- learning curves
st.header("4. Learning curves (equal-token view)")
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

# ----------------------------------------------------------------------------- provenance
st.header("5. Click any number → run id → config, curve, environment")
rid = st.selectbox("Run", list(D["runs"]), format_func=lambda r: f"{r} · seed {D['runs'][r]['seed']} · "
                   f"config {D['runs'][r]['config_hash'][:8]}")
r = D["runs"][rid]
left, right = st.columns(2)
left.json({k: r[k] for k in ("run_id", "status", "seed", "val_loss", "tokens_seen", "steps", "train_seconds", "n_params", "error", "run_order", "started_at")})
right.json({"config": r["config"], "environment": r["env"]}, expanded=False)

edges = pd.DataFrame(S.edges())
with st.expander("Provenance graph (decision → analysis → runs; hypothesis → claim → paper)"):
    nid = st.selectbox("Start from", [f"{d['decision_id']}" for d in decisions])
    st.json(S.provenance("decision", nid), expanded=3)
    st.caption(f"{len(edges)} stored edges")

# ----------------------------------------------------------------------------- LLM log
st.header("6. Every LLM call (recorded)")
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
