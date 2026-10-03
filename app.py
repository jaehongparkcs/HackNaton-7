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

from noesis_lab import stats
from noesis_lab.evidence_chain import chain_dot, evidence_grid
from noesis_lab.schemas import ExperimentConfig, RunResult
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
                queue=s.get_meta("candidate_queue") or [], baseline=s.get_meta("baseline_config"))


D = load(str(books[choice]), books[choice].stat().st_mtime)
analyses = {a["analysis_id"]: a for a in D["analyses"]}
decisions = D["decisions"]
STEPS = profile.get("budget_mode") == "steps"        # equal-token delta is redundant under a step budget

STATUS_LABEL = {
    "rejected_prior_art": ("REJECTED: PRIOR ART", "🟥"),
    "rejected_critic": ("REJECTED BY CRITIC", "🟥"),
    "screened_promising": ("SCREENED: PROMISING", "🟩"),
    "screened_no_improvement": ("SCREENED: NO IMPROVEMENT", "🟨"),
    "screened_harmful": ("SCREENED: HARMFUL", "🟧"),
    "run_failed": ("RUN FAILED", "⬛"),
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
        st.markdown(f"#### {icon} {label}  ·  `{h.get('origin', 'FIXTURE')}` {h.get('title', h['fixture_id'])}")
        st.write(h.get("statement", ""))
        if h.get("origin") == "GENERATED":
            st.caption(f"Chosen by code from the literature queue. Priority tuple {h.get('queue_priority')}; "
                       f"supporting claims {', '.join(h.get('supporting_claim_ids', []))}.")
        pa = h.get("prior_art")
        if pa:
            st.markdown(f"**Prior-art gate:** `{pa['verdict']}` — {pa['label']}  ·  event `{pa['event_id']}`")
            if pa.get("passage"):
                st.markdown(f"> “{pa['passage']}”  \n> — [{pa['paper_title']}]({pa['paper_url']}) (arXiv:{pa['paper_id']}, claim `{pa['claim_id']}`)")
            if pa.get("claim_id"):
                cov = {True: "YES", False: "NO"}.get(pa.get("covers_our_setting"), "n/a")
                st.markdown(f"**Covers our setting: {cov}** · _human-curated_ — {pa.get('coverage_note') or ''}")
            st.caption(f"LLM judged only “same comparison: {pa.get('same_comparison')}”. {pa['rationale']}")
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
st.caption("The PI chooses the objective and the testbed. The lab chooses the experiments: this queue is generated and "
           "ordered by code from the literature snapshot (no LLM). Priority = (best expected outcome among supporting "
           "claims: improves < no_worse < context, more supporting claims first, alphabetical).")
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
        "this session": STATUS_LABEL.get(seen.get(q["key"], ""), (seen.get(q["key"], "not reached"), ""))[0]}
        for i, q in enumerate(D["queue"])]), use_container_width=True, hide_index=True)
for d in decisions:
    if d["kind"] in ("next_action", "extra_seeds_result", "literature_check", "followup_candidate", "budget_exhausted"):
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
