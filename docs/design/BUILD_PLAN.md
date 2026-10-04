# Noesis Lab — 6-Hour Hackathon Build Plan
**Goal:** ship one complete, auditable agentic-science loop: literature → hypothesis → real experiment → measured decision.

**Governing principle:** LLMs propose and interpret. Deterministic code measures, decides and records.
**Invariant (say this in the pitch):** an LLM mistake can waste compute or skip an idea, but it cannot produce a measured result or a decision. Every number comes from a run; every decision comes from a rule written before the run.

`SPEC.md` is the full research-platform design. **This file is today's scope and wins any conflict.**

---

## 0. Landscape and Positioning (as of Oct 2026)
Judges will compare us to these. Know the one-line difference for each.

| System | What it does | Gap we exploit |
|---|---|---|
| **Karpathy autoresearch** (Mar 2026) and forks incl. autoresearch-mlx (Apple Silicon) | Agent edits training code, trains a fixed 5 min, keeps the change if one run beats the last | One run per decision (keeps noise), no literature, metric fixation. **Closest prior art to our testbed. Cite it; don't pretend the loop is new.** |
| **AutoResearch at production scale** (arXiv 2609.30541, Sep 2026) | Documents five failure modes of autoresearch loops: infra fragility, memory decay, search stagnation, iteration-cost asymmetry, metric fixation | Confirms the problems. We address memory (evidence store), stagnation (literature-seeded proposals) and fixation (paired seeds, labeled screening) |
| **EvoMap AutoResearch** (open-sourced Sep 2026) | Plan → implement → experiment → analyze → blind review, persistent workspace of logs and decisions | Closest on provenance. No prior-art gate against literature before compute |
| **Sakana AI Scientist v1/v2** (Nature 2026) | Idea → code → experiments → full paper | Independent evaluation: all ideas marked novel incl. known ones; 4 of 7 manuscripts had wrong or hallucinated numbers |
| **Google Co-Scientist** (Nature 2026) | Multi-agent hypothesis generation with tournament ranking | Never runs experiments; humans validated everything |
| **Kosmos / Edison** | Literature + data-analysis agents with shared world model, 12-hour runs | Interpretation statements only ~58% accurate; commercial, closed |
| **Agent Laboratory** | Literature → experiments → report | LLM reviewers scored its papers 6.1/10 vs. humans 3.8/10 |

Evidence for our design choice: across 25,000+ agent runs, agents ignored evidence in 68% of reasoning traces and revised beliefs after refutation only 26% of the time. Scaffold design explained 1.5% of performance variance (arXiv 2604.18805). **So we don't ask the LLM to update beliefs; code does.**

**One-line pitch:** *"Autoresearch with a literature memory and honest statistics: it refuses to spend compute on known ideas, tests what the literature hasn't, and never lets an LLM state a number or make a decision."*

**What we claim today (and only this):**
1. A known idea is rejected **before** compute, with an exact source passage.
2. An idea in a setting not found in our curated snapshot is tested with **real paired runs**, measured against the measured noise floor.
3. The same data shows how often a **single-run keep/revert decision** (autoresearch-style) would have been decided by the seed alone. This counterfactual costs no extra compute.
4. The measured result changes the **next decision** through a pre-written rule.

We do **not** claim 10×, confirmation, significance, or global novelty.

---

## 1. Freeze the Scope
- Build the hackathon profile, not the full SPEC.
- One robust judge-facing path, not feature count.
- At ~3:30: no new architecture. At ~5:00: feature freeze and record.

### Core loop we must demonstrate
0. **Noise floor:** baseline on 5 seeds → seed-to-seed spread. Every later delta is shown against it.
1. Literature Agent checks a hypothesis against the curated claim snapshot and returns an exact source passage.
2. A known idea is rejected as prior art before compute is spent ("N runs avoided").
3. A related idea in an untested setting passes the gate and becomes a typed experiment config.
4. A real TinyShakespeare experiment runs on the M5 Max (PyTorch/MPS).
5. Deterministic statistics compute paired deltas; the result is labeled **SCREENING RESULT — NOT CONFIRMED**.
6. **Single-seed counterfactual:** for each seed, what a one-run keep/revert loop would have decided. If seeds disagree, autoresearch's decision was a coin flip here.
7. The result is written back as derived evidence and changes the next choice via a **pre-registered rule** (Section 6).

---

## 2. Reduced Architecture
| Component | Type | Responsibility |
|---|---|---|
| Literature Agent | Claude | Retrieves claims from the snapshot, exact source text, prior-art verdict |
| Scientist Agent | Claude | Proposes hypothesis, mechanism, falsification rule, and an allowed config change |
| Critic Agent | Claude | Before: testability and confounds. After: plain-language reading of the stats object (cannot change any label) |
| Orchestrator | Python | Moves between phases; enforces budget |
| Runner | Python | Real TinyShakespeare runs on MPS; captures failures |
| Statistician / Decider | Python | Paired deltas, noise floor, single-seed counterfactual, pre-registered next-action rule |
| Evidence Store | SQLite (or JSON) | Links hypothesis → paper span → config → run → analysis → decision |
| Dashboard | Streamlit | Shows the evidence chain in under 60 seconds |

Merged / removed vs. SPEC:
- Theorist + Planner → Scientist.
- Skeptic + Interpreter → Critic.
- Lab Manager, Decider, Archivist → plain Python.
- **Matchmaker/Elo deleted:** it let an LLM influence ranking.

LLM → system influence is limited to:
- the prior-art verdict (shown with the passage);
- the Critic's pre-run reject (shown with reasons);
- which hypotheses get proposed.

No LLM output touches a measurement, a label or a decision.

---

## 3. Demo Fixtures (required)
Fixtures are **predefined test hypotheses, not fabricated measurements**. Label them "fixture" in the UI.

- **Fixture A, known in corpus.** A claim with a clear published result that the system should reject with a real paper ID and passage. Candidates (verify the exact passage before use):
  - Pre-LN vs. Post-LN Transformer training stability (Xiong et al., 2020, "On Layer Normalization in the Transformer Architecture").
  - Warmup + cosine decay vs. constant LR.
- **Fixture B, method known, setting untested.** RMSNorm vs. LayerNorm (Zhang & Sennrich, 2019) on a tiny char-level model under a ~60 s budget. **Check before using:** quickly search Semantic Scholar/arXiv for that setting. If a paper covers it, pick another fixture. The UI must say **"not found in curated snapshot (N papers)"**, never "novel".

Note: with a 6–12 paper snapshot, *most* things will be "untested". Judges may notice. Address it head-on in the README: the snapshot is deliberately small for the demo; the verdict is scoped to it; the full SPEC describes the 100–300 paper corpus.

---

## 4. Literature Snapshot: Keep It Tiny
- 6–12 papers covering only the experiment dimensions we expose:
  - optimizer;
  - LayerNorm/RMSNorm and Pre/Post-LN;
  - activation;
  - LR schedule;
  - dropout;
  - positional encoding.
- Per claim: `paper_id` (arXiv ID), title, method, setting, claim, exact `source_span` (verbatim text, copied from arXiv), URL.
- Prefer arXiv abstracts/metadata as the shipped source (clearest licensing). Spot-check that every span is verbatim.
- Not today:
  - live ingestion;
  - citation traversal;
  - large-scale extraction;
  - retrieval evaluation.
- UI name: **"curated literature snapshot"**.

---

## 5. Experiment Design for Today
- Tiny char-level transformer, TinyShakespeare (vendored in the repo with a checksum), fixed wall-clock training budget (~60 s; set after the first timing run). The timer excludes model construction, MPS warmup and eval.
- **Noise floor:** baseline on 5 seeds. Seeds 0–2 are reused as the paired baseline for every candidate (no re-runs).
- Candidate on the same 3 seeds → per-seed paired deltas, mean delta, and the noise floor (baseline seed-to-seed SD) shown on the same chart.
- Also log `tokens_seen` and show the delta at equal tokens if throughput differs (separates "learns better" from "runs faster").
- **No p-values. No "confirmed".** Label: **SCREENING RESULT — NOT CONFIRMED**.
- Either outcome is fine. A negative result for a "known-good" method is a stronger story.
- Compute budget for the whole demo: ~5 noise + 3 fixture B + 3 follow-up ≈ 11 runs ≈ 15 min. Leaves room for re-runs.

---

## 6. Pre-registered Next-Action Rule (write it before the first run)
Implemented in Python, shown verbatim on the dashboard:
- **Promising** (mean delta improves by more than 1× noise SD and all 3 seeds agree in sign) → next action = **extra seeds for the same change** (toward confirmation).
- **No improvement** (mean delta within ±1× noise SD) → next action = **next untested-setting candidate**, chosen by a fixed order of literature priority.
- **Harmful** (worse by more than 1× noise SD) → record a **contradiction with the cited claim**; next action = **Literature Agent targeted check** of whether the snapshot already reports this.

The Scientist may propose the follow-up's content. Which branch fires is decided only by the rule.

---

## 7. Golden Demo Path
1. Load curated snapshot (show N papers, dimensions covered).
2. Set objective: improve validation loss for a tiny transformer under a short budget.
3. Show the noise floor from 5 baseline seeds.
4. Inject Fixture A → Literature Agent returns `known_in_corpus` + exact passage → **REJECTED: PRIOR ART**, "3 runs avoided".
5. Inject Fixture B → `method_known_setting_untested` → Scientist writes typed config → Critic checks confounds.
6. Runner executes 3 paired runs on MPS (live, or show the recorded run with timestamps).
7. Statistician shows per-seed deltas against the noise band, **SCREENING — NOT CONFIRMED**, plus the single-seed counterfactual panel.
8. Result is stored as derived evidence next to the cited claim (agrees / contradicts / inconclusive).
9. The pre-registered rule fires → second decision visibly depends on the first measurement.
10. Click any number → run ID → config, curve, environment.

---

## 8. Reproducibility (remote judging, minimum viable)
Judges likely have no Mac and no API key. Keep only what's cheap:
- `uv` + committed lockfile; pinned Python and torch.
- Pin a **dated** Anthropic model ID in config.
- **Record every LLM call** (request + response) during the final run.
- **`make replay`:** rebuilds the dashboard state from recorded LLM responses + recorded run results. No API key, no training, works on CPU/Linux.
- **`make rederive`:** recomputes all stats and plots from stored runs. Exact match.
- Each run stores git commit, config hash, seed, device, torch/OS versions.
- Results bundle in `results/<session>/` with a SHA256 manifest.
- README: "Verify our claims" (two commands) + **claims table** (claim → evidence ID → how to reproduce → hardware) + known limitations (MPS nondeterminism, tiny snapshot, screening only).
- Optional if time: host the replay dashboard on Streamlit Community Cloud.

---

## 9. Model / Tool Choices
- **Runtime:** one Anthropic model for Literature, Scientist and Critic, with distinct prompts + strict JSON/Pydantic schemas, at temperature 0, pinned dated model ID. A $50 credit is enough if prompts stay short and responses are cached. A second provider for the Critic is a robustness feature, not MVP.
- **Cursor:**
  - Grok 4.7 for the first repo-wide build;
  - Composer 2.5 for fast UI/small fixes;
  - a GPT-5.6 variant for hard debugging/architecture (check the exact name in Cursor's model picker).

  Whatever you use, **one person owns the schemas and merges**.

---

## 10. Three-Person Team Split
| Owner | Primary work | Deliverables |
|---|---|---|
| **You: Core + Integration** | Repo, schemas, orchestrator, testbed/runner, stats + pre-registered rule, Claude API wrapper with record/replay, final merges | One command that runs the golden path repeatedly; `make replay` / `make rederive` |
| **Teammate 1: Literature + Evidence** | 6–12 papers; verbatim passages; claim snapshot; Fixtures A and B; setting check for Fixture B; landscape table for README | `data/claims.json` with auditable spans; README "Related work" section |
| **Teammate 2: Dashboard + Submission** | Streamlit against the agreed schema (incl. noise band, counterfactual panel, rule display); README skeleton, claims table, video scripts, one-pager | `app.py` + submission assets |

Integration rule: schemas are frozen at 0:20 and changed only by the core owner. Don't let three AI assistants invent three incompatible systems.

---

## 11. Six-Hour Timeline
| Time | Target |
|---|---|
| 0:00–0:20 | Freeze scope. Repo + shared schemas/interfaces + pre-registered rule text. Hand out contracts. |
| 0:20–1:30 | Parallel: training harness (timing run → set budget); claim snapshot + fixtures; dashboard shell on mock data. |
| 1:30–2:30 | **5-seed noise floor** + Fixture B candidate on 3 paired seeds; structured RunResult objects; stats + counterfactual. |
| 2:30–3:30 | Claude agents (record/replay wrapper from the start) + retrieval + verdicts; golden path from one command. |
| 3:30–4:15 | Integrate UI; run end-to-end repeatedly. **No new architecture after this.** |
| 4:15–5:00 | Fix failures only. Final recorded session → results bundle. Verify `make replay` on a teammate's machine. README + claims table. |
| 5:00–6:00 | **FEATURE FREEZE.** Record demo + tech videos, screenshots, public repo, zip, submission assets. |

---

## 12. Cut / Keep
**CUT**
- Matchmaker / Elo tournament
- 100+ paper corpus, live ingestion, citation traversal
- retrieval benchmark
- confirmation tests and multiple-comparison statistics
- Auditor / sealed audit split / full competitor benchmark (the single-seed counterfactual stands in for it)
- re-execute-trajectory mode, CI, devcontainer
- hash-chain provenance
- source patches / code mutation
- second LLM provider

**KEEP**
- exact paper passages
- three distinct LLM roles
- structured JSON/Pydantic outputs
- typed experiment configs
- real MPS execution
- paired screening runs
- **noise floor**
- **single-seed counterfactual**
- failures logged
- deterministic measurements and **pre-registered** decisions
- evidence provenance
- **record/replay**
- a second decision that changes because of the first measurement

---

## 13. Definition of Done
- One demo path works end to end from a single command without manual surgery.
- Fixture A rejected with an exact, verifiable passage (paper ID + verbatim text).
- Fixture B passes and triggers real paired MPS runs.
- Noise floor shown; screening output shows stored run IDs, per-seed deltas, single-seed counterfactual; labeled **NOT CONFIRMED**.
- The result is stored as derived evidence; the pre-registered rule visibly picks the next action.
- `make replay` works on a clean clone with no API key; `make rederive` reproduces every number exactly.
- Streamlit makes the chain understandable in under 60 s.
- README has: positioning vs. autoresearch and others, claims table, limitations.
- Public repo + zip + demo video + tech video + one-page report are ready before the internal deadline.

---

### Sources for the landscape table
- Karpathy autoresearch: https://rywalker.com/research/autoresearch · autoresearch-mlx: https://github.com/trevin-creator/autoresearch-mlx
- AutoResearch at production scale: https://arxiv.org/abs/2609.30541
- EvoMap AutoResearch: https://www.opensourceforu.com/2026/09/evomap-open-sources-autoresearch-for-ai-agents/
- AI Scientist (Nature 2026): https://www.nature.com/articles/s41586-026-10265-5 · independent evaluation: https://arxiv.org/abs/2502.14297
- Co-Scientist (Nature 2026): https://www.nature.com/articles/s41586-026-10644-y
- Kosmos: https://www.alphaxiv.org/abs/2511.02824
- Agent Laboratory: https://arxiv.org/abs/2501.04227
- Agents ignore evidence: https://arxiv.org/abs/2604.18805
