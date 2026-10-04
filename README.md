# Noesis Lab

> *Autoresearch with a literature memory and honest statistics: it searches the live literature, refuses to spend compute on what is already covered, tests the gaps, and never lets an LLM state a number or make a decision.*

One complete, auditable agentic-science loop: **niche → literature search → gap graph → hypothesis → real experiment → measured decision.**

**Governing principle.** LLMs propose, extract and interpret. Deterministic code measures, scores, ranks, decides and records.
**Invariant.** An LLM mistake can waste compute or skip an idea, but it cannot produce a measured result, a score, a label, a rank or a decision. Every number comes from a run or a formula; every decision comes from a rule written before the run.

Design documents, in the order they were built: [BUILD_PLAN.md](BUILD_PLAN.md) (hackathon profile), [FIXES.md](FIXES.md) (step budget, rule on every result, generated queue), [LIT_SEARCH.md](LIT_SEARCH.md) (live search), [NEXT_VERSION.md](NEXT_VERSION.md) with [EXPLORE.md](EXPLORE.md) and [GRAPH_GAPS.md](GRAPH_GAPS.md) (hypothesis engine), [FINAL_FIXES.md](FINAL_FIXES.md) (polish, a sync hazard, speed). [SPEC.md](SPEC.md) is the fuller platform design.

**Two exhibits.** [`results/golden/`](results/golden/) shows the **core loop** (prior-art rejection, paired runs against the noise floor, the rule on every result). [`results/explore2/`](results/explore2/) shows the **hypothesis engine** (gap graph, explore → confirm → promote, prediction scoring). The dashboard opens on `explore2`, then `golden`.

## Status: what is recorded and what is only built

| | State |
|---|---|
| **`results/golden/`** — exhibit 1, core loop | Live Claude, live arXiv search, real MPS training. Recorded before the hypothesis engine existed: its follow-ups came from the literature-ordered queue. Rule v1, gate v1. `make verify rederive replay` pass. |
| **`results/explore2/`** — exhibit 2, hypothesis engine | Live, 160 papers, 185 claims, real MPS training, 28 runs. Protocol v2 (rule v2, gate v2), the wider testbed and the explore → confirm → promote loop: RoPE was confirmed and promoted, three later finalists were measured on it, and four gap predictions were scored. Prediction v1 (combinations make no directional prediction). Recorded with one limitation: a thin mechanism graph (2 shared mechanism nodes, want ≥ 3). `make verify rederive replay` pass. |
| `results/explore1/` — superseded | Live, 120 papers, 128 claims. The scout failed (a schema bug, since fixed), so only the per-building-block directions were searched. One gap hypothesis ran (dropout: predicted better, measured harmful, a **miss**). It was recorded although the gate check had reported UNSTABLE rows, which is why recording now stops on that. Rule v1, gate v1. Verify, rederive and replay pass. Kept as an exhibit of what went wrong. |
| Prediction v2 (combination predictions), timestamps, the speed work (FINAL_FIXES B1–B4, B6) | Built and tested with a scripted LLM and recorded API responses. Not yet recorded live. |
| `compare-ideation`, `validate-gaps`, stagnation redirect | Not built. |

Every bundle replays under the config and protocol versions it was recorded with, so the recorded bundles are untouched by everything after them. `make verify` also fails if a bundle holds any file its manifest does not list.

## What we claim (and only this)

Backed by the recorded bundles (`golden`, and again in `explore2`):

1. A known idea is rejected **before** compute, with an exact source passage.
2. An idea in a setting not found in the retrieved literature is tested with **real paired runs**, against the measured **noise floor**.
3. The same data shows what a **single-run keep/revert decision** (autoresearch-style) would have decided, per seed. This costs no extra compute.
4. The measured result changes the **next decision** through a pre-registered rule, applied to every screening result.
5. The lab chooses its own experiments. The PI chooses the niche, the objective and the testbed; code generates and orders the queue from the frozen literature.
6. The literature is **searched live, then frozen**: every paper comes from an API response, every quoted claim is verbatim, and replay needs no network.

Backed by `explore2`:

7. The lab finds where the retrieved literature is silent or disagrees for our setting and queues those gaps. Each hypothesis is labeled transfer, combination or resolution, and is "not found in N retrieved papers", never "novel". New mechanisms that need new code are shown on the map but not run.
8. The lab explores gap hypotheses on one seed, confirms the best with the paired protocol, and reports how much the single-seed estimate shrank. A finalist that holds on every seed becomes the incumbent (RoPE did); later changes are credited only with their effect on top of it.
9. The lab scores its own gap predictions after screening: 1 hit, 1 miss, 1 null, 1 no prediction.

We do **not** claim 10×, confirmation, statistical significance or novelty. Results are labeled **SCREENING RESULT — NOT CONFIRMED**. Graph scores are structural hints, not probabilities.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all dependencies are installed from the lockfile.

```bash
make setup               # one-time install (on Linux, torch comes from the CPU-only index)
make test                # ~330 tests on CPU with a mocked LLM and recorded API responses; no key, no GPU, no network
make verify-snapshot     # every curated claim span is verbatim in its arXiv abstract
make smoke               # CPU pipeline check with the MOCK LLM (~10 s). Not a result.
make app                 # dashboard over every notebook in results/ and work/
```

**A real session** (live Claude, live search, real training; needs Apple Silicon and an API key):

```bash
cp .env.example .env     # set ANTHROPIC_API_KEY; optionally OPENALEX_MAILTO (a team address)
make timing              # ~40 s: tokens/s on this machine (the budget is steps, so there is nothing to tune)

make record SESSION=<name> NICHE=niche.yaml     # search → gate stability ×3 → session → verify → rederive → replay
make app
```

Before a recording, `make rehearse CORPUS=work/corpus-<name>` runs the same live pipeline on a frozen corpus in ~5 minutes (500 steps, one cycle of 3 explorations) into `results/rehearse/`. It is labeled "not results", like `smoke`, and never used for a number. The live targets (`search`, `record`, `golden`, `rehearse`) run under `caffeinate -i` on macOS so the machine cannot sleep mid-session.

`make record` is the only way to record an engine session. It **aborts before the session** if any gate row is UNSTABLE (a different verdict or action between repeats). The steps, if you want them one at a time:

```bash
make search NICHE=niche.yaml                                        # search + scout + extract, frozen to work/corpus-niche/ (no training)
uv run python -m noesis_lab lit-dryrun --corpus work/corpus-niche --repeat 3   # exits non-zero on an UNSTABLE row; flags fixture leakage
make golden CORPUS=work/corpus-niche SESSION=<name>                 # record a session on that frozen corpus
make verify rederive replay SESSION=<name>
```

- `make golden` with neither `NICHE` nor `CORPUS` uses the curated 10-paper snapshot only and never touches the network.
- A session whose corpus contains scout directions runs the hypothesis engine. It also searches arXiv once per hypothesis during the session (the tighten pass), so record with a network connection.
- A session refuses to overwrite an existing bundle unless you pass `--force` to the CLI. `golden`, `explore1` and `explore2` exist; use another `SESSION` name.
- Keep the repository outside iCloud-synced folders (`~/Desktop`, `~/Documents` with "Desktop & Documents" on). iCloud writes `<name> 2.<ext>` conflict copies when a session rewrites files, and syncing a live SQLite file can corrupt it. `make verify` reports any such file as "not in MANIFEST.json".
- After the search, read the line "mechanism nodes linked to 2+ runnable methods". Below 3, combination gaps will be sparse; the dashboard reports the number either way.
- Budgets (runs, follow-up candidates, minutes, LLM spend) are in [config.yaml](config.yaml). Hitting one is recorded as a `budget_exhausted` decision.
- Without `make`: `uv run python -m noesis_lab {session,search,lit-dryrun,replay,rederive,verify,timing} --help`.

## Verify our claims

| Level | Command | Needs | Guarantee |
|---|---|---|---|
| R1 re-derive | `make rederive` | CPU, seconds | Recomputes every statistic, every queue and (for engine sessions) every gap score from the stored runs and frozen claims; exact match required |
| R2 replay | `make replay` | CPU, **no API key, no network** | Rebuilds the whole session from recorded LLM responses, recorded runs and the frozen corpus; the final state digest must be identical |
| integrity | `make verify` | nothing | SHA256 manifest of the bundle, including the frozen corpus |

```bash
make setup
make verify rederive replay                    # SESSION=golden by default
make verify rederive replay SESSION=explore2
make app
```

Nothing is retrained, searched or sent to a model in R1/R2.

### Claims table (recorded bundle `results/golden`)

Recorded 2026-10-03 on MPS: 16 runs, 15 loop LLM calls ($0.20) plus the search ($0.64). Corpus: 80 papers retrieved for 20 queries, 96 claims (13 curated, 83 auto-extracted). Noise floor SD 0.0116 over 5 baseline seeds.

| Claim | Evidence in the bundle | Reproduce | Hardware |
|---|---|---|---|
| A known idea is rejected before compute, with the passage | `dec_006` `prior_art_rejection` for Fixture A, claim `c01`, arXiv:2002.04745; 3 runs avoided | `make replay` | none |
| The frozen literature rules candidates out before any LLM gate | `dec_005` `queue_rejection` (squared ReLU, curated claim); `dec_002`–`dec_004` `queue_soft_rejection` (SwiGLU, dropout, Post-LN, auto-extracted claims) | `make rederive` | none |
| Noise floor of the baseline | `noise_baseline`, 5 runs, mean 1.5390, SD 0.0116 | `make rederive` | MPS |
| Fixture B (RMSNorm) measured against the noise floor | `ana_a6ea28bc9a`: no improvement (+0.23× SD) | `make rederive` | MPS |
| A single-run loop would have decided on the seed | same analysis: seed 0 keep, seeds 1–2 revert; 5 of 15 cross-seed pairings keep | `make rederive` | MPS |
| The rule fired on every screening result and each branch was taken | `dec_008` next candidate → `dec_011` literature check (constant schedule, harmful, −1.58× SD) → `dec_015` extra seeds (RoPE, promising, +1.67× SD) → `dec_016` | `make replay` | none |
| Extra seeds shrink the estimate | RoPE: +1.67× SD on 3 seeds (`ana_ebb74f5046`), +1.14× SD on 5 (`ana_dc011cb512`); still a screening result | `make rederive` | MPS |
| Follow-ups were chosen by code from the literature | `dec_010`, `dec_014` `followup_candidate` with `supporting_claim_ids`; meta `candidate_queue` | `make rederive` | none |
| A bound stopped the session and was recorded | `dec_017` `budget_exhausted` (`max_followup_candidates`) | `make replay` | none |
| Speed is reported, never decided on | RoPE throughput 0.88× and RMSNorm 0.92× baseline, stored on the analyses | `make rederive` | MPS |

No row of this table comes from the hypothesis engine; those are in the `explore2` table below.

### Claims table (recorded bundle `results/explore2`)

Recorded 2026-10-04 (UTC) on MPS: 28 runs, 48 loop LLM calls ($0.92) plus the search ($1.58). Corpus: 160 papers retrieved for 28 queries (361 retrieved, 33 excluded, 170 ranked out), 185 claims (13 curated, 172 auto-extracted). Noise floor SD 0.0088 over 5 baseline seeds (mean 1.5367). Protocol: rule v2, gate v2, prediction v1.

| Claim | Evidence in the bundle | Reproduce | Hardware |
|---|---|---|---|
| A known idea is rejected before compute, with the passage | `dec_005` `prior_art_rejection` for Fixture A, claim `c01`, arXiv:2002.04745; 3 runs avoided | `make replay` | none |
| Explore cheaply: six gap hypotheses on one seed, finalists picked by code | `dec_007`–`dec_012` `exploration_result` (cycle 1); `dec_013` `finalists_selected`: RoPE, RMSNorm | `make rederive` | MPS |
| Confirm rigorously: RoPE holds on the paired protocol and on extra seeds | `dec_014` screening (`ana_ec4c8cfc13`, promising, +1.86× SD, 3 seeds) → `dec_015` confirmation → `dec_016` extra seeds → `dec_017` (`ana_7005235c9c`, promising, +1.28× SD, 5 seeds) | `make rederive` | MPS |
| A single-seed winner shrinks to nothing | RMSNorm: +1.77× SD on one seed (`dec_007`), −0.05× SD paired (`dec_018`, `ana_72e97f8dc0`, no improvement); shrinkage in `dec_019`; next candidate `dec_020` | `make rederive` | MPS |
| Promotion: the confirmed change becomes the incumbent | `dec_021` `promotion` (RoPE); its own 5 runs are `noise_incumbent_1` (SD 0.0052); later deltas are on it | `make rederive` | MPS |
| A change on the incumbent is credited with its own effect only | `dec_028` finalists → `dec_029` ReLU² + RMSNorm on top of RoPE: **no improvement vs incumbent (−0.67× SD, `ana_21ef5720f7`)** → `dec_030` confirmation → `dec_031` vs original baseline +1.46× SD (`ana_62d9b5aea9`), **which includes RoPE's gain** (informational) | `make rederive` | MPS |
| A harmful result triggers the literature check | `dec_037` finalists → `dec_038` constant schedule on RoPE: harmful, −1.87× SD, every seed worse (`ana_61c03db3cf`) → `dec_039` → `dec_040` vs original +0.75× SD (no improvement) → `dec_041` next action → `dec_042` literature check: not found | `make replay` | none |
| The engine scores its own predictions | screening decisions carry `predicted_direction` / `prediction_outcome`: RoPE hit (`dec_014`), RMSNorm null (`dec_018`), ReLU² + RMSNorm no prediction (`dec_029`), constant schedule miss (`dec_038`) | `make replay` | none |
| Speed is reported, never decided on | RoPE throughput 0.87×, RMSNorm 0.90× baseline, stored on the analyses | `make rederive` | MPS |

**Replication across sessions.** RoPE was screened in both exhibits with the same protocol on different noise-floor draws:

| Session | RoPE, 3 seeds | RoPE, 5 seeds |
|---|---|---|
| golden | 1.67 SD | 1.14 SD |
| explore2 | 1.86 SD | 1.28 SD |

Both promising; both shrink with more seeds. Still screening results, not confirmations.

**Shrinkage.** RoPE's single-seed estimate was −0.033 val loss (+3.71× SD, seed 0); the paired mean was −0.016 on 3 seeds and −0.011 on 5. The one-run number was about twice the paired one; RMSNorm's one-run "win" disappeared entirely. That is the single-run keep/revert weakness we pitch against, measured inside one session.

**Prediction scorecard** (`explore2`, prediction v1): 1 hit (RoPE, "+", promising), 1 miss (constant schedule, "0" = no worse, measured harmful), 1 null (RMSNorm, "+", no improvement), 1 no prediction (ReLU² + RMSNorm, a combination). Counts, not rates: four is too few for a rate.

**Extraction validation** (auto-extraction vs the 9 human-labeled curated claims with a config change): direction 5/9, setting 3/9, config mapping 6/9 in `explore2`, after the counter-examples added to `prompts/extract.md`; before (`explore1`): direction 4/9, setting 3/9, config mapping 6/9.

## How it works

```
niche.yaml (PI) ─► query plan (LLM) + scout directions (LLM) ─► arXiv / OpenAlex (API)
                ─► dedupe, filter, rank (code) ─► extract claims + mechanisms (LLM)
                ─► verify quotes, map to config, coverage, tier (code) ─► FREEZE corpus/
                                        │
                 graph of methods, mechanisms, settings, outcome (code)
                 gap formulas ─► gap_score ─► ranked queue (code)
                                        │
   Fixture A, Fixture B, then the head of the queue:
        Scientist (LLM) writes the hypothesis for the code-chosen change
        tighten: one targeted search for exactly that change (API + LLM extract, frozen)
        Literature Agent (LLM): "same comparison?"  ─► code: coverage + tier ─► reject | soft-reject | hold | run
        Critic (LLM): accept / reject BEFORE compute
        Runner (PyTorch, MPS): paired runs, baseline runs reused
        Statistician (code): deltas vs noise floor, single-run counterfactual, label
        Decider (code): pre-registered rule ─► extra seeds | literature check | next candidate
        Evidence store (SQLite): paper → claim → verdict → hypothesis → config → runs → analysis → decision
```

Sessions without scout directions (curated-only, or `golden`) skip the graph and use the literature-ordered queue described below.

| Component | Type | Where |
|---|---|---|
| Query plan, scout, extraction | LLM, recorded in `corpus/llm.jsonl` | [noesis_lab/search/](noesis_lab/search/), [prompts/](prompts/) |
| Retrieval, dedupe, rank, coverage, tiers, freeze | plain Python | [noesis_lab/search/](noesis_lab/search/) |
| Graph, gap formulas, gap score, gap queue | pure functions | [noesis_lab/gaps.py](noesis_lab/gaps.py) |
| Literature Agent, Scientist, gap Scientist, Critic | LLM (one pinned model, strict JSON) | [noesis_lab/literature.py](noesis_lab/literature.py), [noesis_lab/agents.py](noesis_lab/agents.py) |
| Orchestrator, Runner, Statistician, Decider | plain Python | [noesis_lab/orchestrator.py](noesis_lab/orchestrator.py), [noesis_lab/runner.py](noesis_lab/runner.py), [noesis_lab/stats.py](noesis_lab/stats.py) |
| Testbed | tiny char-level GPT, TinyShakespeare (vendored + checksum) | [noesis_lab/testbed/](noesis_lab/testbed/) |
| Evidence store, bundle, re-derive | SQLite, SHA256 manifest | [noesis_lab/store.py](noesis_lab/store.py), [noesis_lab/bundle.py](noesis_lab/bundle.py), [noesis_lab/rederive.py](noesis_lab/rederive.py) |
| Dashboard | Streamlit, read-only | [app.py](app.py), [noesis_lab/gap_views.py](noesis_lab/gap_views.py), [noesis_lab/evidence_chain.py](noesis_lab/evidence_chain.py) |

### What each LLM call may and may not do

| Call | Writes | Code decides |
|---|---|---|
| Query plan | search queries | the cap; one extra query per runnable change |
| Scout | directions, search terms, remembered titles | everything is only a search; a title counts only if the API returns that paper |
| Extraction | method, quote, direction, setting fields, mechanism quote | quote and mechanism must be verbatim; config mapping must be allowlisted; coverage and tier |
| Literature Agent | the ids of all retrieved claims that test the same comparison | which claim determines the verdict, the verdict, the gate outcome, the passage shown |
| Scientist | statement, mechanism, falsification rule, typed config | the allowed fields; for gap hypotheses the exact delta and that `grounded_in` is on the gap's path |
| Critic (before) | accept / reject with reasons | nothing downstream of an accept |
| Critic (after) | a plain-language reading | any number not in the facts table discards the reading |

### How the invariant is enforced (not just promised)

- **No field to put a number in.** LLM output schemas have no measurement, label, decision, verdict, tier, coverage, score or priority fields (tested in `tests/test_schemas_model.py`).
- **Passages come from the frozen corpus, not the LLM.** The agent picks a `claim_id` that code validates against the retrieved list; the quoted text is copied from the corpus. Every span is checked verbatim against its arXiv abstract when the corpus is loaded.
- **Prior-art verdicts are derived by code.** The LLM answers one question (does a retrieved claim test the same change?). Whether the claim covers our setting is a human flag on curated claims and a written rule on auto-extracted ones; `stats.prior_art_verdict` and `stats.gate_outcome` combine them. There is no LLM channel for "does the setting match".
- **Papers come from APIs.** Nothing the LLM names becomes a paper unless the API returns it. Raw responses are frozen in the bundle.
- **Gaps, scores, labels and queue order are formulas.** `noesis_lab/gaps.py` has no LLM and no I/O; `make rederive` recomputes its output from the frozen claims.
- **Configs are typed.** The Scientist's changes are parsed into a Pydantic allowlist, restricted per hypothesis to the fields it names, and range-checked. The experiment language is one allowlisted change, or two that touch different fields. No LLM-written model code.
- **Number guard.** The Critic's post-run reading may only quote numbers present in the code-computed facts table; otherwise it is discarded and replaced with a deterministic template (the swap is recorded).
- **Pre-registered rule, versioned.** Which branch fires is computed by `stats.analyze` and `stats.next_action`. The rule text and its version are stored in the bundle and shown verbatim on the dashboard.
- **Replay under the bundle's own config.** Budgets, rule version, gate version and engine settings come from the bundle's `config.snapshot.yaml`, so editing [config.yaml](config.yaml) later cannot change how a recorded session replays.

### Experiment design

- Tiny char-level Transformer (4 layers, 128-d, ~0.8M parameters), TinyShakespeare, a **fixed budget of 2,000 optimizer steps** per run (~30 s on a fast M-series Mac, ~60 s on a slower one). The timer (used only for the throughput metric) excludes model construction, a throw-away prewarm and eval.
- **Noise floor:** baseline on 5 seeds, SD of final validation loss. Seeds 0–2 are reused as the paired baseline for every candidate. The SD is itself an estimate: 0.0116 in `golden` and 0.0083 in `explore1` for the same baseline config. The dashboard shows an exhaustive-bootstrap 95% interval (0.0043–0.0154 and 0.0024–0.0111). We never pool it across sessions.
- **Paired deltas** (candidate − baseline, same seed; negative = better), mean delta, shown against ±1 SD. No p-values.
- Tokens are equal by construction, so the equal-token delta is redundant: the field is still stored (it matters for time-budget bundles) and the dashboard hides it when `budget_mode == steps`.
- **Throughput** (tokens/s per run, candidate/baseline) is logged and shown as a separate, secondary "speed cost/benefit" metric. It never enters a decision. The dashboard plots it against run order and flags baseline-era vs late-session drift above 10% (informational).
- **Single-run counterfactual:** the headline is the decision a single-run keep/revert loop with a fixed seed would have made (one per paired seed). Secondary: all (candidate seed × baseline seed) pairings, "if the seed varies between runs", worded by how many pairings flip: none → "seed never changed the decision"; at most 2 → "almost always KEEP/REVERT (k of n pairings flip)"; otherwise "decided by seed luck (k of n)".

### Why steps, not seconds

Under a wall-clock budget the conclusion depended on the machine. Same config (`6ea351a6bc162b08`, SwiGLU), two Macs/sessions: ~31M tokens per run → **4.0 SD worse** (`results/mps-mock`); ~16M tokens per run (1,984 steps in 60 s) → **1.9 SD better** (live Noesis Lab run). A fixed step budget removes the hardware from the question: every machine sees identical `steps` and `tokens_seen`. This is the autoresearch weakness we pitch against, so we do not reproduce it. The two old bundles are kept as an exhibit, labeled: *different machines, wall-clock profile, a mock-LLM session for one of them*; they are not results.

### The pre-registered rule (version 2)

Let SD be the baseline seed-to-seed SD and improvement = baseline − candidate val loss, averaged over paired seeds.

1. **Harmful**: worse by more than 1×SD **and every paired seed is worse**. Record the result against the cited claims (see "Evidence relations do not overclaim"); next action is a targeted Literature Agent check.
2. **Promising**: improvement > 1×SD and every paired seed improves. Next action is extra seeds for the same change.
3. **No improvement**: otherwise. A mean beyond 1×SD in either direction with seeds that disagree in sign is "no improvement", flagged as seed disagreement.

**Why there is a v2, and why it is not retroactive.** Rule v1 required agreement for *promising* but not for *harmful*. In `explore1`, RMSNorm came out "harmful" at 1.07 SD with per-seed deltas −0.010 / +0.028 / +0.009: one seed decided it. v2 was written on 2026-10-03, after `explore1` and before any v2 session, and makes the requirement symmetric. `golden` and `explore1` keep rule v1: the version is stored in each bundle (`protocol:` in its config snapshot, `rule_version` in the notebook) and `make rederive` uses it. We do not re-label old results with a rule written after seeing them.

The rule is applied to **every** screening result. Extra seeds run at most once per hypothesis. Bounds (`budget.max_runs` 32, `budget.max_followup_candidates` 4) are recorded as `budget_exhausted` decisions when hit. Every decision stores `triggered_by` → the previous decision.

### The prior-art gate (version 2)

Gate v1 asked the LLM for the one "nearest" claim. On the larger corpus that flipped between a curated claim (reject) and a preprint (run) from one call to the next. In gate v2 the LLM lists **every** retrieved claim that tests the same comparison, and `stats.gate_decision` picks the verdict, strongest first: a curated claim that covers our setting → reject; else an uncontested auto-extracted covering claim → soft-reject; else a same-comparison claim that does not cover us → method known, setting untested; else not found. Ties go to the lowest claim id, and the displayed passage is the claim that determined the verdict. Which claim the LLM mentions first can no longer change the outcome. `lit-dryrun --repeat 3` must show every row STABLE before a recording.

### The lab picks the next experiment

The two fixtures run first and are labeled `FIXTURE`. Everything after is labeled `GENERATED` and comes from a queue computed by code. There are two queue builders; which one a session uses depends on its frozen corpus.

- **Gap queue** (corpus has scout directions): one item per config delta, ordered by `gap_score`. See "Hypothesis engine".
- **Literature-ordered queue** (curated-only sessions and `golden`): `stats.candidate_queue`. Candidates are every allowlisted single-field change (`norm`, `norm_position`, `activation`, `pos_encoding`, `schedule`, `optimizer`, `dropout`; never lr or batch size) that at least one claim maps to. Order: claims with `improves` before `no_worse` before `context`, then more supporting claims, then alphabetical.

Both apply the same literature status first: covered by a curated claim → rejected; covered by an auto-extracted claim → end of the queue, not run; possible overlap with an unextracted paper → held for PI review. The dashboard's "Why this experiment next" panel shows the queue, and `make rederive` recomputes it at every decision.

*Gap we filled before any run:* the plan did not say what happens when the mean improvement exceeds 1×SD but the seeds disagree in sign. We route that to "no improvement" and flag it as seed disagreement. It is in the rule text and has a test.

## Live literature search that decides what gets run

The PI defines a niche ([niche.yaml](niche.yaml), or the dashboard sidebar form, which only writes a `niche_<slug>.yaml` and shows the command). The lab searches the live literature for it, extracts checkable claims, maps what is covered, and uses that map to block or deprioritize overlapping ideas and to order the queue. Code is in [noesis_lab/search/](noesis_lab/search/).

```
NicheSpec ─► (LLM) query plan ─► (API) arXiv + OpenAlex ─► (code) dedupe + filter + rank
          ─► (LLM) extract fields per abstract ─► (code) verify quote, map to config, coverage, tier
          ─► freeze into results/<session>/corpus/ ─► coverage grid + paper map ─► queue ─► gate ─► runs
```

Rules it keeps:

1. **APIs find papers; LLMs never recall them.** Every paper has an arXiv id and a verbatim abstract from an API response. Raw responses are cached in `corpus/raw/`.
2. **LLMs extract; code checks and decides.** Claude proposes search queries and transcribes fields from abstracts. Code caps the queries and adds one per runnable config change, drops any claim whose quote is not a verbatim substring, nulls any `config_change` outside the allowlist, and computes coverage, tier, verdict, gate outcome and queue order.
3. **Search live, then freeze.** A session reads only `results/<session>/corpus/` (niche, query plan, raw responses, papers with ranking components, claims with tier and coverage, drops with reasons, the search's own LLM recordings). The manifest hashes all of it; replay and rederive use no network (tested with the network disabled).
4. **Never "novel".** The strongest statement is "not found in N papers retrieved on <date> for K queries".
5. **Novelty errors lean toward "covered"; evidence errors lean toward "inconclusive".** For overlap, `covers` and `partially covers` both count. For `agrees` / `contradicts`, only `covers` counts.

**Coverage rule for auto-extracted claims** (shown verbatim in the UI): covers = model_family ∈ {transformer, general} AND task ∈ {char_language_modeling, language_modeling, general} AND scale ∈ {tiny, small}; partially covers = same with scale ∈ {medium, large, unspecified}; otherwise does not cover. Curated claims keep their human flag.

**Trust tiers.** T1 curated (the 13 human-checked claims): can hard-reject. T2 auto-extracted, quote verbatim, published at a venue: soft-reject. T3 auto-extracted, quote verbatim, preprint or unknown review status: soft-reject, exactly like T2, because being unreviewed does not make a topic uncovered. T4 retrieved but no claim passed the quote check: a TF-IDF similarity above 0.35 between the candidate and the abstract holds the candidate for PI review. Soft-rejected candidates go to the end of the queue and are not run; the PI overrides with `pi_overrides` in the niche file, and dismisses a hold with `pi_dismissed_holds`.

**Literature status of a candidate** (`stats.literature_status`, used by both queues): covered by T1 → rejected; covered by T2/T3 → end of the queue; T4-flagged → held. A context-only claim never "covers" a candidate. The gate then asks the Literature Agent only "same comparison?" over claims of all tiers, and `stats.gate_outcome` applies the tier: only a curated claim can hard-reject.

**Rate limits and failure handling.** One arXiv request is in flight at a time (`arxiv_min_interval_s` 5.0). On HTTP 429/503 the client honors `Retry-After`, else backs off 15 s → 45 s → 120 s, up to `arxiv_max_retries` (3). Every successful response is cached under `corpus/raw/` and keyed by URL; **failures are never cached**, so re-running `make search`/`make record` resumes and only re-fetches what failed. If more than `max_failed_fraction` (10%) of the requests that were actually attempted still fail, `search` exits non-zero and **freezes nothing** (`make record` then aborts; the cache is kept for the resume). A title the scout "remembered" whose lookup errored is recorded as `request_failed`, not `not_found`, and the scout panel shows the three counts separately. When an arXiv scout query keeps failing, the same terms are tried on OpenAlex (`openalex_fallback`, higher limits; abstracts reconstructed from the inverted index, tagged `source: openalex`, same quote check). A fully degraded search (and fewer than `min_papers` retrieved) still falls back to the curated snapshot only when it is below the abort threshold. Without `--niche` / `--corpus` a session never touches the network. Operationally: wait ~15–30 min after an arXiv-heavy run before recording, or the limiter will still be tripped.

**Validation, shown prominently.** The search runs extraction on the curated papers and reports agreement with the human labels on the dashboard (config mapping, direction, covered-or-not). In `explore1` that agreement was low — config mapping 6/9, direction 4/9, setting 3/9 — so the dashboard now prints it as a banner directly above the gap scores and beside every soft-reject: a single auto-extracted claim is noisy, and the measured result still decides. `prompts/extract.md` was tightened with counter-examples from those rows (a theory statement → `context`; "comparable performance" → `no_worse`; no stated scale → `unspecified`); the re-run on a keyed machine will show the before/after. `lit-dryrun --corpus` repeats the gate and flags any fixture whose **action** changes when the corpus grows (a label-only change is reported, not aborted).

**Deterministic gate for generated candidates (FIXES3 P0-2).** A generated candidate's verdict is decided only by listed claims whose `config_change` is part of its delta (`stats.gate_decision` over the filtered set; for a combination this is what lets the tighten pass narrow to the untested field). So c01 (Pre-LN without warm-up) can no longer reject a `norm_position=post` candidate by being the LLM's "nearest" pick. Fixtures keep their human-curated citations.

**Fixture B is dropped in explore sessions (FIXES3 P0-3).** The gap engine already generates `norm=rmsnorm` from claim c04, and Fixture B's verdict depended on corpus size (`method_known_setting_untested` on the curated snapshot, `not_found_in_corpus` at 160 papers; the action, run, never changed). Explore sessions keep only Fixture A, the guaranteed prior-art demo.

What differs from the plan, plainly: the T4 check compares the candidate's own text (the change plus its supporting claims) rather than the full templated hypothesis sentence, because the template's boilerplate matched every on-niche abstract (threshold 0.35 on the curated set: 0 false holds, 6 of 8 own papers found). OpenAlex does not link most arXiv records to their conference version, so it can only add evidence of publication; everything else is "review status unknown" and treated as T3. One-hop seed expansion is not built. A niche outside the testbed still gets a map, and the queue says "no runnable candidates in this testbed".

## Hypothesis engine: scout → one graph → gaps → hypotheses

Recorded in `explore2` (and earlier in `explore1`, without a working scout). With a live search (`--niche`), the lab no longer walks a fixed menu of changes. It maps research directions, finds where the retrieved literature is silent or disagrees for our setting, and tests those gaps. Each hypothesis is labeled **transfer**, **combination** or **resolution** and is "not found in N retrieved papers", never "novel". New mechanisms that need new code are shown on the map but not run. **Graph scores are structural hints, not probabilities**: with ~100 papers they are weak signals, and the prior-art gate and the measured result still decide.

1. **Scout** (one LLM call, [prompts/scout.md](prompts/scout.md)): 8–15 directions. They are searches, never evidence. Paper titles the scout "remembers" are searched by title on arXiv: found → the paper enters with its real id; not found → logged and dropped. Code adds one direction per building block the scout did not mention. Directions tied to a building block are searched harder (`explore.per_block_direction_cap`, 30 papers instead of 15), because only claims about runnable methods can form combination gaps.
2. **One graph from both sources.** The niche search and the per-direction searches feed one frozen corpus; every paper records which direction found it.
3. **Mechanisms from a fixed vocabulary.** Extraction picks the reason an abstract gives from a list of 12 mechanisms (gradient stability, optimization speed, loss-landscape smoothness, implicit regularization, position generalization, long-range dependency, expressivity, representation quality, attention behavior, scale invariance, computational efficiency, parameter efficiency) and quotes the abstract's own words for it. Code keeps the category only if the quote is verbatim. A closed list is what makes two methods land on the same node; free-text reasons rarely match. Right after a search, `make search` prints how many mechanism nodes are linked to two or more runnable methods and warns below 3 (`explore.min_shared_mechanisms`).
4. **Gaps by formula** ([noesis_lab/gaps.py](noesis_lab/gaps.py), pure functions, toy-graph tests with known answers): coverage gap (missing cell of the tier-weighted tensor), Swanson ABC closure through a mechanism (sign-aware: a method→mechanism edge counts only if its claim reports a benefit, so a method that is "worse via k" never inherits k's benefit), link prediction for combinations (common neighbors, Adamic–Adar, resource allocation), contradiction (opposite signs in one setting bucket), structural hole (greedy-modularity communities).
   Each gap stores the direction its path predicts (`+`, `−`, `0`, or none for contradictions; combinations predict from their components' signs under prediction v2), shown on the gap card. A gap family with no gaps is shown as "no structural support found" with the reason, which is a true statement about that corpus, not an empty panel.
5. **One score:** `gap_score = plausibility × (1 − coverage_in_our_setting) × testability × evidence_quality`, components reported. The queue is ordered by it; ties go resolution > combination > transfer, then alphabetically. Already-tested changes are removed before the top 8 are taken. Literature status (rejected / soft-rejected / held) applies on top, with one difference from the older queue: auto-extracted claims that disagree in sign within the covered setting do not count as covering. That change is "open (contested)" and its resolution hypothesis runs; at the gate, a claim that is one side of the disagreement is not treated as prior art either. A curated claim still covers.
6. **Experiment language:** one allowlisted change, or any two that touch different fields. No LLM-written model code.
7. **Hypotheses from the top gaps** ([prompts/scientist_gap.md](prompts/scientist_gap.md)): the delta, the gap type, the score and the novelty label are code's. The Scientist writes the text and names the claims it rests on (`grounded_in`, validated against the gap's path).
8. **Tighten pass:** a targeted search for exactly that change (for a combination, queries name both methods), extraction and the quote check, then the gate. An overlap gets **one** revision that must be a strict subset of the delta; a second overlap is rejected with the overlapping quote. Results are frozen under `corpus/tighten/`, so replay needs no network.

The dashboard shows the ranked gap list with component bars, a gap card (the explanation path with verbatim quotes on the edges), an overview map (methods, mechanisms, communities, signed claim edges, our results, gaps), a direction map colored by gap status, a per-direction view, and the scout panel. `make rederive` recomputes every gap score and the queue from the frozen claims. The engine is active only when the frozen corpus contains scout directions, so `golden` replays unchanged. Settings are under `explore:` in [config.yaml](config.yaml).

What differs from the specs, plainly: mechanisms are a closed list of 12 rather than normalized free text; "contested" treats `covers` and `partially covers` as one bucket, because they are one bucket for the overlap question; a combination hint needs a shared mechanism or a co-tested method (a shared setting bucket alone only adds to the score); a contested cell does not count as coverage; a claim from a niche-search or curated paper is linked to a direction by its building block alone, because it has no direction provenance; a combination result is stored with no agrees/contradicts relation, since no claim is about the combination. 

### The testbed's building blocks

One change, or any two that touch different fields. Original: `norm=rmsnorm`, `norm_position=post`, `activation=swiglu|relu2`, `pos_encoding=rope`, `schedule=constant`, `optimizer=sgd_momentum`, `dropout=0.1`. Added so that more of the retrieved literature maps onto something runnable (in `explore1` only 23 of 128 claims did, and no two runnable methods shared a mechanism): `qk_norm=true`, `weight_tying=true`, `z_loss_coef=0.0001`, `label_smoothing=0.1`, `warmup_frac=0|0.02|0.1`, `grad_clip=0`, `init_scale=0.5|2.0`, `optimizer=lion`. Each is implemented in the testbed, tested on CPU (shape, finite loss, gradients, determinism) and has its own deterministic search direction and query. Label smoothing and z-loss change only the optimized loss; the reported validation loss is always plain cross-entropy. Lion runs at the baseline learning rate, without a retune. Whether this lifts the count of shared mechanism nodes to 3 or more is unknown until the next search; the dashboard reports the number whatever it is.

### Explore cheaply, confirm rigorously, promote

Recorded in `explore2`. For sessions whose corpus has scout directions (`search_loop:` in [config.yaml](config.yaml)):

1. **Explore.** The top 6 open gap hypotheses each run on **one seed**, against the incumbent's run on the same seed. These are labeled **EXPLORATION — NOT A RESULT** everywhere and get no label, branch or relation.
2. **Finalists (code).** The best per mechanism cell, then the top 2 by single-seed improvement among those that improved. If none improved, the best one is confirmed anyway (`min_finalists: 1`), so the session still measures how a single-seed number holds up.
3. **Confirm.** Finalists get the paired 3-seed screening, rule v2, and extra seeds if promising.
4. **Shrinkage.** Per finalist: single-seed estimate − paired estimate, in noise SDs. This is the single-run-loop comparison measured inside one session.
5. **Promotion.** After every finalist of a cycle is confirmed against the same incumbent, the best one that is still promising on all 5 seeds becomes the incumbent. Its own 5 runs become the noise floor (no extra compute). Later candidates are deltas on it. The original baseline and its noise floor never change, and a candidate confirmed after a promotion also gets a headline analysis against the original baseline.
6. **Write back.** Our own single-change results are added to the graph as derived edges — tier `D-explore` for a single-seed exploration (weight 0.3), `D-confirmed` for a paired screening (weight 1.0) — so a measured method becomes less of a gap and the next cycle looks elsewhere. A derived result **never counts as prior art** (`counts_as_covered` excludes it), is **never used as a literature search**, and is never shown to the gate: the gate and the tighten pass resolve claim ids against the frozen literature only, while gap paths and labels resolve against literature plus our own results. A single exploration only lowers a gap's score; it cannot close the gap on its own (only a literature claim or a `D-confirmed` result does). Gaps are recomputed and the next cycle starts while the run budget lasts.

**Crashes seal a partial bundle.** An unexpected exception during a live session is caught, recorded as a `crashed` decision (with the traceback), and the bundle is sealed with status `crashed: …` rather than discarded — the completed training runs are already in `runs.jsonl` and the notebook, manifest and state are written so the bundle still verifies. (A crash during replay re-raises instead, since there are no runs to lose.)

`make rederive` recomputes every single-seed estimate, finalist pick, shrinkage value, promotion check and per-cycle gap list.

### How good were the gaps?

Every gap predicts a direction from the signs of the claims on its path. After screening, code compares it with the measured branch: **hit** (same sign), **miss** (opposite sign), **null** (no improvement). The dashboard shows counts by gap type and by novelty label; counts, not percentages, because the numbers are small. `explore2`: 1 hit, 1 miss, 1 null, 1 no prediction (see the claims table). `explore1`'s dropout gap predicted "+" and measured harmful, −5.3 SD with every seed worse: a miss, reported.

**Combination predictions (prediction v2, not yet recorded).** Up to `explore2`, a combination gap predicted nothing, so it could only score "no prediction". Under `protocol.prediction_version: 2` it predicts the sum of its components' literature signs: each component's sign is the tier-weighted majority sign of its directional literature claims (our own derived results excluded); both "+" → "+", "+" and "0" → "+", both "−" → "−", mixed "+"/"−" or a component with no directional claim → "?" (no prediction). The rule is stored in the bundle and shown on the gap card. Recorded bundles are prediction v1 and replay unchanged.

Deviations from the fix list: finalists are confirmed before any promotion in a cycle (so their shrinkage is against the incumbent they were explored on), and hypotheses that were only explored may be explored again on a new incumbent. `compare-ideation` is not built.

On `golden`'s corpus the formulas return coverage, combination and contradiction gaps but no ABC or bridge gaps, because that extraction predates the mechanism field (0 mechanism nodes). Those types need a new search. On that corpus the gap queue now starts RoPE (transfer), SwiGLU (resolution, open because its preprints disagree), RMSNorm, SGD, constant schedule.

**Not built:** the stagnation redirect (re-running the scout on the least-explored communities), `make compare-ideation` and `make validate-gaps`.

References: literature-based discovery (Swanson's ABC model), arXiv:2506.12385; hypothesis generation over scientific knowledge graphs, ScienceDirect S0950705125003272; HypoChainer, PubMed 41336152; knowledge graphs in AI for science (survey), PMC13154823. Taken from GRAPH_GAPS.md; re-verify each before publishing.

## The curated claims (tier T1; read this before judging the verdicts)

**10 arXiv papers / 13 human-checked claims** covering the dimensions the testbed exposes (norm, Pre/Post-LN, optimizer, schedule, activation, positional encoding, dropout). They are in every corpus, they are the only claims that can hard-reject an idea, and a session without a live search uses them alone.

- Verdicts are scoped to the corpus the session froze. The UI says "not found in curated snapshot (10 papers)" or "not found in N papers retrieved on <date> for K queries", and never "novel".
- Shipped text is arXiv abstracts only (clearest licensing). Claim `source_span`s are verbatim substrings, enforced by code. Claim `setting` fields are paraphrased from the abstract alone, not the full paper.
- Fixtures are predefined hypotheses, labeled **FIXTURE** in the UI. They are not measurements.
  - **A (known):** Pre-LN trains without warm-up (Xiong et al., 2020).
  - **B (setting untested):** RMSNorm vs LayerNorm on a tiny char-level LM under a 2,000-step budget. We searched for a paper covering this exact setting and found none, but broader literature reports RMSNorm ≈ LayerNorm, so "no improvement" is the likely outcome. That is a fine result: it fires the "next candidate" branch.
  - Everything after A and B is **GENERATED** from the queue (not a fixture, not scripted).

### Setting coverage of curated claims is a human flag, not an LLM judgment

Every curated claim has a human-written `covers_our_setting` and a one-line `coverage_note`, applied with one criterion: **covers = the abstract states the result for Transformer language models, or for Transformers in general without restricting the task.** Architecture, task and scale are not weighed case by case. Applied to every claim alike:

- c01 (Pre-LN without warm-up, "on a wide range of applications") covers us; c04 (RMSNorm, "diverse network architectures") does not.
- c09 (GLU variants) does not: the abstract restricts the result to the sequence-to-sequence Transformer. c12 (RoPE) does not: the reported evidence is long text classification.
- c10 (squared ReLU, Primer) does cover us under this criterion (Transformer auto-regressive language modeling) even though its scale is far larger; the criterion deliberately does not use scale. If a squared-ReLU candidate matches that claim as the same comparison, it is gated as known. Change the criterion, not the verdict, if that is not what you want.

Verdict = `stats.prior_art_verdict(same_comparison, coverage)`: same comparison and covered → `known_in_corpus`; same comparison and not covering → `method_known_setting_untested`; otherwise `not_found_in_corpus`. For auto-extracted claims "covered" means the coverage rule says covers or partially covers. The dashboard shows the flag next to the passage, labeled human-curated or computed.

### Evidence relations do not overclaim

`agrees` / `contradicts` are used only when the claim covers our setting. Otherwise a measurement is stored as `consistent_in_our_setting`, `not_reproduced_in_our_setting` or `inconclusive`. A result for a two-field combination gets no relation at all, because no claim is about the combination. For claims with a speed component (c04, c05), if measured throughput moves the other way by more than 5% the note gets `partial: speed claim not reproduced (implementation-dependent)`.

## Related work (positioning)

| System | What it does | Gap we exploit |
|---|---|---|
| Karpathy autoresearch and forks | Agent edits training code, trains a fixed 5 min, keeps the change if one run beats the last | One run per decision (keeps noise), no literature. Closest prior art to our testbed; we cite it rather than pretend the loop is new |
| AI Scientist v1/v2 | Idea → code → experiments → paper | Independent evaluation found known ideas marked novel and wrong or hallucinated numbers |
| Google Co-Scientist | Multi-agent hypothesis generation | Never runs experiments |
| Kosmos / Edison | Literature + data-analysis agents | Closed, commercial |
| Agent Laboratory | Literature → experiments → report | LLM reviewers scored it above human reviewers |

Citations and URLs are in [BUILD_PLAN.md](BUILD_PLAN.md) §0 and were taken from there; re-verify each before publishing.

## Reproducibility notes

- **Pinned:** Python 3.12 (`.python-version`), `uv.lock`, vendored dataset with SHA256 check, committed curated claims and fixtures.
- **The literature is frozen per session** in `results/<session>/corpus/`: niche, query plan, scout directions, raw API responses, papers with ranking components, claims with tier and coverage, drops with reasons, each hypothesis's tighten search, and the search's own LLM recordings. The manifest hashes all of it.
- **Every LLM call is recorded** (full request, response, usage) in `results/<session>/recordings/llm.jsonl`, keyed by a hash of model, effort, role, prompts and schema. Effort is per role (`llm.effort_by_role`: low for the gate, `critic_post` and extraction; medium otherwise); bundles recorded before that block replay with their single effort. Replay serves from it and errors on a miss, never silently going live. An edit to a loop prompt (Literature, Scientist, Critic) after recording therefore makes replay fail loudly; re-record. Search-side prompts (query plan, scout, extraction) are not replayed: replay reads their frozen output.
- **Model:** `llm.model` in [config.yaml](config.yaml) is `claude-opus-5-5`. Claude 5 model IDs have no date suffix, so the pin is the ID, and each recording stores the served model. The plan asked for temperature 0; this model family rejects `temperature`/`top_p`, so determinism comes from the recordings. The refusal fallback (`llm.server_side_fallback`, a beta) is on by default; set it to `false` to disable.
- **Each run stores** git commit and dirty flag, config hash, seed, device, torch/OS versions, `uv.lock` hash, run order and timestamp. **Every event and decision stores `ts`** (UTC), which the dashboard's timeline uses to show where wall time went; like the run timestamp it is excluded from the state digest, so replay is still exact (bundles recorded before it have none).
- **Profiles:** `full` (the headline profile, MPS), `smoke` (CPU, tiny, bit-deterministic; used by tests and `make smoke`) and `rehearse` (500 steps, one short cycle; `make rehearse`). Smoke, rehearse and mock-LLM output is marked in the UI and must never be presented as a result.

## Speed: where the time goes

Measured on `explore2`: training was ~17 minutes (28 runs × ~36 s); LLM calls and tighten searches between runs ~9 minutes; the search ~25–40 minutes (arXiv at one request per 5 s, extraction calls one at a time). After the last run started, ~57 minutes passed with nothing written before the bundle was sealed: most likely the Mac slept (that session was started without `caffeinate`) or an API call stalled (there was no explicit timeout). The real compute is ~17 minutes; the rest was waiting. What changed (FINAL_FIXES B), none of which can change a decision:

- **Never stall.** Live targets run under `caffeinate -i` on macOS. The Anthropic client has `timeout_s: 120`, `max_retries: 2`, so a stalled call fails in ~6 minutes at worst and the session seals a partial bundle (`crashed: …`) instead of hanging. The dashboard timeline flags any silence over 5 minutes.
- **Parallel LLM calls** (`llm.concurrency: 5`). Extraction batches and the `lit-dryrun` rows/repeats run in a thread pool; results are applied in a fixed order. Each loop cycle prefetches every candidate's Scientist proposal and pre-run Critic review in parallel before any training. A prefetched response is used only when the sequential path makes the byte-identical request, and it is recorded at that point, so the recordings, events and decisions are exactly what a sequential run writes (tested: same digest, same recordings file). An unused prefetch is dropped; its cost still counts towards the LLM budget.
- **Cheaper routine roles.** `effort: low` for the gate, `critic_post` and extraction; the pre-run Critic is capped at 5 confounds and 4 controls, one line each.
- **Faster tighten searches.** OpenAlex first (polite pool, ~10 requests/s with `OPENALEX_MAILTO`), arXiv only if that request fails; same quote check. Responses are also kept in a URL-keyed cache under `work/cache/` that later sessions reuse; the bundle freezes its own copy as before.
- **Not done: two training runs at once on MPS** (FINAL_FIXES B5). It needs measuring first (`make timing` with 1 vs 2 processes, adopt only at ≥ 1.5× total throughput), which needs the Mac.

## Known limitations

- **Screening only.** Three paired seeds, five after extra seeds; no confirmation, no p-values, no multiple-comparison control.
- **Prediction v2 and the speed work (FINAL_FIXES B1–B4) have not been recorded live.** They have run only against a scripted LLM and recorded API responses. OpenAlex's boolean `search` syntax for tighten queries is used as documented but was not exercised against the live API from our build machine; a failed OpenAlex request falls back to arXiv.
- **Rule v1 was asymmetric**, and `golden` and `explore1` were judged by it. `explore1`'s "RMSNorm harmful" rests on one seed; under v2 it would be "no improvement, seeds disagree". We leave the recorded label and say so here.
- **The noise floor is a rough estimate** from 5 seeds (see the interval above). A candidate near 1 SD can change branch with a different baseline draw.
- **Graph scores are weak signals**, and the only scored prediction so far is a miss. In `explore1` no two runnable methods shared a mechanism, so no combination gap could form.
- **Auto-extraction is imperfect** (see the agreement numbers above). That is why auto-extracted claims cannot hard-reject and why the quote is always shown.
- **Peer-review status is mostly unknown.** OpenAlex does not link most arXiv records to their conference version, so nearly all auto-extracted claims are treated as preprints.
- **The prior-art judgment is still an LLM call** ("which claims test the same comparison?"). Gate v2 makes the verdict insensitive to which claim it names first, not to whether it names the right set. The passage is shown so a human can check it.
- **New options are not fully tuned.** Lion runs at AdamW's learning rate ÷ 5 with weight decay × 5 (the paper's rough guidance), held constant across every Lion run; the ratio is not swept. A fixed QK-norm scale and one z-loss coefficient are likewise untuned. A negative result for these is a result for this exact setting only.
- **Lion was never tested fairly.** Every Lion exploration in `explore2` scored ~1.64–1.70 val loss vs ~1.53 baseline (single seed each, never confirmed). The likely cause is our lr ÷ 5 scaling under a 2,000-step cosine budget, not Lion itself. Every Lion card says "exploration only; our hyperparameter scaling, not a test of Lion".
- **Cumulative numbers include the incumbent.** After a promotion, the vs-original-baseline number of a later change includes the incumbent's gain (ReLU² + RMSNorm: +1.46× SD vs original, but −0.67× SD vs the RoPE incumbent). The decision and every headline use the vs-incumbent number; the cumulative one is shown, labeled, for reference only.
- **`explore2` was recorded on a thin mechanism graph** (2 mechanism nodes shared by runnable methods, want ≥ 3), so ABC, combination and bridge gaps were sparse. The dashboard states it as "Recorded with this limitation".
- **MPS is not bit-deterministic.** Re-executing on another Mac gives different numbers; R1/R2 verify the shipped numbers, not that training reproduces them to the bit. The CPU profile is bit-deterministic (tested).
- **Step budget, not wall-clock** (see "Why steps, not seconds"). Throughput still differs by machine and implementation; it is reported as a secondary metric and does not enter any decision.
- **One fixed testbed and one model scale.** Results do not transfer by themselves.
- **The tighten pass needs the network during recording.** If arXiv is unreachable the hypothesis is gated on the frozen corpus alone and the degradation is recorded.

## Repository layout

```
config.yaml  niche.yaml  pyproject.toml  uv.lock  Makefile  app.py
noesis_lab/
  schemas.py llm.py mock_llm.py config.py cli.py            contracts, record/replay LLM, CLI
  orchestrator.py runner.py stats.py                        session, runs, statistics and the rule
  literature.py agents.py                                   Literature Agent, Scientist, gap Scientist, Critic
  gaps.py                                                   graph, gap formulas, gap score, gap queue
  search/  plan.py scout.py arxiv.py openalex.py http.py    query plan, scout, API clients, raw cache
           rank.py extract.py coverage.py corpus.py         dedupe/rank, extraction checks, coverage + tiers, freeze
           tighten.py grid.py textsim.py                    per-hypothesis search, coverage grid, TF-IDF / PCA
  store.py bundle.py rederive.py                            SQLite evidence store, manifest, re-derivation
  evidence_chain.py gap_views.py dashboard.py               dashboard graphs (DOT), wording and timeline helpers
  testbed/{harness,model}.py                                tiny GPT and its harness
prompts/   literature.md (gate v1) literature_gate.md (gate v2) scientist.md scientist_gap.md critic_pre.md critic_post.md
           query_plan.md scout.md extract.md
data/      tinyshakespeare.txt papers.json claims.json fixtures.json
scripts/   fetch_snapshot.py verify_snapshot.py
tests/     pytest, CPU, mocked LLM, recorded API responses under tests/data/
results/<session>/   notebook.sqlite runs.jsonl recordings/llm.jsonl env.json state.json
                     inputs.json config.snapshot.yaml MANIFEST.json
                     corpus/{niche.yaml,query_plan.json,scout.json,papers.json,claims.json,
                             dropped.json,meta.json,llm.jsonl,raw/,tighten/}
```

## Not built

Stagnation redirect, `compare-ideation`, `validate-gaps`, `cosine_final_frac` and other parameterized variants, one-hop seed expansion, confirmation tests, a retrieval benchmark, a sealed audit split, re-execute mode, CI, hash-chain provenance, LLM-written code patches, a second LLM provider.
