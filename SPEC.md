# SPEC — "Prior Art": an AI research lab that knows what's already been done

> **Today's hackathon scope is `BUILD_PLAN.md`; it wins any conflict.** This file is the full design, for the README's "where this goes next" and post-hackathon work.

## Context (read first)
We are competing in a hackathon. Challenge: *Build an AI lab for the next breakthrough and make scientific discovery 10x faster. Use multiple AI agents to research, form hypotheses, plan and run experiments, learn from the results, and decide what experiment should happen next.*

Our angle: AI-scientist systems waste budget rediscovering known results. We build a **literature knowledge graph (KG)** that agents query to (1) find under-tested method × setting combinations, (2) flag hypotheses whose effect is already reported in the indexed corpus, and (3) seed hypotheses from neighboring work. The KG is the differentiator, but **the loop must close**: hypotheses become real experiments, results are measured, and they update what we try next.

Domain: CS / AI, narrowed to **one fixed testbed** (small char-level LM training) with a **typed, allowlisted experiment language**.

Python 3.11. Must run on a MacBook (Apple Silicon, torch `mps`, CPU fallback).

## Terminology
- **Run**: one training execution of one config with one seed. Has a `run_id`.
- **Experiment**: the evaluation of one hypothesis = its candidate runs + the paired incumbent runs it is compared against. Has an `experiment_id`.
- **Analysis**: the statistics computed over an experiment. Has an `analysis_id`.
- **Event**: any agent or orchestrator action (LLM call, novelty check, ranking decision). Has an `event_id`.
- **Original baseline**: immutable reference config, fixed at start. Used for all headline reporting.
- **Incumbent**: mutable best-so-far config. New hypotheses are deltas against it.

## Non-negotiable rules
1. **LLMs may choose configuration values and predictions, and may quote code-supplied values, but may never invent empirical measurements.** Every displayed measurement links to a `run_id` or `analysis_id`. Every other displayed number (novelty score, cost, ranking score) links to an `event_id`.
2. **Agents communicate via JSON validated by Pydantic.** Invalid output → one retry with the validation error → fail loudly.
3. **Every prior-work claim cites a paper ID in the corpus plus the exact source span** (sentence offsets in the abstract) it rests on.
4. **Hard budgets** in `config.yaml`: wall-clock, runs, LLM spend, iterations.
5. **Failures are data**: crashes and timeouts are recorded with stderr and returned to the planner.
6. **SQLite is the single source of truth.** Graph views, embeddings index and dashboard are derived from it and rebuildable.

## Positioning
We are "autoresearch with a literature memory and statistics." Autoresearch-style loops (edit → train for a fixed time → keep if better) are fast but keep noise, overfit a frozen metric, and ignore prior work. Hypothesis systems like Co-Scientist reason over literature but never run experiments. Our edge is **trustworthy discovery**:
1. We count only improvements that **survive re-testing on fresh seeds**. "Faster" means time to a *confirmed* improvement.
2. **Literature sets the priors** that decide what we test, and an ablation proves it helps.
3. **"Known method, untested in this setting"** is a legitimate finding, and **contradicting a paper** is a headline event.
4. Our confirmed results are **written back into the knowledge graph** next to published claims, so the next session starts smarter.
5. Judges can **verify every claim** at three reproduction levels.

## Workflow: roles and actions
Governing principle: **LLM roles propose, interpret and argue. Deterministic roles measure, decide and record.** State comes in two kinds, and the rule differs:
- **Evidence state**: measurements, likelihood updates, statistical labels (`screened_promising`, `confirmed`, `refuted`, `inconclusive`), the incumbent, survival results. **Only code computing on runs may change it. No LLM output, ever.**
- **Search state**: which hypotheses exist, which are filtered out, priors, queue order. LLMs may influence it, but only through the declared channels below, each logged and each covered by an ablation.

Invariant: **an LLM mistake can waste compute or skip a good idea, but it cannot produce a confirmed finding.** Confirmation uses fresh seeds and fixed rules, so no prior or filter can manufacture one.

Declared LLM → search-state channels (exhaustive):
1. Librarian claim extraction → literature priors (Phase 1). Priors are bounded and shrunk; data dominates after screening.
2. Librarian prior-art verdict → `rejected_known_in_corpus` (Phase 3). The citing span is shown, and the PI can override.
3. Matchmaker deduplication → merge (Phase 3). Both originals are kept and linked.
4. Skeptic verdict → `rejected_critic` / revise (Phase 3). Reasons are kept, and the PI can override.
5. Interpreter surprise flag → routing to the Librarian and Skeptic (Phase 7). It changes what gets looked at, not any status.

Nothing else LLM-generated enters the Decider. In particular, the Theorist's predicted effect size is the hypothesis's **falsification target**, not a prior, and there is no LLM ranking score.

### Cast
| Role | Kind | Owns | Must never |
|---|---|---|---|
| **Principal Investigator (PI)** | Human | Research question, budget, optional approval/redirect at checkpoints | Hand-edit results or recordings |
| **Lab Manager** | Deterministic | Phase transitions, budgets, stop conditions, choosing which role acts next | Make scientific judgments |
| **Bench Keeper** | Deterministic | Testbed, data splits, the sealed audit split, timer, metrics, seeds, environment fingerprint | Accept changes from any agent |
| **Librarian** | LLM + retrieval | Corpus, extracted claims with source spans, the knowledge graph, field brief, prior-art verdicts | State a claim without a paper ID and span |
| **Theorist** | LLM | Hypothesis proposals | See ranking scores, or assign novelty/impact numbers to its own ideas |
| **Skeptic** | LLM, **different model family** from the Theorist | Attacks: confounds, falsifiability, testability, "is this just a retune" | Approve its own suggested alternatives |
| **Matchmaker** | LLM judge | Deduplication: merging near-identical hypotheses (search state, channel 3) | Rank, score or prioritize hypotheses |
| **Planner** | LLM, constrained | Translating a hypothesis into an allowed config change + controls + seed plan | Write code in the default path |
| **Decider** | Deterministic | Beliefs, priors from the literature, choosing the next action, promoting the incumbent | Read LLM prose as evidence |
| **Runner** | Deterministic | Sandboxed execution, failure capture | Retry silently |
| **Statistician** | Deterministic | Paired deltas, CIs, screening and confirmation rules, status labels | Use seeds reserved for the Auditor |
| **Interpreter** | LLM | Plain-language reading of a finished statistics object; surprise flag vs. literature | Introduce a number not present in the statistics object |
| **Archivist** | Deterministic | Writing derived claims back into the graph, lessons, the limitations ledger, provenance | Delete failed claims (they move to limitations) |
| **Reporter** | LLM + deterministic templates | Findings ledger, claims table, session narrative | Quote any number without its ID |
| **Auditor** | Deterministic, runs after the session | Survival re-test on fresh seeds + sealed split; benchmark versus competitors | Share seeds or split with any other role |

### Phase 0: Calibrate the bench (once per machine)
1. **Bench Keeper** freezes the original baseline, data splits and a **sealed audit split** (never seen by the loop), and records the environment fingerprint.
2. **Bench Keeper** runs the baseline over ≥10 seeds, plus same-seed repeats, to measure seed-to-seed noise and run-to-run nondeterminism.
3. **Lab Manager** sets, from those measurements: run budget, minimum practical improvement (δ_min), screening and confirmation seed counts, drift-check cadence. It also splits seeds into three disjoint pools: **decision seeds**, **confirmation seeds** and **audit seeds**.
4. **PI** sets the research question (e.g. "what improves small-LM training efficiency in a short budget?") and the session budget.

Gate: if noise ≥ typical single-change effect, the Lab Manager reports it and the PI chooses: a longer budget, a bigger model, or leading the pitch with literature priors (Positioning #2–3) rather than survival rates.

### Phase 1: Map the field
1. **Librarian** plans the search: 3–5 research dimensions, each with precise, broad and cross-domain queries, plus key authors and venues.
2. **Librarian** discovers papers by searching, then 2-hop citation expansion (forward and backward) from the most relevant seeds. Duplicates are removed by identifier, not title.
3. **Librarian** curates the corpus to the testbed's scope and records coverage (papers per method, years, sources).
4. **Librarian** extracts for each paper: methods, settings (scale, modality, data, budget) and claims (method → effect → metric → setting), each with its source sentence.
5. **Librarian** builds the graph and produces the **field brief**:
   - **Consensus**: methods with consistent positive claims.
   - **Contradictions**: claims that disagree.
   - **Untested cells**: method × setting combinations with no claim, especially "works at large scale, unknown at our scale."
6. **Decider** converts the claims into **literature priors**: for each allowed method, a prior expected effect and uncertainty. Many consistent claims in similar settings → confident prior. Few, contradictory, or distant-setting claims → wide prior.
7. **Librarian** reports retrieval quality on the labeled evaluation set (recall, verdict accuracy) so the corpus's blind spots are visible.

### Phase 2: Propose
1. **Theorist** receives the field brief, current beliefs, the incumbent, past results, lessons and any PI redirect. It proposes a batch of hypotheses, drawn from four sources:
   - **Transfer**: an untested cell (known elsewhere, untested here)
   - **Contradiction**: test which side holds in our setting
   - **Follow-up**: build on our own confirmed or surprising results
   - **Combination**: two individually confirmed changes together
2. Each hypothesis states: the change; the mechanism; a directional prediction with expected size; what result would refute it; **grounded evidence** (paper claims with spans that motivate it); and which source above it came from.

### Phase 3: Filter
1. **Librarian, prior-art check**: one verdict per hypothesis:
   - `known_in_corpus` → rejected, and shown with the citing sentence
   - `method_known_setting_untested` → kept and labeled
   - `not_found_in_corpus` → kept and labeled with the corpus size
2. **Matchmaker, deduplication**: merges near-identical hypotheses, including ones already tested this session.
3. **Skeptic, adversarial review**, checking:
   - **Confounds**: e.g. faster steps ≠ better learning; more tokens seen.
   - **Testability within the budget.**
   - **Falsifiability.**
   - **Whether it's a disguised retune.**
   - **Required controls.**

   Verdict: accept / revise once / reject. Rejections are kept with reasons.
4. **Planner** turns each accepted hypothesis into an allowed config change, plus controls and the screening seed plan. If the change can't be expressed in the allowed space, it goes back to the Theorist as "untestable here."

### Phase 4: Choose the next action
**Decider** scores every candidate action and picks one:
- **Screen** a new hypothesis. Value = expected improvement over the incumbent under its **literature prior** (or the default wide prior when the corpus has no matching claim), per second of compute. Ties are broken deterministically (older first).
- **Confirm** a promising screened hypothesis. High value because it can produce a real, citable result.
- **Re-measure the incumbent** (drift check) when due.
- **Refresh the literature** (send the Librarian back) when a surprise is pending or proposals are drying up.
- **Ask the Theorist** for a new batch when the queue is thin or stale.

The score breakdown is recorded for every choice. If the PI has turned on checkpoints, the choice waits for approve / skip / redirect.

### Phase 5: Run
1. **Runner** executes the candidate on its assigned **decision seeds** in the sandbox and reuses cached incumbent runs on the same seeds. Methods are interleaved with the drift checks.
2. A failure is captured whole and returned to the Planner once ("can this be expressed differently?"). If it fails again, it becomes a refuted-by-infeasibility record, not a silent drop.

### Phase 6: Analyze
1. **Statistician** computes paired deltas and confidence intervals. It labels the result with the fixed rules:
   - **Screening:** `screened_promising` if the improvement beats δ_min with a consistent sign; otherwise `refuted` or `inconclusive`.
   - **Confirmation (fresh confirmation seeds only):** paired test with correction across the session, plus δ_min → `confirmed` or not.

   It also checks the delta at equal tokens, to separate "learns better" from "runs faster."
2. **Interpreter** writes a short reading of the statistics object and compares the outcome to the hypothesis's grounded evidence. It raises a **surprise** if the result contradicts a cited claim, or if a "known-good" method hurts.

### Phase 7: Learn
1. **Decider** updates beliefs. Confirmed → **promote to incumbent** (the original baseline never changes). Cached incumbent runs are invalidated, and combination hypotheses involving the promoted change become eligible.
2. **Archivist** writes a **derived claim** into the graph for every confirmed or refuted result: method → effect → our setting, linked to its runs and analysis. These sit beside published claims, marked as ours.
3. **Archivist** records **lessons**: which hypothesis sources and kinds of change tend to pay off, used by the Theorist and as a mild adjustment to priors next session. Failed and refuted claims go to the **limitations ledger** (shrinking our claims to what the evidence supports).
4. On a **surprise**, the Lab Manager routes the result to the Librarian for a targeted search: is this contradiction already reported? Is our setting meaningfully different? Then the Skeptic checks it for artifacts before anyone calls it a finding. A surprise that survives both, and confirmation, becomes a **contradiction finding**.
5. The Lab Manager returns to Phase 4. It goes back to Phase 2 when the queue is thin, and to Phase 1 when the Decider requests a literature refresh.

### Phase 8: Stop and report
1. **Lab Manager** stops when the budget runs out, when no candidate's expected value exceeds its cost, or when the PI says so.
2. **Reporter** produces the **findings ledger**, in this order: confirmed improvements; contradiction findings; confirmed "known method, untested setting" results; refuted hypotheses (with what refuted them); prior-art rejections (with citations); limitations. Every number links to its run, analysis or event.
3. **Archivist** seals the results bundle for the three reproduction levels.

### Phase 9: Audit and benchmark (separate, after the session)
1. **Auditor** re-tests every improvement claimed by every method on **audit seeds no method ever saw**, and on the **sealed audit split**. This removes winner's-curse bias, because decision seeds were used to select winners and so are optimistic by construction. It reports each method's **survival rate**.
2. **Auditor** runs the benchmark under identical conditions: same allowed change space, baseline, data, harness, compute accounting and interleaved order. Competitors:
   - random search
   - **autoresearch-style agent** (free proposals, keep if one run beats the incumbent, no seeds or statistics)
   - single LLM agent without literature or critic
   - our system with the literature switched off (ablation)
   - our full system
3. Headline metrics:
   - **Time and compute to first audited improvement**, and the number of audited improvements in the budget.
   - **Survival rate** of claimed improvements.
   - **Prior-art rejection rate.** For no-literature methods it's computed after the fact, which shows how much of their effort went into rediscovery.
   - **Literature-on vs. literature-off ablation.**
   - Total time **with and without LLM latency.**
4. **Reporter** fills in the README claims table. Each claim records its supporting analysis, its reproduction level and its hardware.

### Cadence at a glance
- **Inner loop** (minutes): choose → run → analyze → learn. Repeats until the queue empties or something changes.
- **Middle loop** (each batch): propose → filter → plan.
- **Outer loop** (rare): literature refresh, triggered by surprises or by proposals drying up.
- **Session end**: report and seal. **Post-session**: audit and benchmark.

## Testbed

### Harness-owned (never modifiable by agents)
`testbed/harness.py` owns: dataset download + fixed train/val split (TinyShakespeare, char-level), tokenizer, fixed validation batches and eval token count, the timer, metric computation, seeding, and hardware/software metadata capture.
- Timer starts **after** model construction and an MPS prewarm step and **excludes eval**. Budget is training wall-clock seconds (default 60–90 s; set in step 2 from measured throughput).
- Also log `tokens_seen` so results can be compared at equal tokens as well as equal time.
- Each run records: device, torch version, macOS version, run order index, start timestamp, thermal-relevant info if cheaply available.
- Reproducibility promise: **best-effort**. Same seed → same init and data order. MPS kernels are not guaranteed bit-deterministic. Seed-to-seed noise is measured in step 2, not assumed.

### Experiment language (`testbed/space.py`)
A typed `ExperimentConfig` (Pydantic) with allowlisted fields and ranges, e.g.:
- optimizer: `adamw | lion | sgd_momentum | muon`; lr; betas; weight_decay; grad_clip
- schedule: `constant | cosine | linear_decay | wsd`; warmup_frac
- norm: `layernorm | rmsnorm`; norm_position: `pre | post`; qk_norm: bool
- activation: `gelu | relu2 | swiglu`
- pos_encoding: `learned | rope | none`
- width, depth, heads (bounded), init_scale, weight_tying: bool
- dropout, label_smoothing, z_loss_coef
- batch_size, seq_len (bounded)

Each categorical option carries a `kg_method_id` so hypotheses map onto KG Method nodes. A **hypothesis is a delta** (one or a few fields) on the incumbent, plus a prediction. The final field list lives in code; keep it small enough that every option is implemented and tested.

**Stretch only**: constrained code patches to a single designated module (`testbed/mutable/block.py`) behind a fixed interface, AST-checked import allowlist, harness untouched.

### Sandbox
Runs execute in a subprocess with: timeout, memory limit (`resource.setrlimit` where supported), no network (unset proxies, block sockets via a sitecustomize guard), read-only access to data, write access only to its run directory.

## Lab Notebook (SQLite via SQLModel, WAL mode so the dashboard can read during writes)
Tables: `papers`, `spans`, `kg_nodes`, `kg_edges`, `hypotheses`, `experiments`, `runs`, `analyses`, `beliefs`, `events`, `lessons`.
- `hypotheses`: id, statement, mechanism, config delta, prediction (direction + expected effect size), falsification criterion, status (`proposed | rejected_known_in_corpus | rejected_critic | queued | screening | screened_promising | confirming | promoted | refuted | inconclusive`), novelty event_id, parent id.
- `runs`: run_id, experiment_id, role (`candidate | incumbent | original_baseline`), config hash, seed, val loss, tokens_seen, curves JSON, metadata, status, stderr.

## Knowledge graph (`kg/`)
- **Corpus**: curated **100–300 papers** on small-LM training (optimizers, schedules, normalization, activations, positional encodings, init, regularization). Sources: arXiv / Semantic Scholar / OpenAlex, fetched once and **versioned in the repo** (`kg/corpus_v1.jsonl`). Record coverage stats (papers per method, year range, sources).
- **Extraction**: LLM extracts `Method`, `Setting` (model scale, modality, data), and `Claim` (method → effect on metric in setting), each with the source sentence span. Extraction outputs are cached and versioned too.
- **Index**: NumPy cosine similarity over sentence-transformer embeddings (`all-MiniLM-L6-v2`). No FAISS.
- **Graph views**: built from SQLite into networkx on demand.
- **API**:
  - `search(query, k)`
  - `novelty_check(hypothesis)` → `{verdict, nearest: [{paper_id, span, similarity}], corpus_coverage, rationale}` where verdict is one of:
    - `known_in_corpus`: the corpus reports this method's effect in a comparable setting
    - `method_known_setting_untested`: the method appears, but no claim matches our setting (small char-level LM, short budget)
    - `not_found_in_corpus`
    The UI must never say "novel". It says "not found in indexed corpus (N papers)".
  - `gaps()` → method × setting cells with few or conflicting claims
  - `neighbors(method_id)`
- **Retrieval eval** (`kg/eval_set.jsonl`): 20–30 hand-labeled hypotheses (e.g. "RMSNorm instead of LayerNorm", "cosine vs. constant LR") with their known source papers, plus some that should not match. Report recall@k and verdict accuracy. This set also provides the **deterministic demo fixture**: at least one item is a guaranteed `known_in_corpus` rejection. Label it as a fixture in the UI.

## Agents (`agents/`, common base `Agent.run(input: Model) -> Model`)
Implementation notes. The Workflow section defines roles, decision rights and order and wins any conflict. Mapping: Literature = Librarian; Hypothesis = Theorist; Critic = Skeptic (different model family); add Matchmaker (dedup only; no ranking); Analyst = Statistician + Interpreter; add Archivist (derived claims, lessons, limitations) and Auditor (Phase 9).
1. **Literature**: field summary, gaps, contradictions, all with citations + spans. Re-invoked with a targeted query when a result is surprising (e.g. a known-good method hurts).
2. **Hypothesis**: proposes config deltas with mechanism, directional prediction, expected effect size, and falsification criterion. Calls `novelty_check`. `known_in_corpus` → status `rejected_known_in_corpus` (shown on the dashboard with the citing span). `method_known_setting_untested` is allowed and labeled as such.
3. **Critic**: checks confounds (e.g. speed vs. learning: did it see more tokens?), testability within budget, falsifiability, and whether it's a mere retune. Output: accept / revise / reject + required controls.
4. **Planner**: produces a validated `ExperimentConfig` delta and seed plan. No LLM-written code in the default path.
5. **Executor** (code): schedules runs, enforces sandbox, records results. Config validation errors are returned to the planner.
6. **Analyst** (stats in code; the LLM only interprets an already-computed stats object):
   - Comparisons are **paired by seed**: per-seed delta = candidate − incumbent on the same seed.
   - **Screening** (default 3 seeds): report mean paired delta and a bootstrap CI. Mark `screened_promising` only if mean delta beats `δ_min` (practical threshold set from measured noise in step 2) and the sign agrees on all seeds. No p-value claims at n=3.
   - **Confirmation** (default 5 fresh seeds from the confirmation pool; screening seeds are not reused, to avoid selection bias): paired t-test on deltas, Holm-corrected across all confirmations in the session, plus `δ_min`. Pass → `promoted`.
   - Also report the delta at equal tokens when token counts differ by more than 5%.
7. **Decider**: updates beliefs and picks the next action.
   - Belief per hypothesis: Normal over effect size. The prior comes from the literature prior for the method/setting (Phase 1), shrunk toward 0. Without a matching claim, use the default prior: mean 0, variance set from the typical single-change effect size measured in Phase 0. The Theorist's predicted effect is never used as a prior. Conjugate update from paired deltas, with noise variance estimated in step 2.
   - Score = expected improvement over the incumbent ÷ expected compute seconds, with a bonus for resolving `screened_promising` items (confirmation is information-valuable). The LLM proposes; the scorer ranks. The score breakdown is logged as an event.
   - **Promotion** replaces the incumbent and invalidates cached incumbent runs. The original baseline never changes.

## Throughput and run accounting
- Incumbent runs are **cached per (incumbent config hash, seed)** and reused across hypotheses. Re-run one incumbent seed every K experiments to detect drift (thermal throttling, background load). Flag drift above the noise floor.
- At 60 s/run with cached incumbent: screening a hypothesis ≈ 3 runs ≈ 3 min. A 30-min session ≈ 7–9 screens, or fewer if a confirmation happens.
- Pipelining: agents can think about the next hypothesis while a run executes. Optional; only add it if the simple loop works.

## Orchestrator (`orchestrator.py`)
Plain Python loop:
```
lit = literature.survey()
while budget_left():
    hyps = critic.review(hypothesis.propose(lit, notebook))   # novelty_check inside propose
    queue.add(planner.plan(h) for h in accepted(hyps))
    item = decider.select(queue)          # screen new OR confirm promising
    runs = executor.run(item)
    analysis = analyst.analyze(runs)
    decider.update(analysis)              # beliefs, maybe promote
    if analysis.surprising: lit = literature.targeted(analysis)
```
`--pause-for-approval`: the dashboard shows the selected item with Approve / Skip / Redirect (note passed to the hypothesis agent).

## LLM layer (`llm.py`)
- Anthropic SDK wrapper, model in config, structured output via tool-use/JSON schema, cost accounting into `events`.
- Modes: `live`, `record` (live + save every request/response keyed by a hash of prompt+schema+model), `replay` (serve from recordings; error on a miss; used only by R2, see Reproducibility). Tests use a hand-written `mock`.
- Prompts live in `prompts/*.md`.

## Reproducibility (remote hackathon: judges cannot see our machines)
Assume judges have only the repo, a README, possibly no Mac, no GPU and no Anthropic key. They must be able to verify our claims at three levels, each one cheaper than the next.

### Three levels of reproduction
| Level | Command | Needs | Guarantee |
|---|---|---|---|
| **R1 Re-derive** | `make rederive` | CPU, seconds | Recompute every statistic, plot and table from the shipped results bundle. **Bit-exact**; CI checks it. |
| **R2 Replay** | `make replay` / hosted dashboard | CPU, no API key | Replay the full agent trajectory from recorded LLM responses **and** recorded run results. Dashboard state is **identical** to the original session. |
| **R3 Re-execute** | `make reexecute PROFILE=smoke\|full` | CPU (smoke) or MPS/CUDA (full) | Re-run the recorded config trajectory, or run fresh with a live LLM. Numbers will differ; claims must hold **within reported tolerance**. |

Important: LLM prompts embed run results, so replaying recorded LLM responses while experiments run live will miss the cache as soon as any number differs. Therefore:
- **R2** replays both the LLM responses and the run results. Nothing executes.
- **R3 trajectory mode** (`--trajectory results/<session>/trajectory.json`) re-executes the exact sequence of configs and seeds the agents chose, with no LLM, then compares each paired delta to the original. It reports per-experiment agreement: same sign, and inside the original CI.
- **R3 live mode** needs an API key and produces a new session. It is evidence that the system works again, not that it reproduces the same trajectory.

### What to pin and ship
- **Environment**: `uv` with committed `uv.lock`, pinned Python version (`.python-version`), torch version pinned. `make setup` is the only install step. Also ship a `.devcontainer/` so judges can open the repo in GitHub Codespaces and run R1/R2/R3-smoke in the browser.
- **Data**: vendor TinyShakespeare (~1 MB) in the repo with a SHA256 check in the harness. No download at runtime.
- **Corpus**: `kg/corpus_v1.jsonl` with source, fetch date and per-record hash. Prefer arXiv metadata for shipped abstracts, because its licensing is clearer than some aggregators'; check the license of whatever you ship. Ship cached extractions and **precomputed embeddings**, and pin the embedding model name + revision.
- **LLM**: pin a dated model snapshot ID in config. Record temperature and other parameters. Treat LLM output as nondeterministic even at temperature 0, which is why recordings exist.
- **Provenance on every run**: git commit, dirty flag + diff hash, config hash, seed, `uv.lock` hash, torch/OS/device info, run order index. `eval/benchmark.py` refuses to start on a dirty tree unless `--allow-dirty` (then the diff is stored).
- **Determinism profiles**:
  - `cpu-deterministic`: `torch.use_deterministic_algorithms(True)`, fixed thread count, seeded everything. Same machine type → identical results (verified by a test that runs a run twice).
  - `mps`: best-effort; seed-to-seed **and** same-seed repeat variance measured in step 2 and reported.
  - Headline numbers are always tagged with the hardware profile they came from. All benchmark headline numbers come from **one machine** in one session (multiple teammates' Macs are not mixed).
- **Profiles**: `smoke` (CPU, tiny model, short budget, ~5 min total; proves the pipeline works) and `full` (the configuration behind the headline numbers). Never present smoke numbers as results.

### Results bundle
`results/<session_id>/` contains `notebook.sqlite`, `recordings/`, `trajectory.json`, `env.json` (hardware/software), `config.yaml` snapshot, plots, and `MANIFEST.json` with SHA256 of every file. Bundles are committed if under ~50 MB, otherwise attached to a GitHub Release with the manifest committed. `make verify` checks hashes.

### Continuous verification
GitHub Actions on Linux CPU runs on every push: `pytest` (mocked LLM), `make rederive` and diff against committed outputs, `make replay` headless (assert final state hash), and `make reexecute PROFILE=smoke` on 2 experiments of the trajectory. A green badge in the README is the judges' first evidence.

### For judges (README section)
1. A link to a **hosted replay dashboard** (Streamlit Community Cloud, serving the R2 bundle in read-only mode), so judges can see it with zero setup.
2. A demo video of the live run.
3. A "Verify our claims" block: three commands (R1, R2, R3-smoke) with expected runtimes.
4. A **claims table**: each headline claim → the analysis_id/plot that supports it → which level reproduces it → hardware it was measured on.
5. Known limitations: MPS nondeterminism, corpus coverage, and that ratios are benchmark-specific.

### Team hygiene (remote team)
One `config.yaml` schema; secrets only via `.env` (git-ignored, `.env.example` committed); recordings and bundles written only by the orchestrator, never hand-edited; one person owns the final benchmark machine and session.

## Dashboard (`app.py`, Streamlit)
- Header: LLM mode, budget used (time, $, runs), current agent.
- Hypothesis board by status, including the `rejected_known_in_corpus` column with citing span.
- Best-so-far val loss vs. wall-clock and vs. tokens, for original baseline / incumbent / all methods in a benchmark.
- Per-experiment paired-delta plot with CIs.
- KG subgraph around the current hypothesis (pyvis or streamlit-agraph).
- Decision log with score breakdowns.
- Every number links to its run_id / analysis_id / event_id.

## Benchmark (`eval/benchmark.py`)
Implements Phase 9 of the Workflow (that section is authoritative). Methods: random search; autoresearch-style keep-if-better agent; single LLM agent (no KG, no critic); full system with KG off; full system. Shared across all methods: allowed change space, original baseline, seed pools, data, harness, prewarm, compute accounting, interleaved round-robin order. Survival is always measured on audit seeds and the sealed split. Repeat with ≥2 seed-pool draws if time allows and report the spread. Any "N× faster" claim is stated only as the observed ratio on this benchmark, with conditions attached.

## Repo layout
```
config.yaml
orchestrator.py  app.py  llm.py  schemas.py
notebook/db.py
kg/{fetch.py, extract.py, index.py, graph.py, novelty.py, corpus_v1.jsonl, eval_set.jsonl}
agents/{literature,hypothesis,critic,planner,executor,analyst,decider}.py
testbed/{harness.py, model.py, space.py, sandbox.py, data/}
prompts/*.md
recordings/
results/<session_id>/
Makefile  uv.lock  .python-version  .env.example  .devcontainer/  .github/workflows/ci.yml
eval/benchmark.py
tests/
```

## Build order (each step demoable)
1. `schemas.py`, `notebook/db.py`, `llm.py` (mock + record/replay).
2. Harness + `space.py`: implement and test every allowlisted option. **Measure** throughput and seed noise (≥10 seeds of the baseline) → set run budget, `δ_min`, and seed counts.
3. Executor + sandbox + Analyst (paired screening/confirmation) + random-search loop. → benchmark method (1).
4. Hypothesis → Critic → Planner, no KG. → benchmark method (2)/(4).
5. KG: curated corpus, extraction with spans, index, novelty verdicts, retrieval eval set.
6. Decider scorer, beliefs, promotion, incumbent caching + drift check.
7. Dashboard.
8. Results bundle + `make rederive / replay / reexecute / verify`, devcontainer, CI workflow. Do the dress rehearsal in `record` mode and verify R1–R3 on a clean clone (ideally a teammate's machine).
9. Benchmark + plots.
10. Stretch: constrained code patches; cross-run `lessons`.

## Done means
- `python orchestrator.py --budget-min 30` runs unattended and records ≥20 runs, ≥5 screened hypotheses, and ≥1 confirmation attempt.
- At least one `known_in_corpus` rejection appears with a citing span (the labeled fixture guarantees this).
- KG retrieval eval reports recall@5 and verdict accuracy.
- On a fresh clone: `make setup && make rederive && make replay` succeeds on Linux CPU with no API key; CI is green; R3-smoke trajectory check passes.
- Hosted replay dashboard link works.
- README claims table is complete.
- `python eval/benchmark.py` produces the comparison plots and `eval/results.json`.
- `pytest` passes with mocked LLM.
