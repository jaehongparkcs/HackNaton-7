# SPEC — "Prior Art": an AI research lab that knows what's already been done

## Context (read first)
We are competing in a hackathon. Challenge: *Build an AI lab for the next breakthrough and make scientific discovery 10x faster. Use multiple AI agents to research, form hypotheses, plan and run experiments, learn from the results, and decide what experiment should happen next.*

Our angle: most "AI scientist" systems waste their budget rediscovering known results. We build a **literature knowledge graph (KG)** that every agent queries. It's used to (1) find open gaps, (2) reject or down-rank hypotheses that are already published, and (3) seed hypotheses from neighboring work. The KG is the differentiator, but **the loop must close**: hypotheses get turned into real code, run, measured, and the results update what we try next. A system that only does literature review and hypotheses does NOT satisfy the challenge.

Domain: CS / AI, narrowed to **one fixed experimental testbed** (below) so experiments are cheap, fast, and verifiable.

Language: Python 3.11. Must run on a MacBook (Apple Silicon, torch `mps` device, CPU fallback).

## Non-negotiable rules
1. **LLMs never state numbers.** Every metric in the UI or report comes from executed code and links to a `run_id`.
2. **Agents talk in JSON**, validated with Pydantic schemas, never free text. Invalid output → one retry with the validation error, then fail loudly.
3. **Every claim about prior work cites a paper ID** that exists in the KG. No citation, no claim.
4. **Hard budgets**: max iterations, max experiments, max wall-clock per run, max LLM spend. All set in `config.yaml`.
5. **Failures are data.** Crashed or timed-out runs are logged with stderr and fed back to the planner, not silently retried forever.

## Experimental testbed (default; keep it pluggable)
- `testbed/` holds a fixed harness: a tiny character-level transformer trained on TinyShakespeare (download once, cache locally).
- Each experiment = a **patch to `testbed/train.py`** or to a config dict (optimizer, LR schedule, normalization, init, attention variant, data ordering, regularization…), trained for a **fixed budget of ~90 s wall-clock** (budget in seconds, not steps, so changes that are faster count).
- Primary metric: validation loss at end of budget. Also log: train loss curve, tokens/sec, peak memory.
- Every hypothesis gets run on **≥3 seeds**, plus the current baseline on the same seeds.
- Interface so another testbed can be swapped in later:
  ```python
  class Testbed(Protocol):
      def describe(self) -> str            # given to agents as context
      def baseline_config(self) -> dict
      def run(self, patch: ExperimentPatch, seed: int, budget_s: int) -> RunResult
  ```

## Architecture

### Shared state: the Lab Notebook (SQLite via SQLModel)
Tables: `papers`, `kg_edges`, `hypotheses`, `experiments`, `runs`, `beliefs`, `events`.
- `hypotheses`: id, statement, mechanism, prediction (direction + expected effect size), falsification criterion, status (`proposed|rejected_prior_art|rejected_critic|queued|running|supported|refuted|inconclusive`), novelty score, nearest prior-art paper IDs, parent hypothesis id.
- `runs`: run_id, experiment_id, seed, config/patch, metric values, curves (JSON), stdout/stderr, status, duration.
- `events`: append-only log of every agent action (agent, input summary, output JSON, timestamp, cost). This drives the dashboard and is the audit trail.

### Knowledge graph (`kg/`)
- **Sources**: arXiv API + Semantic Scholar Graph API (citations, abstracts; use an API key if available, and respect rate limits) and/or OpenAlex as fallback. Cache every response to disk; the demo must work offline from the cache.
- **Scope**: seed with ~300–1000 papers from the testbed's area (small-LM training efficiency, optimizers, LR schedules, normalization, initialization). Seed via keyword queries plus 1-hop citation expansion.
- **Nodes**: `Paper`, `Method`, `Task/Dataset`, `Claim` (method X → effect on metric Y in setting Z). Methods and claims are extracted from abstracts by an LLM with a strict schema, and each one keeps a pointer to its source paper.
- **Edges**: `cites`, `proposes(Paper→Method)`, `evaluates_on`, `claims`, `variant_of(Method→Method)`.
- **Storage**: networkx graph pickled + embeddings (sentence-transformers, e.g. `all-MiniLM-L6-v2`) in a numpy/FAISS index.
- **API** used by agents:
  - `search(query, k)` → papers/methods with scores
  - `novelty_check(hypothesis)` → `{novelty: 0–1, nearest: [paper_ids], verdict: novel|incremental|known, rationale}`. Combines embedding similarity against claims and methods with an LLM judge that must cite the specific nearest claims.
  - `gaps(topic)` → method×setting combinations that are sparsely covered (e.g. methods with claims in vision but none for small LMs, or contradicting claims)
  - `neighbors(method_id)` → related methods, for hypothesis seeding

### Agents (`agents/`, one module each, all subclass a common `Agent` with `run(input: Model) -> Model`)
1. **Literature agent**: builds/updates the KG; given the current notebook state, returns a field summary, open gaps, and contradictions, each with paper citations. Re-invoked with a *targeted* query when results surprise us.
2. **Hypothesis agent**: proposes N hypotheses from KG gaps + past results. Each one has a mechanism, a directional prediction with expected effect size, and a falsification criterion. Must call `novelty_check`; hypotheses marked `known` are logged as `rejected_prior_art` with the citing paper (show these on the dashboard, since this is the KG proving its value).
3. **Critic agent**: attacks each surviving hypothesis: confounds (e.g. "faster wall-clock, not better learning"), whether it's testable in the 90 s budget, whether the prediction is falsifiable, and whether it's just a hyperparameter retune of baseline. Output: accept / revise / reject + required controls.
4. **Planner agent**: turns an accepted hypothesis into an `ExperimentPatch` (a config diff or a unified diff against `train.py`) plus required controls and number of seeds.
5. **Executor** (mostly code, minimal LLM): applies the patch in a temp copy of the testbed, runs in a subprocess with timeout, captures everything. On crash: an LLM repair step gets the stderr and may fix the patch at most twice; otherwise the failure is recorded and returned to the planner.
6. **Analyst + Decider**:
   - Analyst runs **real statistics in code**: per-seed deltas vs. baseline, mean ± 95% CI (bootstrap), Welch t-test, and Holm correction across hypotheses tested this round. Sets status `supported|refuted|inconclusive` by a rule in code, not by the LLM. The LLM only writes the interpretation, and only quotes numbers from the stats object.
   - Decider updates beliefs and picks the next action (see below). If something is supported, it becomes the new baseline ("ratchet"), and the Literature agent is asked whether the combined change is known.

### Next-experiment selection (this is where "10x" comes from; do not just ask the LLM)
- Each hypothesis carries a belief that it improves the metric: a Normal prior on effect size, with the prior mean set from the hypothesis agent's predicted effect, shrunk toward 0, and the variance widened by novelty.
- After runs, update with the observed per-seed deltas (conjugate Normal update).
- Score queued items with **expected improvement per second of compute**: `EI(effect posterior over current best) / expected_runtime`, plus a small bonus for novelty and for information gain on hypotheses that are currently inconclusive (decide between more seeds vs. moving on).
- The LLM can propose; the scorer ranks. Log the score breakdown for every decision so it's visible in the UI.

### Orchestrator (`orchestrator.py`)
A plain Python loop (no heavy framework needed):
```
lit = literature.survey()
while budget_left():
    hyps = hypothesis.propose(lit, notebook)          # with novelty_check
    hyps = critic.review(hyps)
    queue.add(planner.plan(h) for h in accepted(hyps))
    exp = decider.select(queue)                        # scorer, not LLM
    results = executor.run(exp)                        # seeds + baseline
    analysis = analyst.analyze(results)
    decider.update(analysis)                           # beliefs, ratchet baseline
    if analysis.surprising: lit = literature.targeted(analysis)
```
- Human-in-the-loop: a `pause_for_approval` flag. When on, the dashboard shows the selected next experiment with Approve / Skip / Redirect (free-text note passed to the hypothesis agent).

### LLM layer (`llm.py`)
- Single wrapper around the Anthropic SDK (model name in config), with structured output via tool-use/JSON schema, retry, caching of identical prompts to disk, and token/cost accounting written to `events`.
- Keep prompts in `prompts/*.md`, not inline strings.

## Dashboard (`app.py`, Streamlit)
Live view refreshed from SQLite:
- **Loop status**: which agent is active, iteration, budget used (time, $, experiments).
- **Hypothesis board**: kanban by status. Include a separate column for "rejected — prior art" showing the citing paper.
- **Results**: best-so-far val loss vs. wall-clock, our system vs. baselines; per-experiment CI plot.
- **KG view**: pyvis/streamlit-agraph subgraph around the current hypothesis, with nearest prior art highlighted.
- **Decision log**: why each experiment was chosen (score breakdown).
- Every number is clickable → run_id → config, curves, logs.

## Evaluation: proving "10x faster"
`eval/benchmark.py` runs, under the **same compute budget** (e.g. 45 min) and same seeds:
1. **Random search** over the same config space.
2. **Single LLM agent**, no KG, no critic, asked "what should I try next?" each step.
3. **Full system.**

Report: best val-loss improvement vs. time; **experiments-to-target** (time to reach X% improvement over baseline); **rediscovery rate** (fraction of proposed hypotheses that the KG flags as known, ablation with KG off); wasted runs (crashes, invalid). Save as `eval/results.json` + plots. Be honest in the pitch about what is measured: "10x" must be one of these ratios, not a vibe.

## Repo layout
```
config.yaml
orchestrator.py
app.py
llm.py
schemas.py          # all Pydantic models shared between agents
notebook/db.py
kg/{ingest.py, extract.py, graph.py, novelty.py}
agents/{literature,hypothesis,critic,planner,executor,analyst,decider}.py
testbed/{train.py, harness.py, data/}
prompts/*.md
eval/benchmark.py
tests/
```

## Build order (each step demoable on its own)
1. `schemas.py`, `notebook/db.py`, `llm.py` with a mock-LLM mode for tests.
2. Testbed harness: baseline run in ≤90 s on MPS, deterministic per seed; `run()` returns `RunResult`.
3. Executor + Analyst stats (no LLM): random-search loop end to end. **This alone is baseline #1.**
4. Hypothesis → Critic → Planner agents; full loop without the KG. **This is baseline #2-ish.**
5. KG ingest from cache, novelty check, gaps; wire into Hypothesis + Literature agents.
6. Decider scorer + belief updates + baseline ratchet.
7. Streamlit dashboard.
8. Benchmark script + plots.
9. Stretch: cross-run memory (`lessons` table: which hypothesis types succeeded/failed, fed back into the hypothesis prompt).

## Done means
- `python orchestrator.py --budget-min 30` runs unattended, produces ≥10 executed experiments, and at least one hypothesis rejected for prior art with a real citation.
- `streamlit run app.py` shows the loop live.
- `python eval/benchmark.py` produces the comparison plot.
- `pytest` passes, with LLM calls mocked.
