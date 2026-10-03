# PRIOR ART

> *Autoresearch with a literature memory and honest statistics: it refuses to spend compute on known ideas, tests what the literature hasn't, and never lets an LLM state a number or make a decision.*

One complete, auditable agentic-science loop: **literature → hypothesis → real experiment → measured decision.**

**Governing principle.** LLMs propose and interpret. Deterministic code measures, decides and records.
**Invariant.** An LLM mistake can waste compute or skip an idea, but it cannot produce a measured result or a decision. Every number comes from a run; every decision comes from a rule written before the run.

This is the hackathon profile ([BUILD_PLAN.md](BUILD_PLAN.md)). [SPEC.md](SPEC.md) is the fuller platform design and where this goes next.

## What we claim (and only this)

1. A known idea is rejected **before** compute, with an exact source passage.
2. An idea in a setting not found in our curated snapshot is tested with **real paired runs**, measured against the measured **noise floor**.
3. The same data shows how often a **single-run keep/revert decision** (autoresearch-style) would have been decided by the seed alone. This costs no extra compute.
4. The measured result changes the **next decision** through a pre-registered rule.

We do **not** claim 10×, confirmation, statistical significance, or global novelty. Results are labeled **SCREENING RESULT — NOT CONFIRMED**.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all dependencies are installed from the lockfile.

```bash
make setup               # one-time install
make test                # 60+ tests on CPU with a mocked LLM; no key, no GPU
make verify-snapshot     # every claim span is verbatim in its arXiv abstract
make smoke               # CPU pipeline check with the MOCK LLM (~10 s). Not a result.
make app                 # dashboard over every notebook in results/ and work/
```

**The real session** (live Claude + real training; needs a Mac with Apple Silicon and an API key):

```bash
cp .env.example .env     # then set ANTHROPIC_API_KEY
make timing              # ~1 min: measure throughput, then set profiles.full.train_seconds in config.yaml
make golden              # ~15-20 min: 5 baseline + 3 candidate + 2-3 follow-up runs, every LLM call recorded
make verify rederive replay
make app
```

`make golden` writes `results/golden/` (override with `SESSION=name`). It refuses to overwrite an existing session unless you pass `--force` to the CLI. Commit the bundle so judges can run the verification commands below with no key.

Budgets (runs, wall-clock minutes, LLM spend) are in [config.yaml](config.yaml); the session stops cleanly and records `budget_exhausted` if one is hit.

To run without `make`: `uv run python -m prior_art session --session golden --profile full --llm live` (also `replay`, `rederive`, `verify`, `timing`; `--help` lists them).

## Verify our claims

| Level | Command | Needs | Guarantee |
|---|---|---|---|
| R1 re-derive | `make rederive` | CPU, seconds | Recomputes every statistic from the stored runs; exact match required |
| R2 replay | `make replay` | CPU, **no API key** | Rebuilds the whole session from recorded LLM responses + recorded runs and asserts the final state digest is identical |
| integrity | `make verify` | nothing | SHA256 manifest of the results bundle |

```bash
make setup          # uv sync --frozen (pinned python + torch via uv.lock)
make verify rederive replay      # SESSION=golden by default
make app            # dashboard
```

Nothing is retrained and no model is called in R1/R2. Re-executing the experiments needs a Mac (MPS) and re-recording needs an API key (`make golden`); the numbers will differ slightly, see Limitations.

### Claims table

Filled from the recorded session. Each row must point at stored IDs; the dashboard resolves them.

| Claim | Evidence | How to reproduce | Hardware |
|---|---|---|---|
| Fixture A rejected as prior art with an exact passage | decision `prior_art_rejection` (`dec_002`), claim `c01`, arXiv:2002.04745 | `make replay` | none (CPU) |
| Noise floor of the baseline | `noise_baseline`, runs `run_…` ×5 | `make rederive` | M-series Mac (MPS) |
| Fixture B measured against the noise floor | analysis `ana_…` | `make rederive` | M-series Mac (MPS) |
| Single-run loop would have flipped on the seed | analysis `.counterfactual` | `make rederive` | M-series Mac (MPS) |
| Pre-registered rule picked the next action | decision `next_action` | `make replay` | none (CPU) |

> IDs and values are filled in after the recorded `make golden` run. Do not publish this table with placeholders.

## How it works

```
fixtures ──► Literature Agent (Claude) ──► prior-art verdict + exact passage (from the snapshot, by code)
                │ known_in_corpus ──► REJECTED, "N runs avoided"
                ▼
           Scientist (Claude) ──► typed config delta (validated against an allowlisted schema)
                ▼
           Critic (Claude) ──► accept / reject BEFORE compute (confounds, controls)
                ▼
           Runner (PyTorch, MPS) ──► paired runs, baseline runs reused
                ▼
           Statistician (pure Python) ──► deltas, noise floor, single-run counterfactual, label
                ▼
           Decider (pure Python) ──► pre-registered rule ──► extra seeds | next candidate | literature check
                ▼
           Evidence store (SQLite) ──► hypothesis → claim span → config → run → analysis → decision
```

| Component | Type | Where |
|---|---|---|
| Literature, Scientist, Critic | LLM (one pinned model, three prompts, strict JSON) | [prior_art/literature.py](prior_art/literature.py), [prior_art/agents.py](prior_art/agents.py), [prompts/](prompts/) |
| Orchestrator, Runner, Statistician, Decider | plain Python | [prior_art/orchestrator.py](prior_art/orchestrator.py), [prior_art/runner.py](prior_art/runner.py), [prior_art/stats.py](prior_art/stats.py) |
| Testbed | tiny char-level GPT, TinyShakespeare (vendored + checksum) | [prior_art/testbed/](prior_art/testbed/) |
| Evidence store | SQLite (WAL) | [prior_art/store.py](prior_art/store.py) |
| Dashboard | Streamlit | [app.py](app.py) |

### How the invariant is enforced (not just promised)

- **No field to put a number in.** LLM output schemas have no measurement, label or decision fields (tested in `tests/test_schemas_model.py`).
- **Passages come from the snapshot, not the LLM.** The agent picks a `claim_id` that code validates against the retrieved list; the quoted text is copied from the snapshot. Every span is checked verbatim against its arXiv abstract at load time (`make verify-snapshot`).
- **Configs are typed.** The Scientist's changes are parsed into a Pydantic allowlist, restricted per fixture to the fields the hypothesis names, and range-checked.
- **Number guard.** The Critic's post-run reading may only quote numbers present in the code-computed facts table; otherwise it is discarded and replaced with a deterministic template (the swap is recorded).
- **Pre-registered rule.** Which branch fires is computed by `stats.next_action`. The rule text is stored and shown verbatim on the dashboard.

### Experiment design

- Tiny char-level Transformer (4 layers, 128-d, ~0.8M parameters), TinyShakespeare, fixed training wall-clock budget (default 60 s). The timer excludes model construction, a throw-away prewarm and eval.
- **Noise floor:** baseline on 5 seeds, SD of final validation loss. Seeds 0–2 are reused as the paired baseline for every candidate.
- **Paired deltas** (candidate − baseline, same seed; negative = better), mean delta, shown against ±1 SD. No p-values.
- We also log `tokens_seen` and read both eval curves at equal tokens, which separates "learns better" from "runs faster". With a cosine schedule a mid-curve point is not a finished schedule, so this is a diagnostic, not a headline number.
- **Single-run counterfactual:** every (candidate seed × baseline seed) pairing is replayed through a "keep if one run beats one run" rule using runs already made.

### The pre-registered rule

Let SD be the baseline seed-to-seed SD and improvement = baseline − candidate val loss, averaged over paired seeds.

1. **Harmful** (worse by more than 1×SD): record a contradiction with the cited claim; next action is a targeted Literature Agent check.
2. **Promising** (improvement > 1×SD and every paired seed improves): next action is extra seeds for the same change (toward confirmation).
3. **No improvement** (otherwise): next action is the next untested-setting candidate in a fixed literature-priority order (`candidate_order` in [config.yaml](config.yaml)).

*Gap we filled before any run:* the plan did not say what happens when the mean improvement exceeds 1×SD but the seeds disagree in sign. We route that to "no improvement" and flag it as seed disagreement. It is in the rule text and has a test.

## The curated literature snapshot (read this before judging the verdicts)

The snapshot is **10 arXiv papers / 13 claims** covering exactly the dimensions the testbed exposes (norm, Pre/Post-LN, optimizer, schedule, activation, positional encoding, dropout). It is deliberately tiny for the demo. Therefore:

- With so few papers, *most* things will be "not found in curated snapshot". The UI says exactly that, with the paper count, and never says "novel". Verdicts are scoped to the snapshot.
- Shipped text is arXiv abstracts only (clearest licensing). Claim `source_span`s are verbatim substrings, enforced by code. Claim `setting` fields are paraphrased from the abstract alone, not the full paper.
- The full design is a 100–300 paper corpus with retrieval evaluation ([SPEC.md](SPEC.md)).
- Fixtures are predefined hypotheses, labeled **FIXTURE** in the UI. They are not measurements.
  - **A (known):** Pre-LN trains without warm-up (Xiong et al., 2020).
  - **B (setting untested):** RMSNorm vs LayerNorm on a tiny char-level LM under a 60 s budget. We searched for a paper covering this exact setting and found none, but broader literature reports RMSNorm ≈ LayerNorm, so "no improvement" is the likely outcome. That is a fine result: it fires the "next candidate" branch.
  - Follow-up candidates (fixed order): SwiGLU, squared ReLU, RoPE.

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

- **Pinned:** Python 3.12 (`.python-version`), `uv.lock`, vendored dataset with SHA256 check, committed snapshot and fixtures.
- **Every LLM call is recorded** (full request, response, usage) in `results/<session>/recordings/llm.jsonl`, keyed by a hash of model, effort, role, prompts and schema. Replay serves from it and errors on a miss, never silently going live. A prompt edit after recording therefore makes replay fail loudly; re-record.
- **Model:** `llm.model` in [config.yaml](config.yaml) is `claude-opus-5-5`. Claude 5 model IDs have no date suffix, so the pin is the ID, and each recording stores the served model. The plan asked for temperature 0; this model family rejects `temperature`/`top_p`, so determinism comes from the recordings. The refusal fallback (`llm.server_side_fallback`, a beta) is on by default; set it to `false` to disable.
- **Each run stores** git commit and dirty flag, config hash, seed, device, torch/OS versions, `uv.lock` hash, run order and timestamp.
- **Profiles:** `full` (the headline profile, MPS) and `smoke` (CPU, tiny, bit-deterministic; used by tests and `make smoke`). Smoke and mock-LLM output is marked in the UI and must never be presented as a result.

## Known limitations

- **Screening only.** Three paired seeds; no confirmation, no p-values, no multiple-comparison control.
- **MPS is not bit-deterministic.** Re-executing on another Mac gives different numbers; R1/R2 verify the shipped numbers, not that training reproduces them to the bit. The CPU profile is bit-deterministic (tested).
- **Wall-clock budgets** make step counts depend on machine speed (and e.g. RMSNorm is not faster in eager PyTorch). That is why tokens seen and the equal-token delta are reported.
- **Tiny snapshot** (see above) and an LLM prior-art verdict that is a judgment call over retrieved claims. The passage is shown so a human can check it.
- **One fixed testbed and one model scale.** Results do not transfer by themselves.
- **LLM calls have not been exercised against the live API in the development environment** (no key available); the record/replay layer and all agents are tested with a scripted mock. First `make golden` on a keyed machine is the live integration test.

## Repository layout

```
config.yaml  pyproject.toml  uv.lock  Makefile  app.py
prior_art/   schemas.py llm.py literature.py agents.py orchestrator.py runner.py stats.py
             store.py bundle.py rederive.py cli.py mock_llm.py  testbed/{harness,model}.py
prompts/     literature.md scientist.md critic_pre.md critic_post.md
data/        tinyshakespeare.txt papers.json claims.json fixtures.json
scripts/     fetch_snapshot.py verify_snapshot.py
tests/       pytest, CPU, mocked LLM
results/<session>/  notebook.sqlite runs.jsonl recordings/llm.jsonl env.json state.json MANIFEST.json
```

## Cut for today (see BUILD_PLAN §12)

Matchmaker/Elo, 100+ paper corpus and live ingestion, retrieval benchmark, confirmation tests, sealed audit split and full competitor benchmark, re-execute mode, CI, devcontainer, hash-chain provenance, code patches, second LLM provider.
