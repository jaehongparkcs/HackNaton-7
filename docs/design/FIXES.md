# FIXES — before re-recording `make golden`

Source: review of the codebase, the `results/mps-mock` bundle, and the live Noesis Lab run.
Order = priority. Each fix has an acceptance check. **Stop any in-progress `make golden`: every P0 fix changes runs or recordings, so record once, after P0.**

Invariant to preserve in every fix: LLMs propose and interpret; code measures, decides and records. No LLM output may set a number, a label, a branch or a decision.

---

## P0-1. Decide on a fixed step budget, not wall-clock
**Problem.** Same config (`6ea351a6bc162b08`, SwiGLU), two Macs/sessions:
- `results/mps-mock`: ~31M tokens per run → **4.0 SD worse**.
- Live run: ~16M tokens per run (1,984 steps in 60 s) → **1.9 SD better**.

Under a wall-clock budget, the conclusion depends on the machine. That is the autoresearch weakness we pitch against. Judges re-executing on other hardware would get different conclusions.

**Fix.**
- `config.yaml` → `profiles.full`: `budget_mode: steps`, `max_steps: 2000`, `eval_every_steps: 250`. That is ~30 s on a fast M-series Mac, ~60 s on the slower one.
- Keep logging `train_seconds` and tokens/s per run. Show throughput (candidate/baseline tokens/s) on the dashboard as a **separate, secondary** metric: "speed cost/benefit", never mixed into the decision.
- `prompts/*` and `TESTBED_DESCRIPTION` in `literature.py`: replace "fixed ~60 second wall-clock budget" with "fixed 2,000-step budget (~0.5–1 min on Apple Silicon)".
- The equal-token delta becomes redundant (tokens are equal by construction). Keep the field, but hide it in the UI when `budget_mode == steps`.
- README: replace the wall-clock description; add a "Why steps, not seconds" note citing the sign flip above. Keep both old bundles as an exhibit, labeled "different machines, wall-clock profile, mock-LLM session for one of them".

**Accept.** Two runs of the same config + seed on two different Macs see identical `tokens_seen` and `steps`. `make test` passes. Nothing in the UI says "60 second budget".

---

## P0-2. Apply the pre-registered rule to every screening result
**Problem.** `Session.apply_rule` runs only on the first analysis. In the live run SwiGLU was **promising**, yet no extra seeds ran, while the rule text on the same page says "PROMISING → extra seeds". In the mock run, a **harmful** follow-up never triggered the literature check.

**Fix (`orchestrator.py`).**
- After every `_analyze(...)` that produces a screening result (not an `extra_seeds_result`), call `stats.next_action` and execute it.
- Bound the chain with `budget.max_followup_candidates` and `budget.max_runs` (already in config). Record `budget_exhausted` when a bound stops it.
- After `extra_seeds` or `literature_check`, continue with the next candidate (P0-3) if budget remains. Never run extra seeds twice for the same hypothesis.
- Every decision stores `triggered_by` → previous decision id (already supported).

**Accept.** New test in `tests/test_session.py`: with a mock where the first follow-up is promising, an `extra_seeds_result` decision exists for it. With a harmful follow-up, a `literature_check` decision exists. The dashboard shows a chain of ≥3 decisions.

---

## P0-3. The lab picks the next experiment from the literature (replace `candidate_order`)
**Problem.** The niche, the papers, the hypotheses and the follow-up order are all human-scripted (`fixtures.json`, `candidate_order`). The Scientist only translates text into a config change. The challenge asks the system to form hypotheses and decide what runs next.

**Fix.**
1. `data/claims.json`: add a human-curated field `config_change` (a single-field delta, e.g. `{"activation": "swiglu"}`) to every claim that maps to the config space. Use `null` for context-only claims.
2. New pure function `stats.candidate_queue(baseline, claims, tested)` (no LLM, no I/O):
   - **Candidates** = every allowlisted single-field change from the baseline that at least one claim maps to:
     - `norm=rmsnorm`
     - `norm_position=post`
     - `activation=swiglu|relu2`
     - `pos_encoding=rope`
     - `schedule=constant`
     - `optimizer=sgd_momentum`
     - `dropout=0.1`

     Exclude pure retunes (lr, batch size).
   - **Priority (deterministic):**
     - (a) supporting claims with `expected_outcome == "improves"` before `no_worse` before `context`;
     - (b) more supporting claims first;
     - (c) alphabetical by field=value.
   - Exclude anything already tested or gated out this session.
3. The orchestrator pops candidates in that order. For each one:
   - the Literature agent gates it;
   - the Scientist writes statement, mechanism and falsification rule (allowed field = the candidate's field only);
   - the Critic reviews;
   - code runs it.
4. Keep **Fixture A and Fixture B** as the first two items, labeled `FIXTURE`. Everything after comes from the queue and is labeled `GENERATED`.
5. Delete `candidate_order` and the `followup_candidate` fixtures (or keep them only in mock tests).
6. Dashboard: a "Why this experiment next" panel showing the queue, each item's priority tuple and its supporting claim IDs.
7. README: "The PI chooses the objective and testbed (the niche). The lab chooses the experiments: the queue is generated and ordered by code from the literature snapshot."

**Accept.** Unit test: `candidate_queue` is deterministic and ordered as specified, and an already-tested candidate is not returned. Session test: follow-ups come from the queue, and each has `supporting_claim_ids` stored.

---

## P0-4. Make the prior-art verdict consistent: setting coverage is curated, not judged
**Problem.** The live Fixture A rationale reads: *"Under the strict setting rule, that general cross-application statement covers our tiny character-level Transformer LM, even though the abstract does not specifically test…"*. Meanwhile RMSNorm's equally general "diverse network architectures" was judged *not* to cover us. The prompt contains a clause that looks written to pass Fixture A, and a judge can see the inconsistency on screen.

**Fix: move setting coverage out of the LLM.**
- `data/claims.json`: add a human-curated `covers_our_setting: true|false` and a one-line `coverage_note` to every claim. Apply one written criterion to all claims and put it in the README, e.g.: *covers = the abstract states the result for Transformer language models, or for Transformers in general without restricting the task.* Apply it honestly to c01 and c04 alike, even if that changes which fixture is "known".
- LLM output (`LiteratureOutput`) becomes: `{same_comparison: bool, claim_id, rationale}`. The LLM only judges whether a retrieved claim tests the same change.
- Code derives the verdict:
  - `same_comparison` and `covers_our_setting` → `known_in_corpus`;
  - `same_comparison` and not covering → `method_known_setting_untested`;
  - otherwise → `not_found_in_corpus`.

  This removes one LLM decision channel, which strengthens the invariant.
- `prompts/literature.md`: delete the "strict setting rule … state generally …" clause. Ask only "is this the same comparison?".
- If c01 no longer qualifies under the criterion, pick a Fixture A whose claim does. The candidate must be verified verbatim from arXiv.
- Dashboard: show `covers_our_setting` + `coverage_note` next to the passage, labeled "human-curated".

**Accept.** Unit test: the verdict is a pure function of (`same_comparison`, `covers_our_setting`). Live dry-run of the Literature agent on all fixtures + queue candidates, 2× each: identical verdicts both times. Cost is cents; record it, don't ship it.

---

## P1-5. Evidence relations that don't overclaim
**Problem.**
- SwiGLU harmful → stored as **"contradicts"** GLU Variants (c09), a T5-scale seq2seq claim that our own gate says is a different setting.
- RMSNorm → stored as **"agrees"** with c04, although c04 also claims RMSNorm is 7–64% faster and we measured ~13% fewer tokens/s. That slowdown comes from the hand-written implementation.

**Fix (`stats.relate_to_claim`, pure table lookup):**
- If the claim does not cover our setting (P0-4) → relations become `consistent_in_our_setting` / `not_reproduced_in_our_setting` / `inconclusive`. Use `contradicts` / `agrees` only when `covers_our_setting` is true.
- For claims with a speed component (add `claims_speed: true` to c04/c05): if measured throughput moves the other way by more than 5%, append `partial: speed claim not reproduced (implementation-dependent)`.
- `DerivedEvidence.relation` Literal updated accordingly. The dashboard shows the relation verbatim.

**Accept.** Unit tests for each branch of the table.

---

## P1-6. Use PyTorch's RMSNorm kernel
**Problem.** The hand-written `RMSNorm` in `testbed/model.py` is slower than fused `nn.LayerNorm` (~487k vs ~516k tok/s in `mps-mock`), so we partly measure our own code. The Critic flagged this itself.

**Fix.** `make_norm`: `nn.RMSNorm(dim, eps=1e-6)` (torch ≥ 2.4; pinned torch is 2.14). Keep the class only if needed for old bundles.

**Accept.** `make timing`-style check: RMSNorm tokens/s ≥ LayerNorm tokens/s − 2% on MPS. If it is still slower, keep it anyway and say so in the README (an honest implementation cost).

---

## P1-7. Counterfactual: lead with same-seed decisions, grade the wording
**Problem.**
- The headline uses all 15 cross-seed pairings, which overstates seed dependence relative to a loop with a fixed seed.
- 14/15 keeps is worded "depended on seed luck".

**Fix (`stats.counterfactual`).**
- Headline = same-seed decisions (`keep, keep, revert`).
- Cross-seed pairings become a secondary line: "if the seed varies between runs".
- Wording by keep rate:
  - 0 or n → "seed never changed the decision";
  - ≤ 2 flips or ≥ n−2 → "almost always X (k of n pairings flip)";
  - otherwise → "decided by seed luck (k of n)".
- Facts table for the Critic updated to match (number guard).

**Accept.** Unit tests for the three wording bands.

---

## P2-8. Drift check
All baseline runs happen first; candidates and follow-ups later. With a step budget, drift affects only speed, not the decision, so this is now low priority.

**Fix.** Record tokens/s against `run_order` and plot it on the dashboard. Flag if baseline-era vs. late-session throughput differs by more than 10%.

---

## P2-9. Linux judges: CPU torch wheel
`uv sync` on Linux pulls ~3 GB of CUDA wheels, which filled a 10 GB test VM.

**Fix.** In `pyproject.toml`, point torch at the PyTorch CPU index on Linux only:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then `uv lock`.

**Accept.** `make setup && make replay SESSION=golden` on a Linux CPU box downloads < 1 GB.

---

## P2-10. Evidence-chain view (the viewable "knowledge graph")
Replace the JSON tree in the dashboard's provenance section with `st.graphviz_chart`, generated from the `links` table for a selected decision:

> paper → claim (passage, coverage flag) → verdict → hypothesis → config → runs → analysis → decision → next decision

- LLM-produced nodes dashed, code-produced nodes solid.
- Fixture A's chain ends in a "3 runs avoided" node.
- Optional second panel: a grid with one row per candidate, showing paper claim, coverage, verdict, measured result and relation.

---

## After the fixes
1. `make test lint verify-snapshot smoke`, then `make check`.
2. Live dry-run of the Literature agent only (P0-4 acceptance).
3. One `make golden` on **one** machine. Commit `results/golden/`.
4. `make verify rederive replay`, then replay on a teammate's machine (and Linux if P2-9 is done).
5. Fill the README claims table from the recorded IDs; add the "why steps" exhibit and the "lab picks the experiments" paragraph.
