# FIXES2 — From explore1 to a presentable hypothesis-engine session

Source: review of `results/explore1`, the `lit-dryrun` output and the search log (2026-10-03).
Work on branch `explore`. `main` + `results/golden` stay the headline submission until a new session passes everything below.

Order = priority. Every item has an acceptance check. Invariant unchanged: LLMs propose, extract and interpret; code measures, scores, ranks and decides.

---

## P0 — Bugs (≈ 1 h total)

### P0-1. Scout schema drops its own `title` field
**Problem.** `llm.py:45`, `strict_schema`, removes every key named `"title"` to strip Pydantic metadata. It also removes it *inside* `properties`, which deletes the `Direction.title` field from the schema. The API never asks for it, so validation fails ("directions.0.title Field required") and the scout is skipped.
**Fix.** Strip `title` only as a metadata key of a schema node, never as a key of a `properties` dict.
**Accept.** New test: a Pydantic model with a field named `title` keeps it in `strict_schema` output and in `required`. `make search` runs without "scout failed".

### P0-2. Prior-art gate flips between reject and run
**Problem.** On the bigger corpus, `lit-dryrun` shows `optimizer=sgd_momentum` and `norm_position=post` alternating between a T1 claim (reject) and a T3 claim (run). The LLM picks one "nearest" claim, and which one varies.
**Fix.**
- `LiteratureOutput` returns `same_comparison_claim_ids: list[str]` (all retrieved claims that test the same comparison, each validated against the retrieved list) plus a rationale.
- Code then picks the verdict deterministically, strongest first:
  1. any T1 claim that covers our setting → reject;
  2. any T2/T3 covering claim (not contested) → soft-reject;
  3. any same-comparison claim not covering us → `method_known_setting_untested`;
  4. none → `not_found_in_corpus`.

  Ties → lowest claim id.
- The displayed passage = the claim that determined the verdict.

**Accept.** Unit test for the selection function. `lit-dryrun --repeat 3`: every row STABLE (same verdict + same action; the cited claim may differ only if the verdict and action are identical).

### P0-3. Recording must stop on an unstable gate
**Problem.** The four commands ran back to back, so `explore1` was recorded after `lit-dryrun` reported UNSTABLE.
**Fix.**
- `lit-dryrun` exits non-zero if any row is UNSTABLE.
- New `make record SESSION=… NICHE=…` target: `search` → `lit-dryrun --repeat 3` (abort on failure) → `session` → `verify` → `rederive` → `replay`.

**Accept.** `make record` aborts with a clear message when a row is UNSTABLE.

### P0-4. Pre-registered rule v2 (versioned, not retroactive)
**Problem.** Rule v1 requires all seeds to agree for PROMISING but not for HARMFUL. In `explore1`, RMSNorm is "harmful" (1.07 SD) with seeds −0.010 / +0.028 / +0.009: one seed decided it.
**Fix.**
- Add `RULE_V2_TEXT` with **symmetric** agreement: HARMFUL also requires every paired seed to be worse. Otherwise it is NO IMPROVEMENT, flagged `seed_disagreement`.
- Store `rule_version` in every session. Sessions recorded before the change (golden, explore1) keep v1. `rederive` uses the version stored in the bundle.
- The README states when v2 was written and why (citing `explore1`'s RMSNorm), and lists v1's asymmetry under limitations.

**Accept.** Unit tests for both versions. `make rederive SESSION=golden` and `SESSION=explore1` still pass unchanged.

### P0-5. Budget for the engine
`max_followup_candidates: 2` meant only one generated hypothesis ran. Raise it to 4 and `max_runs` to 32 (≈ 20 min at ~35 s/run). Keep these in config, recorded in the bundle.

---

## P1 — Wider testbed (≈ 1.5 h) — the reason combination gaps never fire
**Problem.** Only 23 of 128 claims map onto the 8 runnable methods. Those carry 7 mechanism tags across 5 categories, so no two runnable methods share a mechanism. Link and bridge (combination) gaps cannot appear.

**Fix: add building blocks that are (a) cheap to implement and test, (b) common in the retrieved literature:**

| Field | Values | Literature it maps to |
|---|---|---|
| `qk_norm` | true / false | QK-normalization (attention logit stability) |
| `weight_tying` | true / false | tied input/output embeddings |
| `z_loss_coef` | 0 / 1e-4 | output-logit z-loss (PaLM-style stability) |
| `label_smoothing` | 0 / 0.1 | label smoothing |
| `warmup_frac` | 0 / 0.02 / 0.05 / 0.1 | warm-up length |
| `grad_clip` | 0 / 1.0 | gradient clipping |
| `init_scale` | 0.5 / 1.0 / 2.0 × default | initialization scale |
| `optimizer` | + `lion` | Lion optimizer |

- Every new option: implemented in `testbed/model.py` / `harness.py`, unit-tested on CPU (shape, finite loss, deterministic smoke).
- Every new option is added to the deterministic per-block search directions, so the search fetches papers for it.
- Re-run `make search`, then the mechanism check. **Target: ≥ 3 mechanism nodes shared by ≥ 2 runnable methods.** Report the number on the dashboard whatever it is.

**Accept.** `make test` passes. The mechanism check reaches ≥ 3, or the dashboard says honestly how many there are.

---

## P1 — Explore cheaply, confirm rigorously, promote (≈ 2–3 h) — GRAPH_GAPS Part B
1. **Explore:** the top-k open gap hypotheses (k = 6) run on **1 seed** each (seed 0, reusing the baseline seed-0 run). They are labeled **EXPLORATION — NOT A RESULT**.
2. **Select finalists (code):** the best per mechanism cell, then the top 2 by single-seed improvement.
3. **Confirm:** finalists go through the paired 3-seed screening → rule v2 → extra seeds if promising.
4. **Shrinkage:** per finalist, `single-seed estimate − paired estimate`, stored and shown. This is the autoresearch-vs-us comparison measured inside one session.
5. **Promotion:** a finalist still promising on 5 seeds with all seeds agreeing becomes the **incumbent**. Its 5 runs become the new noise floor (no extra compute). Later candidates are deltas on the incumbent. The original baseline never changes; headline numbers are always vs the original baseline.
6. **Write back** results as derived edges, recompute gaps, take the next cycle while the budget lasts.

**Accept.**
- Session test with the mock LLM: one cycle explores ≥ 3, confirms ≥ 1, records shrinkage.
- With a scripted promising finalist, promotion happens and the next candidate's delta is relative to the incumbent.

---

## P1 — Score the engine's own predictions (≈ 45 min)
- Every gap already has `predicted_direction`. After screening, code records `prediction_outcome`:
  - `hit`: the predicted sign matches the measured branch;
  - `miss`: the measured branch has the opposite sign;
  - `null`: no improvement.
- The dashboard panel **"How good were the gaps?"** shows hit / miss / null counts by gap type (coverage, ABC, contradiction, link, bridge) and by novelty type. Small numbers: show counts, not percentages.
- README: report it plainly, including misses. `explore1`'s dropout gap predicted "+" (transfer of an "improves" claim) and measured harmful: a miss, and a correct one to report.

**Accept.** Unit test on the outcome table. The panel renders for `explore1` (1 miss) and new sessions.

---

## P2 — If time remains
- **`make compare-ideation`:** niche-only vs scout-only vs hybrid (no training). Report hypotheses surviving the tighten pass, novelty-type mix and overlap rate.
- **Noise floor display:** show that it is itself an estimate from 5 seeds (golden 0.0116 vs explore1 0.0083). Optionally show a bootstrap interval on the SD. Do not pool across sessions.
- **Dashboard:** gap card shows predicted direction next to the measured result; gap map highlights hits and misses.

---

## Then record `explore2`
```bash
uv run pytest -q && uv run ruff check .
make record SESSION=explore2 NICHE=niche.yaml      # aborts if the gate is unstable
make app
```
**`explore2` is presentable as the "hypothesis engine" exhibit if:**
- the scout ran (no "scout failed");
- `lit-dryrun` was all STABLE;
- ≥ 3 gap-driven hypotheses were explored and ≥ 1 confirmed;
- shrinkage and prediction outcomes are shown;
- verify, rederive and replay all pass.

Present it **alongside** `golden`, not instead of it, unless it is clearly stronger.
