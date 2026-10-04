# FINAL_FIXES — Polish, a sync hazard, and speed

Source: review of `results/explore2` and timing analysis (2026-10-03 20:05). Branch `explore`.
**Do section 0 before any commit.**

---

## 0. URGENT — iCloud is duplicating files inside the repo
The repo lives on `~/Desktop`, which macOS syncs to iCloud ("Desktop & Documents"). When the session rewrote files at seal time, iCloud created conflict copies named `<name> 2.<ext>`. There are **126** of them:
- `results/explore2/corpus/raw` (70), `corpus/tighten` (14), `corpus/tighten/raw` (21);
- `results/smoke` (15);
- `tests` (3);
- `work/replay-*` (3).

Risks:
- They would be committed.
- `make verify` does not notice extra files.
- iCloud syncing a live SQLite file (`notebook.sqlite` + WAL) can corrupt it.

**Fix.**
1. **Move the repo out of iCloud.** `mv ~/Desktop/HackNaton-7 ~/dev/HackNaton-7`, or any folder not under Desktop/Documents. Then reopen the terminal there.
2. **Remove the duplicates.** Check `tests/` by hand first, in case a real test file was overwritten:
   ```bash
   find . -name "* 2.*" -not -path "./.git/*" -not -path "./.venv/*" -print
   find . -name "* 2.*" -not -path "./.git/*" -not -path "./.venv/*" -delete
   ```
3. **`verify` must fail on unexpected files.** Any file in the bundle that is not in `MANIFEST.json` (except `MANIFEST.json` itself) → error listing them. Add a test.
4. Add `* 2.*` to `.gitignore` as a safety net.
5. Re-run `make verify rederive replay` for golden, explore1, explore2 from the new location.

---

## A. Minor bugs and polish

### A1. Dashboard opens on the right session
Default session order: `explore2`, then `golden`, then the rest. Label `explore1` as "superseded (scout schema bug, rule v1)", `mps-mock` and `smoke` as "pipeline checks: not results".

### A2. "Search degraded" wording on sealed bundles
For a sealed session, render a corpus `degraded` note as "Recorded with this limitation: …" and drop "Fix before recording".

### A3. Credit the right change when an incumbent exists
For hypotheses tested on top of an incumbent, every card, table and headline must show both:
- **vs incumbent** (this change's own effect): the branch the rule used;
- **vs original baseline** (cumulative, *includes the incumbent's gain*).

Example (explore2): "ReLU² + RMSNorm on top of RoPE: no improvement vs incumbent (−0.67 SD). vs original +1.46 SD, which includes RoPE."
No headline may attribute the cumulative gain to the new change.

### A4. Lion caveat
All Lion explorations scored ~1.64–1.70 vs ~1.53 baseline (single seed, never confirmed). The likely cause is our lr ÷ 5 scaling with a 2,000-step cosine budget, not Lion itself. Add this to limitations, and add a note on any Lion card: "exploration only; our hyperparameter scaling, not a test of Lion".

### A5. Predictions for combinations
Combination gaps currently record `no_prediction`. Add a code rule:
- predicted sign = the sum of the components' literature signs: both "+" → "+"; mixed → "?" (no prediction);
- record it as `predicted_direction`, rule shown on the card.

Optional; if skipped, say in the README that combinations make no directional prediction.

### A6. Event timestamps
`events` and `decisions` payloads carry no timestamps, so the run log can't show where wall time went. Add `ts` (UTC ISO) to every event and decision. It is excluded from the state digest, like `started_at`, so replay stays identical. Add a "timeline" expander on the dashboard.

### A7. README and claims table
- **Two exhibits:** `golden` (core loop) and `explore2` (hypothesis engine).
- **Replication table:**

  | Session | RoPE, 3 seeds | RoPE, 5 seeds |
  |---|---|---|
  | golden | 1.67 SD | 1.14 SD |
  | explore2 | 1.86 SD | 1.28 SD |

  Both promising; both shrink with more seeds.
- **Shrinkage:** RoPE's single-seed estimate −0.033 vs paired −0.016 (3 seeds) / −0.011 (5 seeds).
- **Prediction scorecard:** 1 hit, 1 miss, 1 null, 1 no prediction.
- **Extraction validation:** direction 5/9, setting 3/9 (before: 4/9, 3/9).
- **Claims table:** fill with explore2 IDs (`dec_005`, `dec_013`–`dec_021`, `dec_028`–`dec_031`, `dec_037`–`dec_042`, and the analysis IDs).

---

## B. Why it's slow, and how to make it faster

### Where the time went (measured from explore2)
| Phase | Wall time | Notes |
|---|---|---|
| Search (`make search`) | ~25–40 min | arXiv at 5 s/request × ~80 requests + backoff; 34 extraction LLM calls, **sequential** |
| Gate dry run (×3) | ~5–10 min | 28 LLM calls × 3, **sequential** |
| Session: training | **~17 min** | 28 runs × ~36 s (2,000 steps) |
| Session: LLM + tighten searches between runs | **~9 min** | 48 Opus calls; tighten = 2 arXiv requests per hypothesis at 5 s each |
| Session: after the last run until sealing | **~57 min, nothing written** | last run started 19:06 local, bundle sealed 20:04. Most likely the Mac slept (this run was started **without `caffeinate`**) or an API call stalled (no explicit timeout) |

So the real compute is ~17 minutes. The rest is waiting.

### B1. Never stall (biggest win)
- Always run sessions under `caffeinate -i`. Put it inside `make record` / `make golden` so it can't be forgotten (`caffeinate -i` only on macOS).
- **Explicit API timeout:** `anthropic.Anthropic(timeout=120, max_retries=2)`. A stalled call fails in ~6 min at worst and goes through the crash-seal path instead of hanging.
- With A6 timestamps, the dashboard timeline would show a gap like this immediately.

### B2. Run LLM calls in parallel
Anthropic calls are I/O-bound; run them concurrently with a thread pool (4–6 workers). Replay is keyed by request hash, so order doesn't matter for correctness, **but** results must be *applied* in a fixed order (sort by candidate key) so decisions stay deterministic.
- Extraction batches during search (34 calls) → concurrent.
- `lit-dryrun` repeats and candidates → concurrent.
- **Per cycle:** prepare all k candidates (gate → scientist → critic → tighten) concurrently *before* training starts, then train.

Expected: search LLM time ÷3–4; per-cycle prep from ~5 min to ~1–2 min.

### B3. Cheaper settings for routine roles
- `effort: low` for `literature` (gate), `critic_post` and extraction. Keep `medium` for `scientist_gap` and `critic_pre`. Per-role effort lives in config and is recorded in each request (it is part of the replay key, so this applies to new sessions only).
- Cap `critic_pre` output: it produced ~1,100 tokens per call. Ask for ≤ 5 confounds and ≤ 4 controls, one line each.

### B4. Faster tighten searches
- Query OpenAlex first for tighten searches (polite pool allows ~10 requests/s with `OPENALEX_MAILTO`), and arXiv only as fallback. The quote check is identical.
- Reuse tighten results across sessions with a URL-keyed cache under `work/cache/` (copied into the bundle as now).

### B5. Two training runs at once (optional, measure first)
The model (~0.8M parameters, batch 64×128) leaves most of an M-series GPU idle.
- Run 2 training processes concurrently on MPS.
- With the step budget, the **decision is unaffected** (same steps, same data order per seed). Only the tokens/s numbers change, so mark them "measured under 2× concurrency" and don't compare them across modes.
- **Measure before adopting:** `make timing` with 1 vs 2 concurrent processes. Adopt only if total throughput improves ≥ 1.5×.
- Keep pairing intact: a candidate and its baseline seed must use the same mode.

### B6. A rehearsal profile
`profile: rehearse` with `max_steps: 500`, `explore_k: 3`, `cycles: 1`, for dry runs of the whole pipeline in ~5 minutes. It is labeled like `smoke` ("not results") and never used for headline numbers.

### Expected after B1–B4
- **Search:** ~40 min → ~12–15 min.
- **Session:** ~26 min of active time with no stalls → ~20 min. Training (~17 min) now dominates.
- **With B5 (if it measures well):** ~12 min.

---

## Order of work
1. Section 0 (move the repo, delete the duplicates, `verify` checks for extra files), then re-verify all sessions.
2. A1–A4, A7 (presentation-critical). A6 + B1 (cheap, prevents another stall).
3. B2–B4 if you plan to record again. B5 only after measuring.
4. Merge `explore` → `main`, `make check`, verify all sessions, push, feature freeze, record the video.
