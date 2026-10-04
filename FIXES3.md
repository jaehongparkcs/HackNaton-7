# FIXES3 — After the aborted `explore2` recording

Source: `make record SESSION=explore2` log (2026-10-03 17:05). The gate check correctly aborted; nothing was recorded.
Order = priority. Same invariant and acceptance-check style as FIXES2.

---

## P0-1. arXiv rate limiting emptied out the scout (most important)
**What happened.**
- ~65 arXiv requests failed with **HTTP 429**: most scout-direction queries (`arxiv_dir`) and title-hint lookups (`arxiv_title`).
- Result: `papers_from_directions: 4` (14 directions found 4 papers), and `title_hints_not_found: 21`, which is mostly *failed requests*, not hallucinated titles.
- The corpus was frozen anyway, labelled only "degraded". The scout part of the corpus is effectively hollow.

**Fix.**
1. **Backoff that respects arXiv.**
   - On 429/503: honor `Retry-After` if present, else exponential backoff (15 s → 45 s → 120 s), up to 3 retries.
   - Raise `lit_search.arxiv_min_interval_s` from 3.0 to 5.0.
   - Keep exactly one request in flight at a time (no concurrency).
2. **Resume from cache.** `RawCache` must key successful responses by URL and reuse them on a re-run, so a re-run only fetches what failed. Never cache failures.
3. **Failed ≠ not found.** A title-hint lookup that errored is recorded as `request_failed`, not `not_found`. The scout panel shows the three counts separately (found / not found / request failed).
4. **Do not freeze a badly degraded corpus.**
   - If more than 10% of planned requests failed after retries, `search` exits non-zero with a summary.
   - `make record` aborts and nothing is frozen.
   - Re-running resumes from cache.
5. **Optional fallback:** for scout directions only, if arXiv keeps failing, query OpenAlex search (already integrated; higher limits; reconstruct abstracts from `abstract_inverted_index`). Mark those papers `source: openalex`. The quote check works the same.

**Accept.**
- Unit tests with a fake HTTP layer: 429 → backoff → success; cache hit on re-run; >10% failures → non-zero exit.
- Live: `papers_from_directions` is roughly 10 × number of directions, and no 429s remain in `degraded`.

**Operational:** wait ~15–30 minutes before re-running. Today's repeated searches and dry runs likely tripped arXiv's limiter.

## P0-2. Gate still unstable on `norm_position=post`
**What happened.** Verdict `known_in_corpus` both times, but the action alternated reject (via c01, T1) and soft-reject (via a T3 claim). The LLM's list of same-comparison claims sometimes included c01, which is about Pre-LN *without warm-up*: arguably not the same comparison as "switch to Post-LN".

**Fix (deterministic constraint, not more prompting).**
- For **generated** candidates, code keeps only listed claims whose `config_change` **equals the candidate's delta**. Claims with a different or null `config_change` cannot decide the verdict; they stay as context.
- **Fixtures** keep using their human-curated `cited_claim_ids` (Fixture A → c01).
- Optional robustness: ask the gate 3 times at the dry run and use a claim only if it appears in ≥ 2 of 3 lists. Record all three lists.

**Accept.**
- Unit test: a T1 claim with a non-matching `config_change` cannot reject a generated candidate.
- `lit-dryrun --repeat 3`: all STABLE.

## P0-3. Fixture B leakage
**What happened.** With 160 papers, Fixture B (RMSNorm) became `not_found_in_corpus`; with the curated snapshot only, it was `method_known_setting_untested`. The action is the same (run) but the verdict changed. Separately, the gap engine already generates `norm=rmsnorm` with the c04 citation, so Fixture B is redundant in explore sessions.

**Fix.**
- In explore sessions, **drop Fixture B**: the gap engine owns RMSNorm. Keep Fixture A (the guaranteed prior-art demo).
- The leakage check compares **action** (run / reject / soft-reject / hold) for STABLE vs UNSTABLE. Label-only changes are reported as `LEAKAGE (label only)` and do not abort.
- README: report the leakage plainly ("Fixture B's verdict depended on corpus size; we dropped it in favor of the generated candidate").

**Accept.** The dry run no longer lists Fixture B in explore mode. A label-only change doesn't abort the recording.

## P1-4. Extraction disagrees with the human labels; show it, then improve it
**What happened.** The validation on the 9 curated papers: config mapping **6/9**, direction **4/9**, setting **3/9**. That means auto-extracted claims (all T2/T3, and all mechanisms) are noisy, and they drive soft-rejects and gap scores.

**Fix.**
1. **Show it prominently** on the dashboard and in the README next to every soft-reject and gap score: "auto-extracted claims agree with human labels on direction 4/9 and setting 3/9 in our validation".
2. **Inspect the 9 rows** (`validation.rows`) and the extraction outputs for c02, c03, c05, c06, c13. Likely causes:
   - "context" vs "improves" confusion (theory or difficulty statements read as results);
   - scale/task left as specific when the abstract is general, or vice versa.
3. **Tighten `prompts/extract.md`** with 2–3 counter-examples drawn from those rows (theory statement → `context`; "comparable performance" → `no_worse`; no scale stated → `unspecified`).
4. Re-run the validation (cheap: 9 papers) before the full search. Target: direction ≥ 7/9, setting ≥ 6/9. Whatever the result, report it.

**Accept.** The validation numbers are shown on the dashboard; the prompt change is recorded with before/after numbers.

## P1-5. Small checks before recording
- **Lion:** confirm the option uses the paper's scaling (lr ÷ 5, weight decay × 5) or is excluded. Otherwise a likely "harmful" result is an artifact of our setup.
- **Combination gap `relu2 + rmsnorm`:** both share only `computational_efficiency`, so complementarity is 0.5 (likely redundant). That's fine to run; its card should say so.
- **Mechanism graph:** 2 shared mechanism nodes (want ≥ 3). After P0-1 fills the direction searches, check again. Record anyway if it stays at 2, and say so.

---

## Re-run procedure
```bash
uv run pytest -q && uv run ruff check .
# wait ~15–30 min after the last arXiv-heavy command
make record SESSION=explore2 NICHE=niche.yaml     # resumes from cache; aborts on >10% failed requests or an unstable gate
```
Presentable if:
- search had < 10% failed requests and `papers_from_directions` ≥ 50;
- the gate was all STABLE;
- the extraction validation is shown;
- ≥ 3 gap hypotheses were explored and ≥ 1 confirmed, with shrinkage and prediction outcomes;
- verify, rederive and replay all pass.
