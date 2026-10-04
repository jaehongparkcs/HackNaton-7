# LIT_SEARCH — Niche definition, live literature search and the literature map

**Goal.** The PI defines a niche. The lab searches the live literature for it, extracts checkable claims, maps what is already covered, and uses that map to (a) block or deprioritize ideas that overlap existing work and (b) choose what to run next.

**Pitch line.** *Live literature search that decides what gets run.*

Builds on FIXES.md (do P0-1 step budget and P0-2 rule-after-every-result first; this file replaces P0-4's hand-curated coverage for new papers and feeds P0-3's candidate queue).

## Rules this feature must keep
1. **APIs find papers; LLMs never recall them.** Every paper comes from an API response with a real ID and a verbatim abstract. No paper may originate from LLM memory or LLM web browsing.
2. **LLMs extract facts; code checks and decides.** Claude extracts fields from an abstract. Code verifies the quote is verbatim, maps methods onto the allowed config space, computes coverage by a written rule, and sets every verdict, tier and priority.
3. **Search live, then freeze.** Search runs live when a session is recorded. Everything it returned is frozen into `results/<session>/corpus/`, and replay uses only the frozen copy.
4. **Never claim novelty.** The strongest statement is "not found in N papers retrieved on <date> for queries Q1..Qk".
5. **Novelty errors lean toward "covered"; evidence errors lean toward "inconclusive".** If unsure whether a paper overlaps our idea, treat the idea as covered (we avoid rediscovery). If unsure whether a paper's setting matches ours, never say our result "contradicts" it.

---

## 1. Niche definition (PI input)

### 1.1 `NicheSpec` (new schema; stored in `inputs.json` and the notebook)
| Field | Example | Used for |
|---|---|---|
| `title` | "Training efficiency of tiny character-level Transformers" | UI, report |
| `description` | 2–4 sentences in the PI's words | Query planning prompt |
| `include_terms` | ["layer normalization", "activation functions", "positional encoding", "optimizer", "learning-rate schedule"] | Query planning + relevance scoring |
| `exclude_terms` | ["vision transformer", "speech", "reinforcement learning"] | Hard filter (code) |
| `arxiv_categories` | ["cs.LG", "cs.CL", "cs.NE"] | arXiv query filter |
| `date_from` / `date_to` | 2017-01-01 / today | API filter |
| `seed_papers` | ["2002.04745", "1910.07467"] | Always fetched; citation expansion starts here |
| `max_papers` | 80 | Budget |
| `testbed` | "tiny_char_lm" (only option today) | Which config space candidates must map to |

### 1.2 Where it's entered
- **File:** `niche.yaml` at repo root (committed default = the current niche). The CLI reads `--niche niche.yaml`.
- **Dashboard:** sidebar form "Define the niche" that writes a new `niche_<slug>.yaml`, then shows the exact command to run (`make search NICHE=...` / `make golden NICHE=...`). The dashboard does not start training itself.
- The niche is a **PI decision** and is labeled that way in the UI ("chosen by the PI"). The lab chooses experiments, not the niche.

### 1.3 Niche vs. testbed
The niche guides the search. Only ideas that map onto the testbed's allowed config changes can be **run**. A niche outside the testbed (e.g. "diffusion samplers") still produces a map and an overlap analysis, but the queue shows "no runnable candidates in this testbed". Say this plainly in the UI instead of failing.

---

## 2. Pipeline

```
NicheSpec ─► (LLM) query plan ─► (API) retrieve ─► (code) dedupe + filter + rank
          ─► (LLM) extract fields per abstract ─► (code) verify quote, map to config, compute coverage, assign tier
          ─► freeze corpus into bundle ─► coverage grid + paper map ─► candidate queue (FIXES P0-3) ─► gate ─► runs
```

### 2.1 Query planning (LLM, recorded)
- Input: `NicheSpec` + the list of allowed config changes (field=value pairs) + `TESTBED_DESCRIPTION`.
- Output schema `QueryPlan`: 3–5 `dimensions`. Each has `precise` (term-level) and `broad` (area-level) queries. Max 12 queries total, enforced by code.
- Code appends deterministic queries: one per allowed config change (e.g. `"RMSNorm" AND "language model"`), so every runnable change is always searched even if the LLM forgets it.

### 2.2 Retrieval (code)
- **arXiv API** (`export.arxiv.org/api/query`), filtered by `arxiv_categories` and dates. Respect ~1 request / 3 s; cache every raw response to `corpus/raw/`.
- **OpenAlex** (optional, for publication venue, peer-review status and citation counts; free, high rate limits). Send a team-owned contact email from the `OPENALEX_MAILTO` env var; never a personal address hard-coded in the repo.
- **Seed expansion** (optional, time permitting): 1 hop of references/citations of `seed_papers` via OpenAlex, capped.
- Skip Semantic Scholar without an API key (shared rate limits will fail mid-demo).

### 2.3 Dedupe, filter, rank (code, deterministic)
- Dedupe by arXiv ID, then DOI, then normalized title.
- Drop papers matching `exclude_terms` in title/abstract. Log every drop with its reason.
- `relevance = 0.6 × TF-IDF cosine(abstract, niche description + include_terms) + 0.2 × log-citations (normalized) + 0.2 × recency`. Keep the top `max_papers`. Ties broken by arXiv ID.
- Store the ranking components per paper, so the dashboard can show why a paper made the cut.

### 2.4 Extraction (LLM, recorded, batched 5 abstracts per call)
Output schema `ExtractedClaim` (0–3 per paper):
| Field | Notes |
|---|---|
| `method` | free text |
| `config_change` | must be one of the allowed field=value pairs, or `null` (context only). Code rejects anything else. |
| `claim` | one-sentence paraphrase |
| `source_span` | must be a verbatim substring of the abstract; **code drops the claim if not** |
| `direction` | improves / no_worse / worse / context |
| `setting.model_family` | transformer / rnn / cnn / mlp / general / unspecified |
| `setting.task` | language_modeling / char_language_modeling / translation / classification / vision / speech / general / unspecified |
| `setting.scale` | tiny (<10M params) / small (<100M) / medium (<1B) / large / unspecified |
| `setting.evidence` | empirical / theoretical / survey |

The LLM only transcribes what the abstract says. "unspecified" is always allowed and is the correct answer when the abstract doesn't say.

### 2.5 Coverage rule (code; one rule for every paper, shown verbatim in the UI)
Our setting: `transformer`, `char_language_modeling`, `tiny`, step-budgeted training.

| Coverage | Condition |
|---|---|
| **covers** | model_family ∈ {transformer, general} AND task ∈ {char_language_modeling, language_modeling, general} AND scale ∈ {tiny, small} |
| **partially covers** | model_family ∈ {transformer, general} AND task ∈ {char_language_modeling, language_modeling, general} AND scale ∈ {medium, large, unspecified} |
| **does not cover** | anything else |

How it is used (Rule 5):
- **For overlap/novelty:** `covers` OR `partially covers` → the idea counts as **covered**.
- **For evidence relations** (agrees/contradicts): only `covers` counts. Otherwise the relation is `consistent_in_our_setting` / `not_reproduced_in_our_setting` / `inconclusive` (FIXES P1-5).

### 2.6 Trust tiers: unverified papers still count for overlap
Two separate questions, never merged:
- **Did our pipeline verify the quote?** (extraction check)
- **Was the paper peer-reviewed?** (venue status)

| Tier | Meaning | Counts for overlap / prior art? | Can hard-reject an idea? | Can change our priors / queue order? |
|---|---|---|---|---|
| **T1 curated** | Claim checked by a human (current 13 claims) | yes | **yes** | yes |
| **T2 verified-auto** | Auto-extracted; quote verbatim; published at a venue (OpenAlex) | yes | soft-reject (see below) | yes |
| **T3 preprint** | Auto-extracted; quote verbatim; arXiv only, not (yet) peer-reviewed | **yes** | soft-reject | yes, flagged "preprint" |
| **T4 unextracted** | Paper retrieved and relevant, but no claim passed the quote check | **yes, via text similarity** (below) | no; flags for PI review | no |

- **Recent and unverified work still blocks rediscovery.** T3 preprints count as covering a topic exactly like T2: being unreviewed does not make a topic uncovered. This also covers papers newer than the LLM's training data, because the LLM judges only the text it's given, never its memory.
- **T4 catches important papers whose claims we couldn't extract cleanly.** For every candidate, compute TF-IDF cosine between the hypothesis text and each T4 abstract. Above a threshold (start at 0.35; tune on the curated set), flag "possible overlap: unextracted paper <id>". The candidate waits for the PI to dismiss or confirm. It is never silently treated as untested.
- **Gate outcomes:**
  - `known_in_corpus` from T1 → **reject** (as today).
  - From T2/T3 → **soft-reject**: the candidate moves to the end of the queue, labeled "covered by auto-extracted / preprint claim, PI may override". Show the quote and tier.
  - T4 similarity hit → **hold for PI review**.
  - The demo's Fixture A must rest on a T1 claim.
- Every displayed claim shows its tier badge, the paper's date, and "peer-reviewed / preprint".

### 2.7 Freeze
`results/<session>/corpus/` contains:
- `niche.yaml`
- `query_plan.json`
- `raw/` (API responses)
- `papers.json` (with ranking components)
- `claims.json` (with tier, coverage, extraction event IDs)
- `dropped.json` (filtered/deduped/failed-quote, with reasons)

The manifest hashes all of it. Replay and rederive read only this folder: no network.

---

## 3. The literature map (dashboard)

### 3.1 Coverage grid (the functional view; build first)
- **Rows:** allowed config changes (rmsnorm, post-LN, swiglu, relu2, rope, constant schedule, sgd_momentum, dropout 0.1, …) + one row "outside testbed" grouping methods with `config_change = null`.
- **Columns:** setting buckets: covers our setting / partial / other settings.
- **Cell:** count of claims by tier (e.g. `T1·1 T3·2`) and the dominant direction. Click → list of quotes with paper links.
- **Right-hand columns:** our gate verdict, our measured result (branch + mean Δ in noise SDs), evidence relation.
- **Empty "covers" cells with "improves" claims elsewhere are the gaps.** Highlight them; they are exactly what the candidate queue (FIXES P0-3) tries first. Show the queue order next to the grid.

### 3.2 Paper map (the overview)
- 2D scatter of retrieved papers: TF-IDF vectors → PCA (NumPy, no new dependencies). Colored by dimension (norm / activation / positional / optimizer / schedule / other), shape by tier, size by relevance.
- Our candidates are drawn as stars at the centroid of their supporting claims' papers, colored by outcome: rejected-prior-art / soft-rejected / held / screened (branch).
- Caption: "Positions show text similarity only; use the coverage grid for decisions."

### 3.3 Search transparency panel
Show:
- the niche (labeled PI-chosen);
- the query plan (LLM-proposed + deterministic queries);
- papers retrieved / deduped / filtered / kept, with reasons;
- extraction yield (claims kept vs. dropped for non-verbatim quotes);
- the coverage rule text;
- the tier counts;
- the date of the search.

---

## 4. How it plugs into the loop
1. **Session start:** load the niche → search → freeze corpus (or load a frozen corpus in replay).
2. **Candidate queue (FIXES P0-3):** priority uses the frozen claims:
   - (a) exclude candidates covered by T1 → rejected;
   - (b) push candidates covered by T2/T3 to the end (soft-reject);
   - (c) hold T4-flagged candidates for review;
   - (d) among the rest, prefer changes with "improves" claims in other settings (transfer), then more supporting claims, then alphabetical.
3. **Gate:** the Literature agent sees the retrieved claims for a candidate (now from the frozen corpus, all tiers) and answers only "same comparison?" (FIXES P0-4). Code applies tier + coverage to produce the verdict.
4. **Write-back:** each screening result is stored as derived evidence next to the claims it relates to, and appears in its coverage-grid row.

---

## 5. Validation (cheap, do it)
- **Extraction check against the curated set:** run extraction on the 10 curated papers and compare `config_change`, `direction` and setting fields to the human labels. Show agreement on the dashboard (e.g. "config mapping 12/13, setting 10/13"). If agreement is poor, demote auto claims to T4 behavior for the demo and say so.
- **Gate stability:** run the gate twice on Fixture A, Fixture B and the top 5 queue candidates. Verdicts must match.
- **Leakage check:** fixture verdicts must not change when the corpus grows. If Fixture B becomes "covered" by a newly found paper, that is a real finding: report it and pick a new Fixture B rather than hiding it.

---

## 6. Budgets and failure handling
- `max_queries: 12`, `max_papers: 80`, extraction batch 5 abstracts per call (~16 calls). The Anthropic cost estimate is about $1–2 at the current pricing in `config.yaml`. Put all of these in `config.yaml` under `lit_search:`.
- API failure / timeout → retry once, then continue with what was retrieved and record `search_degraded` with the reason. If fewer than 10 papers are retrieved, fall back to the curated snapshot and label the session "curated snapshot only".
- Record the session early in the day; never depend on live APIs during the final recording window.

---

## 7. Schemas and code layout (additions)
- `schemas.py`: `NicheSpec`, `QueryPlan`, `RetrievedPaper` (adds `source`, `venue`, `peer_reviewed`, `published`, `relevance_components`, `tier`), `ExtractedClaim`, `SettingFields`, `Coverage` (Literal covers/partial/none), `Tier` (Literal T1–T4).
- `prior_art/search/`:
  - `plan.py` (LLM query plan)
  - `arxiv.py`, `openalex.py` (retrieval + raw cache)
  - `rank.py` (dedupe/filter/rank; pure)
  - `extract.py` (LLM + quote check)
  - `coverage.py` (rule; pure)
  - `corpus.py` (freeze/load)
- `stats.py`: `candidate_queue` takes the frozen claims (tiers + coverage).
- `prompts/query_plan.md`, `prompts/extract.md`.
- `app.py`: niche form, coverage grid, paper map, search transparency panel.
- `Makefile`: `make search NICHE=niche.yaml` (search + freeze only, no training), and `golden` accepts `NICHE=`.

## 8. Tests (CPU, no network)
- `rank.py` and `coverage.py`: pure-function unit tests, including every coverage-table row.
- Quote check: a non-verbatim span is dropped and logged in `dropped.json`.
- `config_change` outside the allowlist → dropped to `null`.
- Tier assignment and gate outcomes (reject / soft-reject / hold) for each tier.
- Replay from a frozen corpus with the network disabled → identical state digest.
- Search clients tested against recorded raw responses (fixtures under `tests/data/`), never live.

---

## 9. Build order and time (one person ≈ 3–4 h; others do FIXES P0 in parallel)
1. `NicheSpec` + `niche.yaml` + retrieval from arXiv with raw cache + freeze/load (≈ 1 h).
2. Rank/dedupe/filter + extraction with quote check + coverage rule + tiers (≈ 1–1.5 h).
3. Wire into the candidate queue and the gate (≈ 30 min).
4. Coverage grid → search transparency panel → paper map (≈ 1 h; drop the paper map first if late).
5. Validation runs (§5) and one recorded session (≈ 30 min).

**Cut first if late:** OpenAlex (then T2/T3 can't be told apart: label all auto claims "auto-extracted, review status unknown" and treat them as T3), seed expansion, paper map, dashboard niche form (keep `niche.yaml`).

## 10. Demo beats this enables
1. "PI defines the niche" → show `niche.yaml` / the form.
2. "Lab searches live": query plan, 80 papers, extraction yield, tiers (incl. recent preprints counted as covered).
3. Coverage grid: the gaps light up; the queue is ordered from them.
4. Fixture A rejected on a T1 quote; a generated candidate soft-rejected by a recent preprint ("we don't re-run what was just posted").
5. Top gap gets run → result written back into its grid row.
