# DEEP_READ — Full-text reading of the closest papers, and hypothesis updates from it

**Goal.** For the hypotheses the lab is about to test, read the full text (arXiv HTML) of the few papers *closest to them in the knowledge graph*. Extract checkable facts the abstract doesn't give, and use them to update coverage, verdicts, hypotheses and the queue before any compute is spent.

**Why.** Abstract-only extraction is the weakest link, and the README says so:
- setting agreement with human labels is 3/9;
- Lion ran with our guessed hyperparameters;
- the mechanism graph had only 2 shared nodes;
- limitations / future-work sections, the best source of gaps, were never seen.

**Scope.** A new stage between the frozen search corpus and the session. Old bundles (golden, explore1, explore2) are untouched; new sessions record `protocol.deep_read_version = 1`.

**Invariant (unchanged).** LLMs extract facts (each with a verbatim quote) and rewrite hypothesis text. Code verifies quotes, decides which facts override which, recomputes coverage, verdicts and gap scores, and orders the queue.

**One session commits.** Build on a branch (`deep-read`). Only one AI session pushes to it; merge to `main` after a verified recording.

---

## 1. Which papers: the closest nodes in the graph
Run after the gap graph is built from the frozen corpus (the initial queue). Deterministic, no LLM.

For each of the top **N = 8** queued hypotheses (by `gap_score`), plus any contradiction gap and any finalist from a previous session if re-running, collect candidate papers with a **proximity score**:

| Relation to the hypothesis | Base weight |
|---|---|
| Paper of a claim on the hypothesis's explanation path (`gap.path`) | 1.0 |
| Paper of a claim the prior-art gate listed as same comparison or partial overlap | 1.0 |
| Paper with a claim on the **same method node** (any setting) | 0.7 |
| Paper with a claim on a **shared mechanism node** of the hypothesis | 0.5 |
| Paper found by the scout direction that generated the hypothesis | 0.3 |

`proximity = base × tier_weight (T1 1.0, T2 0.8, T3 0.6) × setting_similarity`, summed over all relations a paper has to that hypothesis.

- Keep the top **3 papers per hypothesis**, dedupe across hypotheses, cap at **20 papers** total. Ties: higher citation count, then arXiv id.
- Store the selection table (hypothesis → papers, with each paper's relations and score) as `corpus/deep/selection.json`. The dashboard shows it ("why this paper was read").

## 2. Fetching and parsing (code)
- **Source, in order:**
  1. `https://arxiv.org/html/<id>` (native HTML, most papers since late 2023);
  2. `https://ar5iv.labs.arxiv.org/html/<id>` (older papers);
  3. otherwise mark `full_text: unavailable`, do not fall back to PDF parsing, and keep the abstract-only claims.
- Same HTTP layer as search: one request in flight, ≥ 5 s interval, backoff on 429/503, URL-keyed cache under `work/cache/html/` (git-ignored). Record the arXiv version actually served.
- **Parse** with BeautifulSoup (`html.parser`; add `beautifulsoup4` to dependencies). Map section headings to canonical sections:
  - `abstract`, `introduction`, `method`;
  - `experimental_setup` (setup / experiments / implementation details / hyperparameters / training details);
  - `results`, `limitations`;
  - `conclusion` (conclusion / future work / discussion);
  - `appendix_setup` (appendix sections whose heading mentions hyperparameters / training / setup).

  Drop references, figures and math markup; keep table text (hyperparameter tables matter).
- **Store in the bundle only:** URL, arXiv version, SHA256 of the fetched HTML, section names with character lengths, and the quoted spans extracted below. **Never the full text** (licensing). The raw HTML stays in the local cache.
- `make verify-deep SESSION=…` (needs network): re-fetches the same version, checks the SHA256 matches, and re-checks every stored quote against it. Anyone can verify the quotes without us redistributing the papers.

## 3. Extraction (LLM, recorded, one call per paper)
Input: the relevant sections, trimmed to a budget (default 12k tokens; priority `experimental_setup` > `appendix_setup` > `results` > `limitations` > `conclusion` > `method`), plus the paper's existing abstract claims and the testbed's allowed fields.

Output schema `DeepFindings`; **every item carries `quote` + `section`**, and code drops any item whose quote is not verbatim in that section:

| Field | Content | Used for |
|---|---|---|
| `setting` | model family, task, **parameter count**, dataset, training steps/tokens, batch size | override abstract setting fields → coverage |
| `recipes` | per method: hyperparameters the paper used or recommends (lr, weight decay, warmup, betas, schedule), each as stated | method recipes (§4.2) |
| `results` | per comparison: direction (improves / no_worse / worse) and the setting it was measured in. Numbers may appear inside the quote but are **never used as our measurements** | claim direction refinement |
| `mechanisms` | reason given for an effect, as one of the 12 mechanism categories + quote | more mechanism edges in the graph |
| `limitations` | limitation or future-work sentences about a method | stated gaps (§4.4) |
| `small_scale_evidence` | any experiment at ≤ 10M parameters or character-level LM | tighten / coverage (§4.3) |

Use `unspecified` when the paper doesn't say. Never fill from memory.

## 4. How findings update the lab (code decides)

### 4.1 Settings and coverage
- For each claim of a deep-read paper, full-text setting facts **override** abstract-extracted fields. Mark the claim `setting_source: full_text`.
- **Recompute coverage, tiers, gate verdicts and gap scores** with the same rules as before.
- Every changed outcome is recorded as a `deep_read_update` decision with before/after and the quote. Examples: "coverage partial → none: experiments use 1.3B parameters, quote …"; "gap score 0.18 → 0.31".

### 4.2 Method recipes (fixes the Lion problem)
- Add two typed fields to `ExperimentConfig`: `lr_mult` ∈ {0.1, 0.2, 0.33, 0.5, 1, 2} and `wd_mult` ∈ {1, 3, 5, 10}, default 1 (left out of the hash at defaults, as before).
- If a deep-read recipe for a method in a queued hypothesis gives a learning rate or weight decay relative to AdamW (or absolute values convertible against the paper's own AdamW baseline), code maps it to the nearest allowed multiplier. The hypothesis variant then becomes `method + paper recipe`, labeled "recipe from arXiv:… (quote)".
- **Not a retune:** one recipe per method, taken from the paper, decided before any run. No sweeps.
- If no recipe is found, the method keeps today's default and its card says "no recipe found in full text".

### 4.3 Tighten with full text
- The tighten pass also sees `small_scale_evidence`. If a deep-read paper already reports this exact comparison at ≤ 10M parameters or for character-level LM, then:
  - **T1** → reject;
  - **T2/T3** → soft-reject;

  both with the quote. The hypothesis is still narrowable once, as before.
- If the full text reports the comparison only at larger scale, the card says so explicitly ("tested at 125M–1.3B only").

### 4.4 Stated gaps (new gap type)
- A `limitations` item about a method in the testbed becomes a candidate **`stated_gap`**:
  - the Scientist proposes a typed delta that tests the stated limitation;
  - code validates it (allowed fields, ≤ 2 fields);
  - plausibility = tier weight × setting similarity × 1.0 if the sentence names a concrete untested condition, 0.5 otherwise.
- `novelty_type: author_stated`. It enters the same queue and scoring (§A3 of GRAPH_GAPS) and the same gate.

### 4.5 Hypothesis revision (LLM writes, code checks)
For each queued hypothesis whose inputs changed (coverage, recipe, small-scale evidence, new mechanism), the Scientist writes **revision r1**:
- updated statement, mechanism and falsification rule, citing finding ids;
- the config delta may change only via §4.2 recipes or a §4.3 narrowing.

**Predicted direction:** code recomputes it from the updated claims (same rule as gap prediction, full-text claims taking precedence over abstract claims for the same comparison). The LLM does not set it.

Store r0 → r1 as a diff on the hypothesis card: "what the full text changed".

## 5. Pipeline and commands
```
make search NICHE=…                 # unchanged (or reuse a frozen corpus)
make deep-read CORPUS=work/corpus-X # select → fetch → parse → extract → freeze into work/corpus-X/deep/
uv run python -m noesis_lab lit-dryrun --corpus work/corpus-X --repeat 3   # gate stability AFTER the deep read
make record SESSION=explore3 CORPUS=work/corpus-X   # session reads corpus/deep/ (frozen), never the network
```
- `make record` runs `deep-read` automatically when `deep_read.enabled: true`.
- **Replay** reads only `corpus/deep/` (no network). The state digest includes deep-read decisions.

**Config** (`config.yaml`):
```yaml
deep_read:
  enabled: true
  top_hypotheses: 8
  papers_per_hypothesis: 3
  max_papers: 20
  section_token_budget: 12000
  small_scale_max_params: 10000000
  max_llm_cost_usd: 4.0
```

## 6. Validation (cheap, do first)
- Run the deep read on the **9 curated papers** that have human labels, and recompute agreement (config mapping / direction / setting).
- **Target:** setting ≥ 6/9 (from 3/9 with abstracts).
  - *Post-run note:* this target was mis-specified. The curated labels follow an abstract-only criterion (`data/claims.json` note), so full text can legitimately disagree with them, and 2 of the 9 labels never match a change (ceiling 7/9). Live result: 5/9 (5/7 matched), abstract-only 4/9. The labels were left as they are.
- Report before/after on the dashboard and in the README whatever the result. If setting agreement does not improve, say so and do not let full-text settings override curated (T1) labels.
- **Spot-check 5 recipes by hand** against the paper (especially Lion). Record the check in the README.

## 7. Dashboard
- **"Deep read" panel:** the selection table (hypothesis → papers, relations, proximity), and per paper: sections found, findings with quotes and section names, URL + version + SHA256.
- **"What the full text changed":** every `deep_read_update` (coverage / verdict / gap-score / recipe / stated gap), each with its quote.
- **Hypothesis card:** r0 → r1 diff, predicted direction before/after, "recipe from arXiv:…" badge.

## 8. Tests (CPU, no network)
- HTML fixtures (native arXiv HTML + ar5iv) → section mapping, including a hyperparameter table and an appendix setup section.
- Quote check against section text: non-verbatim items are dropped and logged.
- Selection: deterministic ranking and caps on a toy graph.
- Overrides: a full-text setting flips coverage and changes the gate verdict deterministically; a curated T1 label is never overridden.
- Recipe mapping to the nearest multiplier; no recipe → default; a recipe never changes after the first run.
- Stated gap → typed delta → queue; an invalid delta is dropped.
- Revision: predicted direction comes from code; the LLM's text cannot change it.
- Replay of a deep-read session with the network disabled → identical digest; old bundles still verify, rederive and replay.

## 9. Build order and time (≈ 3–4 h, then one recording)
1. Selection + fetch/cache + section parsing + freeze (≈ 1 h).
2. `DeepFindings` extraction + quote check + validation on the curated 9 (≈ 1 h). **Stop here if setting agreement doesn't improve,** and report it.
3. Overrides → recompute coverage / verdicts / gaps; `lr_mult` / `wd_mult` + recipes; tighten with small-scale evidence (≈ 1 h).
4. Stated gaps + hypothesis revision r1 + dashboard panels (≈ 1 h).
5. `make record SESSION=explore3 CORPUS=work/corpus-explore2` (reuses the frozen search; ~35–45 min with the deep read). Then verify, rederive, replay; README update.

**Cost:** about 20 papers × ~12k input tokens + extraction output ≈ $1.5–3, plus the session.

## 10. What we will be able to claim (README wording, only if measured)
- "Before testing, the lab read the full text of the N papers closest to its top hypotheses. It changed K coverage verdicts, supplied method recipes for M methods, and added S author-stated gaps. Every change cites a verbatim passage with the paper's version and hash."
- "Full-text setting agreement with human labels: X/9 (abstract-only: 3/9)."
