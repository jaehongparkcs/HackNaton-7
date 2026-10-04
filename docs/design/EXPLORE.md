# EXPLORE — Scout → targeted search → gap graph → hypotheses

**Goal.** Replace "fixed menu of changes ordered by the niche corpus" with:
1. a quick LLM **scout** of research directions;
2. a **targeted literature search per direction**;
3. a **knowledge graph of gaps**, with every paper tied to each direction;
4. **hypotheses generated from those gaps**, each tightened by its own targeted search;
5. the existing queue → run → write-back loop.

Builds on LIT_SEARCH.md (retrieval, extraction, quote check, tiers, coverage) and FIXES.md (queue, rule after every result). Reuse those modules; do not fork them.

## Rules (in addition to LIT_SEARCH.md's)
1. **The scout produces search directions, never evidence.** Anything the LLM names, including paper titles it "remembers", is used only as a search query. Only retrieved, quote-verified claims enter the graph. Unmatched remembered titles are logged as "not found" and dropped.
2. **Gap status and novelty type are computed by code** from the graph, by the rules below. The LLM never labels a gap or a novelty type.
3. **Nothing is called "novel".** Every hypothesis shows its novelty type and "not found in N papers retrieved for queries Q1..Qk".
4. **Hypotheses must compile to the experiment language** (typed building blocks, §4) or they are shown on the map as `outside_testbed` and not run.

---

## 1. Scout (LLM, recorded, one call)
Input: `NicheSpec` + testbed description + the list of building blocks (§4).
Output schema `ScoutOutput`: 8–15 `Direction`s, each with:

| Field | Notes |
|---|---|
| `direction_id` | assigned by code (`dir_01`…) |
| `title` | short |
| `idea` | 1–2 sentences |
| `mechanism` | why it might matter in this niche |
| `search_terms` | 3–6 terms |
| `possibly_related_titles` | 0–3 titles the LLM recalls. **Search hints only**: searched by title on arXiv; matched → paper added with its real ID; unmatched → logged `not_found` |
| `building_blocks` | list of block names it would touch, or `[]` → outside testbed |

Code adds one deterministic direction per building block that the scout did not mention, so the map always covers what the testbed can run.

## 2. Targeted search per direction (code)
- 2–3 arXiv queries per direction, built from `search_terms` (+ niche categories/dates). Cap 15 papers per direction, ~120 total after dedupe. Respect arXiv's rate limit.
- Record provenance: `paper —found_by→ direction` (a paper can belong to several directions).
- Then the existing steps, unchanged: dedupe → exclude filter → extract claims (quote check) → coverage rule → tiers (T1–T4).
- **Link claims to directions:** a claim belongs to a direction if its paper was found by that direction AND (the claim's method maps to one of the direction's building blocks OR TF-IDF similarity between the claim and the direction text ≥ threshold, default 0.25, tuned on the curated set).
- Freeze everything into the corpus folder (scout output, per-direction queries, provenance), as in LIT_SEARCH §2.7.

## 3. Gap graph (code)
**Nodes:** Direction · Paper · Claim · BuildingBlock · SettingBucket (covers / partial / other) · Hypothesis · OurResult.

**Edges:**
- `direction→paper (found_by)`
- `paper→claim (states)`
- `claim→block (about)`
- `claim→setting (in)`
- `direction→hypothesis (generated)`
- `hypothesis→claim (grounded_in / overlaps)`
- `hypothesis→result (tested_by)`
- `result→claim (consistent / not_reproduced / inconclusive)`

**Direction gap status (pure function, rule text shown in the UI):**

| Status | Rule |
|---|---|
| `covered` | a T1/T2 claim with coverage `covers` reports this direction's method in our setting |
| `partial` | covered only by `partial`-coverage claims or only by T3 preprints |
| `contested` | ≥2 linked claims with opposite `direction` (improves vs worse/no_worse) |
| `open` | linked claims exist for the method elsewhere, none covers our setting |
| `unexplored` | no linked claims at all (only papers, or nothing found); show the paper count |
| `outside_testbed` | `building_blocks == []` (any of the above can co-apply; show both) |

## 4. Experiment language: building blocks (typed, allowlisted)
Extend `ExperimentConfig` only with options that are implemented, tested and cheap:
- **Existing single changes:** norm, norm_position, activation, pos_encoding, optimizer, schedule, dropout.
- **Combinations:** any 2 single changes together (config delta with 2 fields). The validator rejects pairs that touch the same field.
- **Parameterized variants (optional, if time):** `warmup_frac` ∈ {0, 0.02, 0.05, 0.1}, `cosine_final_frac` ∈ {0.0, 0.1, 0.3}, `qk_norm: bool`.

No LLM-written model code. A new mechanism that needs code is `outside_testbed`.

## 5. Hypotheses from gaps (LLM writes, code checks)
- For each direction with status `open`, `contested` or `unexplored` that is runnable: the Scientist writes 1–2 hypotheses as **typed config deltas** (1–2 fields). Each has: statement, mechanism, prediction, falsification rule, and the `grounded_in` claim IDs (must exist in the graph; validated by code).
- **Novelty type (code):**

| Type | Rule |
|---|---|
| `transfer` | 1 field; that method has claims elsewhere; none covers our setting |
| `combination` | 2 fields; each has claims; **no linked claim mentions both** (TF-IDF + config mapping) |
| `resolution` | the direction is `contested` |
| `new_mechanism` | not expressible as a delta → `outside_testbed`, map only |

- **Tighten pass (per hypothesis):**
  1. One targeted arXiv search built from the hypothesis's own text (2 queries, cap 10 papers), then extraction and the quote check.
  2. Gate as today (LLM: "same comparison?"; code: tier + coverage → reject / soft-reject / hold / pass).
  3. If it overlaps: the Scientist gets **one** revision to narrow it to the untested part (still a typed delta). Code re-checks it. A second overlap → rejected, with the overlapping quote shown.
- Dedupe hypotheses by config delta. The same delta from two directions → one hypothesis linked to both.

## 6. Queue order (code, deterministic)
- **Open hypotheses first, by novelty type:** `resolution` > `combination` > `transfer`. Resolving disagreements and untested combinations are more informative than plain transfers.
- **Within a type:** more `grounded_in` claims with "improves" first, then fewer overlapping papers, then alphabetical.
- Soft-rejected / held as in LIT_SEARCH §2.6.
- Fixture A stays as the guaranteed prior-art demo. Fixture B can be replaced by the top generated hypothesis.

## 7. Dashboard
- **Gap map (overview):** directions as large nodes colored by gap status (covered grey, partial amber, contested purple, open green, unexplored outline, outside-testbed dashed), each connected to its papers. Shared papers connect directions. Built from stored links with `st.graphviz_chart`.
- **Direction view:** the selected direction's subgraph: papers (tier badge, date, peer-reviewed/preprint) → claims (verbatim quote) → building blocks → our hypotheses (novelty type) → our results.
- **Scout panel:** the scout's raw directions, which remembered titles were found or not found, and the queries per direction with paper counts ("how hard we looked").
- **Hypothesis card:** novelty type badge, `grounded_in` quotes, tighten-pass result (passed / narrowed / rejected, with quote), then the existing screening result.

## 8. Tests (CPU, mocked LLM, recorded API fixtures)
- Gap-status rules, novelty-type rules, claim↔direction linking: pure-function unit tests for each table row.
- Remembered titles: matched → paper added with its real ID; unmatched → logged, never cited.
- Combination validator: same-field pairs rejected; pairs outside the allowlist rejected.
- Tighten pass: overlap → one revision → re-check; second overlap → rejected.
- Replay from frozen corpus with the network disabled → identical digest.

## 9. Build order (≈ 3–4 h)
1. Scout schema + prompt + title-hint search; per-direction retrieval with provenance (≈ 1 h).
2. Claim↔direction linking + gap-status rules + combination building blocks (≈ 1 h).
3. Hypotheses from gaps + novelty type + tighten pass + queue order (≈ 1 h).
4. Gap map + direction view + hypothesis card badges (≈ 45 min).
5. Re-record one session; `make verify rederive replay`; update the README claims table.

**Cut first if late:** parameterized variants, the scout panel's title-hint details, the overview map (keep the per-direction view).

## 10. What we claim (README wording)
"The lab maps research directions, finds where the retrieved literature is silent or disagrees for our setting, and tests those gaps. Each hypothesis is labeled transfer, combination or resolution, and is 'not found in N retrieved papers', never 'novel'. New mechanisms that need new code are shown on the map but not run."
