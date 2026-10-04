# GRAPH_GAPS — Gaps from graph structure + an autoresearch-style search loop

Extends EXPLORE.md. Two parts:
- **A.** Find gaps from the structure of a knowledge graph, by scoring formulas, not by an LLM.
- **B.** A search loop that borrows from existing auto-research systems, keeping our invariant.

Part A is the core (≈ 2–3 h). Part B is the extension (≈ 3–4 h). Build A first.

Invariant (unchanged): LLMs propose, extract and interpret. Code measures, scores, ranks and decides. Every gap and every hypothesis shows *why* it exists as a path in the graph.

---

## Recommended ideation: hybrid, one graph
- **Both sources feed one graph:**
  - (1) the niche search (LIT_SEARCH);
  - (2) the scout's directions with targeted searches (EXPLORE).

  The niche search gives depth in the area. The scout gives breadth from adjacent areas.
- **Gaps are computed on the merged graph** (Part A). Hypotheses are generated only for the top-scoring gaps.
- **The scout re-runs only on stagnation** (Part B, step 7), not every cycle.
- **Measure, don't assert:** `make compare-ideation` runs (niche-only) vs (scout-only) vs (hybrid) on the same niche. It reports hypotheses surviving the tighten pass, the novelty-type mix, and the targeted-search overlap rate. No training runs; LLM + search cost only.

---

## Part A — Mathematical gap finding

### A1. Graph schema (extends EXPLORE §3)
Add one extracted field per claim: **`mechanism`**, the intermediate concept the claim gives as the reason ("stabilizes gradient norm", "reduces attention entropy collapse", "improves token-position extrapolation").
- The extraction LLM must quote it from the abstract (verbatim span check) or leave it empty.
- Code normalizes mechanisms: lowercase, lemmatize, merge near-duplicates by TF-IDF cosine ≥ 0.6, keep a `mechanism_id`.

**Typed nodes:**
- **M** method / building block
- **K** mechanism
- **S** setting bucket (model family × task × scale)
- **O** outcome (val_loss ↓, stability, speed)
- **P** paper
- **C** claim

**Edges come from claims.** A claim produces a signed hyperedge (method, setting, outcome, sign ∈ {+, 0, −}), and optionally method→mechanism and mechanism→outcome edges. Our own results add edges tagged `derived`.

### A2. Gap types and their formulas (all pure functions; each returns a score + explanation path)

1. **Coverage gap (missing cell).** Build the tensor `T[m, s, o]` = number of claims (weighted by tier: T1 = 1.0, T2 = 0.8, T3 = 0.6) for method m, setting s, outcome o.
   - Gap if `T[m, s*, o*] = 0` for our setting s* and outcome o*, while `Σ_s T[m, s, o*] > 0` elsewhere.
   - **Plausibility** = how well m's effect transfers: the fraction of m's claims across settings with the same sign, times the similarity of those settings to s* (fixed similarity table over the setting fields).
   - Novelty type: `transfer`.

2. **Swanson ABC closure (hidden connection).** Literature-based discovery: if A→B and B→C are reported but A→C is not, then A→C is a candidate.
   - Here: method m → mechanism k (some claim) and mechanism k → outcome o* in setting s* (another claim, possibly about a different method), but no claim m → o* in s*.
   - Score = (tier-weighted support of m→k) × (support of k→o* in s*) × 1/(number of mechanisms on the path).
   - Explanation path: `m —claim c1→ k —claim c2→ o*`.
   - Novelty type: `transfer` (by mechanism).

3. **Link prediction on the method graph (combination gaps).** Project to a method–method graph: methods are linked if they share a mechanism or are both tested in the same claim/setting.
   - For unlinked method pairs (a, b) with **no claim mentioning both**, compute:
     - common neighbors `|N(a) ∩ N(b)|`;
     - Adamic–Adar `Σ_{z ∈ N(a)∩N(b)} 1/log|N(z)|`;
     - resource allocation `Σ 1/|N(z)|`.
   - Score = normalized Adamic–Adar × complementarity, where complementarity = 1 if a and b work through *different* mechanisms on the same outcome (likely additive) and 0.5 if they share one (likely redundant).
   - Explanation: the shared neighbors.
   - Novelty type: `combination`.

4. **Contradiction (signed edges).** The same (m, s-bucket, o) has claims with opposite signs.
   - Score = min(support+, support−) / max(...) (1 = evenly split) × similarity of those settings to ours.
   - Novelty type: `resolution`.

5. **Structural hole (bridging clusters).** Run community detection (greedy modularity, networkx) on the method–mechanism graph.
   - Method pairs from different communities, with a path of length 2 through a mechanism but no direct claim, are **bridges**.
   - Score = (1 − fraction of cross-community edges between the two communities) × path support.
   - Novelty type: `combination` (cross-area).

### A3. One gap score, one ranked list
`gap_score = plausibility × (1 − coverage_in_our_setting) × testability × evidence_quality`
- `coverage_in_our_setting` ∈ [0, 1] from the tensor (tier-weighted, capped).
- `testability` = 1 if expressible as a typed delta (1–2 fields), 0.3 if it needs a parameterized variant not built yet, 0 if outside the testbed (shown on the map, not queued).
- `evidence_quality` = mean tier weight along the explanation path. A path resting only on T3 preprints scores lower.

Report the components, not just the product. Ties: alphabetical by delta key.

### A4. Honesty constraints
- With ~100–150 papers, link-prediction scores are weak signals. Show them as "structural hint", never as probability. The prior-art gate and the measured result still decide.
- Every gap card shows: gap type, formula components, the explanation path with quotes, papers searched, and "not found in N papers for queries Q…".
- `make validate-gaps`: hold out 20% of claims, recompute, and report how many held-out claims fall in top-ranked gap cells (precision@k). Small and noisy, but an honest check that the scores aren't arbitrary. Show it on the dashboard.

### A5. Dashboard
- **Gap list:** ranked, with type badge (transfer / ABC / combination / resolution / bridge), component bars and testability.
- **Gap card:** the explanation path drawn as a small graph (graphviz) with quotes on the edges.
- **Overview map:** methods and mechanisms as nodes, communities colored, claim edges signed (green +, red −, grey 0), our results as thick edges, gaps as dashed edges with their score.

---

## Part B — Borrowed from existing auto-research systems (each mapped to our invariant)

| Borrowed from | Idea | Our version (code decides) |
|---|---|---|
| Kosmos (world model) | Shared structured memory updated every cycle | The knowledge graph *is* the world model: our results are written back as `derived` edges and gaps are recomputed each cycle |
| AI Scientist v2 (agentic tree search) | Tree of experiments; expand promising nodes | **Experiment tree:** children of a node = 1-field refinements or combinations with other promising nodes; expansion chosen by code (best-first on posterior mean improvement + exploration bonus) |
| AlphaEvolve / ShinkaEvolve (evolution, archive, novelty filter) | Population + archive of diverse good solutions | **Quality-diversity archive** (MAP-Elites style) keyed by graph cell (method family × mechanism). Keep the best measured candidate per cell; the novelty filter = graph distance to existing claims |
| Karpathy autoresearch (keep/revert ratchet, `program.md`) | Accept improvements, keep going; a research charter | **Promotion with statistics:** a candidate that stays promising on 5 paired seeds becomes the incumbent (its runs become the new noise floor; no extra compute). `niche.yaml` is our charter |
| Co-Scientist (generate, reflect, evolve, proximity) | Debate-driven refinement, deduplication | Critic before compute + one tighten revision + dedup by config delta. **No LLM tournament** (an LLM would rank) |
| MARs (claim graph, builder/skeptic, admission gate) | Claims with dependencies; skeptic; code gate | Hypotheses carry `grounded_in` claim edges; Critic = skeptic; admission = the pre-registered rule; failed claims go to limitations |
| AutoResearch at production scale (failure modes) | Stagnation, memory decay, metric fixation | **Stagnation redirect:** k cycles without a promising result → re-run the scout on the least-explored communities. Memory = graph. Metric fixation → report speed and stability as secondary metrics, never in the decision |
| Agent Laboratory (co-pilot) | Human checkpoints improve quality | PI overrides / holds in `niche.yaml`; optional pause before each expansion |

### B1. The cycle
1. **Recompute gaps** on the graph (Part A) → ranked gap list.
2. **Generate:** the Scientist writes typed hypotheses for the top gaps (EXPLORE §5), each `grounded_in` its path's claims.
3. **Tighten:** a targeted search per hypothesis → gate → at most one narrowing revision.
4. **Explore cheaply:** run surviving hypotheses on **1 seed** and place each in the archive cell. Reuse baseline runs.
5. **Select finalists (code):** the best per archive cell, then the top-k by single-seed improvement.
6. **Confirm rigorously:** finalists get the paired 3-seed screening → the pre-registered rule → extra seeds → promotion if they still hold.
7. **Write back** results as `derived` edges. Record the **shrinkage** (single-seed estimate vs. paired estimate) per finalist. **Stagnation check** → scout redirect if needed.
8. **Repeat** until the budget runs out.

### B2. What this demonstrates (headline metrics)
- **Shrinkage:** how much the single-seed (autoresearch-style) estimates shrink under paired re-testing. This is the survival-rate idea, measured inside the system.
- **Gap hit rate:** the fraction of tested gaps that came out promising, by gap type (do ABC / combination gaps pay off more than transfers?).
- **Ideation comparison:** hybrid vs niche-only vs scout-only (no training needed).

### B3. Budget at 2,000 steps (~35 s/run)
A 45-minute session ≈ 75 runs:
- noise floor 5;
- explore 2 cycles × 10 hypotheses × 1 seed = 20;
- confirm 4 finalists × 3 seeds = 12;
- extra seeds 2 × 2 = 4;
- promotion re-baseline 0 (reused);
- remaining ~30 for a third cycle.

Put these in `config.yaml` (`explore_seeds: 1`, `finalists_per_cycle`, `cycles`, `stagnation_k`).

---

## Tests
- Every gap formula and the tensor: pure-function unit tests on a hand-built toy graph with known answers. For example, a toy ABC triangle must be found; a contradiction pair must score 1 when evenly split.
- Mechanism normalization merges near-duplicates and keeps distinct ones.
- Archive keeps the best per cell; finalist selection is deterministic.
- Promotion reuses the candidate's runs as the new noise floor; the original baseline is never modified.
- Replay with the network disabled → identical digest.

## Build order
1. **(A, ≈ 2–3 h)** mechanism extraction + normalization → tensor + 5 gap formulas + gap score → gap list + gap card on the dashboard → hypotheses from top gaps → re-record.
2. **(B, ≈ 3–4 h)** explore-1-seed / confirm-paired cycle + archive + promotion + stagnation redirect + shrinkage metric → re-record.
3. `make compare-ideation`, `make validate-gaps`.

Keep `main` on the verified golden session; build on a branch; switch the submission only after `make verify rederive replay` passes on the new session.

## References for the README
- Literature-based discovery (Swanson's ABC model) and recent advances: arXiv:2506.12385
- Hypothesis generation over scientific knowledge graphs: ScienceDirect S0950705125003272
- LLM + knowledge graph hypothesis systems (e.g. HypoChainer): PubMed 41336152
- Survey, knowledge graphs in AI for science: PMC13154823
