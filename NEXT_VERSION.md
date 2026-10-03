# NEXT_VERSION — Build order for the hypothesis engine (EXPLORE + GRAPH_GAPS)

This file is the entry point. It sequences **EXPLORE.md** (scout → targeted search → gap graph → hypotheses) and **GRAPH_GAPS.md** (mathematical gap scoring + an autoresearch-style search loop), and resolves where they conflict. **If the two specs disagree, this file wins.**

## Ground rules
- `main` keeps the verified golden session (`make verify rederive replay` all OK). Build on branch `explore`. Switch the submission only after a new session passes all three checks.
- Invariant: LLMs propose, extract and interpret. Code measures, scores, ranks and decides. Nothing is called "novel".
- Reuse LIT_SEARCH modules (retrieval, extraction, quote check, tiers, coverage). Do not fork them.

## Conflicts resolved
1. **Queue order.** EXPLORE §6 (order by novelty type) is **replaced** by GRAPH_GAPS A3 (`gap_score` with its four components). Novelty type stays as a label and a tie-breaker only: resolution > combination > transfer, then alphabetical.
2. **Ideation sources.** Hybrid: the niche search (LIT_SEARCH) and the scout's per-direction searches (EXPLORE §1–2) feed **one** graph. The scout re-runs only on stagnation (GRAPH_GAPS B1 step 7).
3. **Gap status vs gap score.** EXPLORE §3's direction statuses (covered / partial / contested / open / unexplored / outside_testbed) are kept for the map colors. GRAPH_GAPS A2's five gap types and A3's score decide what gets hypotheses.
4. **Seeds.** Exploration uses 1 seed (GRAPH_GAPS B1). Anything that changes a label, the rule's branch or the incumbent still requires the paired protocol (3 seeds, then extra seeds) and the pre-registered rule. Single-seed results are labeled **EXPLORATION — NOT A RESULT** in the UI.
5. **Mechanism field.** Added to the existing extraction schema (verbatim span check; empty if the abstract gives none). Not a separate extraction pass.

## Build stages (each ends in a working, tested state)

| Stage | What | From | Est. |
|---|---|---|---|
| 1 | Scout (one LLM call), title hints used only as searches, per-direction retrieval with provenance, merged into the niche corpus | EXPLORE §1–2 | 1 h |
| 2 | Mechanism field + normalization; method–mechanism–setting–outcome graph; tier-weighted coverage tensor | GRAPH_GAPS A1, A2.1 | 1 h |
| 3 | Gap formulas: ABC closure, link prediction for combinations, contradictions, structural holes; `gap_score` with components; explanation paths | GRAPH_GAPS A2.2–A3 | 1–1.5 h |
| 4 | Two-field combination deltas; hypotheses from top gaps (`grounded_in` validated); novelty-type labels; tighten pass with one revision | EXPLORE §4–5 | 1 h |
| 5 | Dashboard: gap list + gap card (path with quotes) + overview map + direction view | GRAPH_GAPS A5, EXPLORE §7 | 45 min |
| — | **Checkpoint:** re-record a session with the existing paired loop (no exploration cycle yet). `make verify rederive replay`. This is a valid submission on its own. | | 30 min |
| 6 | Explore-1-seed → archive per graph cell → finalists → paired confirmation → promotion (incumbent) → shrinkage metric → stagnation redirect | GRAPH_GAPS B1–B3 | 3–4 h |
| 7 | `make compare-ideation` (niche-only vs scout-only vs hybrid; no training), `make validate-gaps` (held-out claims, precision@k) | GRAPH_GAPS | 45 min |
| — | Re-record, verify, update the README claims table and "What we claim" | | 30 min |

**If time runs short:** stop at the checkpoint after stage 5. Stages 6–7 go in the README as "next version".

## Config additions (`config.yaml`)
```yaml
explore:
  scout_directions: [8, 15]
  per_direction_queries: 3
  per_direction_cap: 15
  link_threshold: 0.25          # claim↔direction TF-IDF, tune on curated set
  mechanism_merge_cosine: 0.6
  top_gaps_for_hypotheses: 8
  tighten_queries: 2
  tighten_cap: 10
search_loop:                    # stage 6
  explore_seeds: 1
  cycles: 3
  finalists_per_cycle: 2
  stagnation_k: 2
```

## Definition of done (stage 5 checkpoint)
- A recorded session in which at least one queued hypothesis came from a graph gap. Its card shows the gap type, score components, the explanation path with verbatim quotes, the tighten-pass outcome and the paired screening result.
- The gap map renders from stored links; replay with the network disabled reproduces it exactly.
- All gap formulas have toy-graph unit tests with known answers.
- The README states: hypotheses are transfer / combination / resolution questions "not found in N retrieved papers"; graph scores are structural hints, not probabilities.

## Prompt for Claude Code
> Read NEXT_VERSION.md, then EXPLORE.md and GRAPH_GAPS.md. NEXT_VERSION.md wins any conflict. Work on branch `explore`. Implement stages 1–5 in order. After each stage run `make test lint` and fix failures before moving on. Keep the invariant: no LLM output may set a score, label, rank, branch or decision. Add unit tests for every pure function you add. Do not modify `results/golden`. Stop at the checkpoint and report what was built, what was cut and why.
