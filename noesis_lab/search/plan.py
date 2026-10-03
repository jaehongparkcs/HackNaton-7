"""Query planning: the LLM proposes queries for the niche; code caps them and always appends one
deterministic query per runnable config change, so every runnable change is searched."""
from __future__ import annotations

from pathlib import Path

from ..literature import TESTBED_DESCRIPTION
from ..llm import LLM
from ..schemas import NicheSpec, QueryPlan
from ..stats import ALLOWED_CHANGES

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
MAX_LLM_QUERIES = 12

# Human-written search phrase per allowed change (never LLM-generated).
CHANGE_QUERIES = {
    "norm=rmsnorm": '"RMSNorm" AND "language model"',
    "norm_position=post": '"Post-LN" AND "Transformer"',
    "activation=swiglu": '"SwiGLU" AND "Transformer"',
    "activation=relu2": '"squared ReLU" AND "Transformer"',
    "pos_encoding=rope": '"rotary position embedding" AND "language model"',
    "schedule=constant": '"learning rate schedule" AND "Transformer" AND "constant"',
    "optimizer=sgd_momentum": '"SGD" AND "Adam" AND "Transformer"',
    "dropout=0.1": '"dropout" AND "Transformer" AND "language model"',
    "qk_norm=true": '"QK normalization" AND "Transformer"',
    "weight_tying=true": '"weight tying" AND "language model"',
    "z_loss_coef=0.0001": '"z-loss" AND "Transformer"',
    "label_smoothing=0.1": '"label smoothing" AND "Transformer"',
    "warmup_frac=0": '"warmup" AND "Transformer" AND "learning rate"',
    "warmup_frac=0.02": '"warmup" AND "Transformer" AND "learning rate"',
    "warmup_frac=0.1": '"warmup" AND "Transformer" AND "learning rate"',
    "grad_clip=0": '"gradient clipping" AND "Transformer"',
    "init_scale=0.5": '"initialization" AND "Transformer" AND "scale"',
    "init_scale=2.0": '"initialization" AND "Transformer" AND "scale"',
    "optimizer=lion": '"Lion" AND "optimizer" AND "Transformer"',
}


def deterministic_queries() -> list[dict]:
    """One query per runnable change. Changes that share a query (the warm-up lengths, the two
    initialization scales) are searched once; the entry names every change it serves."""
    by_q: dict[str, list[str]] = {}
    for k in ALLOWED_CHANGES:
        by_q.setdefault(CHANGE_QUERIES[k], []).append(k)
    return [{"q": q, "kind": "deterministic", "dimension": ", ".join(ks)} for q, ks in by_q.items()]


def plan_queries(llm: LLM, niche: NicheSpec, max_queries: int = MAX_LLM_QUERIES
                 ) -> tuple[QueryPlan | None, list[dict], str]:
    """Returns (plan, queries, error). On an LLM failure the deterministic queries still run."""
    user = (f"{TESTBED_DESCRIPTION}\n\nNICHE (chosen by the PI)\nTitle: {niche.title}\n"
            f"Description: {niche.description}\nInclude terms: {', '.join(niche.include_terms)}\n"
            f"Exclude terms: {', '.join(niche.exclude_terms)}\n\n"
            "Config changes the testbed can run (field=value):\n"
            + "\n".join(f"- {k}" for k in ALLOWED_CHANGES) + "\n")

    def validate(o: QueryPlan) -> None:
        if not 3 <= len(o.dimensions) <= 5:
            raise ValueError("return 3 to 5 dimensions")
        if not all(d.precise or d.broad for d in o.dimensions):
            raise ValueError("every dimension needs at least one query")

    plan, err, queries = None, "", []
    try:
        plan, _ = llm.call("query_plan", (PROMPTS / "query_plan.md").read_text(), user, QueryPlan,
                           validate=validate)
    except Exception as e:  # noqa: BLE001 - search degrades, it does not abort the session
        err = f"{type(e).__name__}: {e}"[:300]
    if plan:
        # interleave precise before broad across dimensions, then cap (enforced by code)
        for kind in ("precise", "broad"):
            for d in plan.dimensions:
                for q in getattr(d, kind):
                    if q.strip() and q.strip() not in [x["q"] for x in queries]:
                        queries.append({"q": q.strip(), "kind": kind, "dimension": d.name})
        queries = queries[:max_queries]
    seen = {x["q"] for x in queries}
    queries += [q for q in deterministic_queries() if q["q"] not in seen]
    return plan, queries, err
