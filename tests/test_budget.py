"""P0-1: the headline profile decides on a fixed step budget, and nothing still says wall-clock."""
from pathlib import Path

from noesis_lab.config import ROOT, get_profile
from noesis_lab.evidence_chain import chain_dot, evidence_grid
from noesis_lab.literature import TESTBED_DESCRIPTION
from noesis_lab.store import Store

from .test_session import smoke  # noqa: F401  (fixture)


def test_full_profile_is_a_fixed_step_budget():
    p = get_profile("full")
    assert (p.budget_mode, p.max_steps, p.eval_every_steps) == ("steps", 2000, 250)


def test_no_prompt_or_description_mentions_a_wall_clock_budget():
    texts = [TESTBED_DESCRIPTION] + [f.read_text() for f in (ROOT / "prompts").glob("*.md")]
    texts.append((ROOT / "data" / "fixtures.json").read_text())
    for t in texts:
        assert "60 second" not in t and "60 s " not in t and "wall-clock budget" not in t


def test_evidence_chain_marks_llm_nodes_dashed_and_ends_fixture_a_in_runs_avoided(smoke):  # noqa: F811
    _, _, d = smoke
    S = Store(Path(d) / "notebook.sqlite", readonly=True)
    D = dict(hyps=S.hypotheses(), analyses=S.analyses(), decisions=S.decisions(),
             evidence=S.derived_evidence(), claims={c["claim_id"]: c for c in S.claims()})
    dot = chain_dot(D, "hyp_fixture_a_preln_nowarmup")
    assert "runs avoided" in dot and "dashed" in dot and "solid" in dot
    assert "known_in_corpus" in dot
    rows = evidence_grid(D)
    assert len(rows) == len(D["hyps"]) and rows[0]["covers our setting (human-curated)"] == "yes"
