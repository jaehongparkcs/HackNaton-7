import shutil

import pytest

from prior_art.config import ROOT
from prior_art.schemas import CurvePoint, ExperimentConfig, RunResult


def make_run(val: float, seed: int, tokens: int = 1000, cfg: ExperimentConfig | None = None,
             curve: bool = True) -> RunResult:
    cfg = cfg or ExperimentConfig()
    pts = [CurvePoint(train_s=0, step=0, tokens_seen=0, val_loss=4.0),
           CurvePoint(train_s=1, step=10, tokens_seen=tokens, val_loss=val)] if curve else []
    return RunResult(run_id=f"run_{cfg.config_hash()}_{seed}", config_hash=cfg.config_hash(),
                     config=cfg, profile_hash="p", seed=seed, status="ok", val_loss=val,
                     tokens_seen=tokens, steps=10, curve=pts)


@pytest.fixture
def session_name():
    name = "pytest-smoke"
    for p in (ROOT / "results" / name, ROOT / "work" / f"replay-{name}"):
        shutil.rmtree(p, ignore_errors=True)
    yield name
    for p in (ROOT / "results" / name, ROOT / "work" / f"replay-{name}"):
        shutil.rmtree(p, ignore_errors=True)
