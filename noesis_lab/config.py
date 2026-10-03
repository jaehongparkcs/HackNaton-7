from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .schemas import ExperimentConfig, Profile, apply_delta

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def path_of(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


def get_profile(name: str) -> Profile:
    profs = load_config()["profiles"]
    if name not in profs:
        raise KeyError(f"unknown profile {name!r}; have {sorted(profs)}")
    return Profile(name=name, **profs[name])


def baseline_config(profile: Profile) -> ExperimentConfig:
    base = ExperimentConfig.model_validate(load_config()["baseline"])
    return apply_delta(base, profile.baseline_overrides) if profile.baseline_overrides else base
