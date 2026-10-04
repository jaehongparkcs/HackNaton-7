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


def bundle_config(bundle: Path) -> dict[str, Any]:
    """The config a bundle was recorded with. Replay and rederive use it, so later edits to
    config.yaml (budgets, rule version, engine settings) never change how an old bundle replays."""
    snap = Path(bundle) / "config.snapshot.yaml"
    return yaml.safe_load(snap.read_text()) if snap.exists() else load_config()


def protocol(cfg: dict[str, Any]) -> dict[str, int]:
    """Versions of the pre-registered rule, the prior-art gate and the gap-prediction rule. Bundles
    recorded before a key existed are version 1 of it."""
    p = cfg.get("protocol", {})
    return {"rule_version": int(p.get("rule_version", 1)), "gate_version": int(p.get("gate_version", 1)),
            "prediction_version": int(p.get("prediction_version", 1)),
            "deep_read_version": int(p.get("deep_read_version", 0)),       # 0 = no deep read
            "search_loop": bool(cfg.get("search_loop", {}).get("enabled", False))}


def with_profile_overrides(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    """`profile_overrides.<profile>` merged into the config's sections (one level deep), e.g. the
    rehearse profile's shorter search loop. Replay applies it from the bundle's own snapshot."""
    over = cfg.get("profile_overrides", {}).get(profile)
    if not over:
        return cfg
    return {**cfg, **{k: ({**cfg.get(k, {}), **v} if isinstance(v, dict) else v) for k, v in over.items()}}


def path_of(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


def get_profile(name: str, cfg: dict[str, Any] | None = None) -> Profile:
    profs = (cfg or load_config())["profiles"]
    if name not in profs:
        raise KeyError(f"unknown profile {name!r}; have {sorted(profs)}")
    return Profile(name=name, **profs[name])


def baseline_config(profile: Profile, cfg: dict[str, Any] | None = None) -> ExperimentConfig:
    base = ExperimentConfig.model_validate((cfg or load_config())["baseline"])
    return apply_delta(base, profile.baseline_overrides) if profile.baseline_overrides else base
