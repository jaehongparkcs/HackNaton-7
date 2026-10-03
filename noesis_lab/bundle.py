"""Results bundle: results/<session>/ with a SHA256 manifest (BUILD_PLAN s8)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from .config import ROOT, load_config
from .store import Store

BUNDLE_FILES = ["notebook.sqlite", "runs.jsonl", "recordings/llm.jsonl", "env.json",
                "config.snapshot.yaml", "inputs.json", "state.json"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    cfg = load_config()["paths"]
    out = {k: sha256_file(ROOT / cfg[k]) for k in ("data", "papers", "claims", "fixtures")}
    for p in sorted((ROOT / "prompts").glob("*.md")):
        out[f"prompts/{p.name}"] = sha256_file(p)
    return out


def seal(out_dir: Path, store: Store, *, meta: dict) -> dict:
    """Checkpoint the DB into a single file, write state/env/inputs, then the manifest."""
    state = {"state_digest": store.state_digest(), **meta,
             "counts": {"runs": len(store.runs()), "events": len(store.events()),
                        "hypotheses": len(store.hypotheses()), "analyses": len(store.analyses()),
                        "decisions": len(store.decisions())}}
    store.db.commit()
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.db.execute("PRAGMA journal_mode=DELETE")
    store.close()
    (out_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(load_config(), sort_keys=True))
    (out_dir / "inputs.json").write_text(json.dumps(input_hashes(), indent=2, sort_keys=True) + "\n")
    runs = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines() if line]
    env = runs[0]["env"] if runs else {}
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    write_manifest(out_dir)
    return state


def write_manifest(out_dir: Path) -> None:
    files = {}
    for rel in BUNDLE_FILES:
        p = out_dir / rel
        if p.exists():
            files[rel] = sha256_file(p)
    (out_dir / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")


def verify_manifest(out_dir: Path) -> list[str]:
    """Returns a list of problems (empty = every file matches its recorded SHA256)."""
    mf = out_dir / "MANIFEST.json"
    if not mf.exists():
        return ["MANIFEST.json missing"]
    problems = []
    for rel, digest in json.loads(mf.read_text())["files"].items():
        p = out_dir / rel
        if not p.exists():
            problems.append(f"{rel}: missing")
        elif sha256_file(p) != digest:
            problems.append(f"{rel}: sha256 mismatch")
    return problems


def fresh_dir(path: Path, force: bool) -> Path:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
