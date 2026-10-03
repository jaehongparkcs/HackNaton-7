"""Check that every claim span is a verbatim substring of its paper's abstract (BUILD_PLAN s4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_art.config import path_of  # noqa: E402
from prior_art.literature import Snapshot, SnapshotError  # noqa: E402

try:
    s = Snapshot(path_of("papers"), path_of("claims"))
except SnapshotError as e:
    sys.exit(f"SNAPSHOT INVALID: {e}")
print(f"OK: {s.size} papers, {len(s.claims)} claims, all spans verbatim; dimensions: {s.dimensions()}")
