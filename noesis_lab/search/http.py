"""HTTP GET with a polite rate limit, one retry and a raw-response cache (the frozen record of
what the APIs returned). Tests inject a fake `fetch`; nothing here is called in replay."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path


def urllib_fetch(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "noesis-lab/0.1 (research prototype)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


class RawCache:
    """Every raw API response is written to `raw/` with an index (url -> file)."""

    def __init__(self, raw_dir: Path, fetch: Callable[[str], str] = urllib_fetch,
                 min_interval_s: float = 3.0, sleep: Callable[[float], None] = time.sleep):
        self.dir, self.fetch, self.min_interval, self.sleep = raw_dir, fetch, min_interval_s, sleep
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []
        self.failures: list[dict] = []
        self._last = 0.0

    def get(self, kind: str, url: str, ext: str) -> str | None:
        """Returns the body, or None after one retry failed (recorded in `failures`)."""
        err = ""
        for _attempt in (1, 2):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                body = self.fetch(url)
            except Exception as e:  # noqa: BLE001 - network failures are data (search_degraded)
                err = f"{type(e).__name__}: {e}"[:300]
                continue
            name = f"{len(self.index):03d}_{kind}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.{ext}"
            (self.dir / name).write_text(body)
            self.index.append({"kind": kind, "url": url, "file": name})
            return body
        self.failures.append({"kind": kind, "url": url, "error": err})
        return None

    def write_index(self) -> None:
        (self.dir / "index.json").write_text(json.dumps(self.index, indent=2) + "\n")
