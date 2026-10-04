"""HTTP GET with a polite rate limit, backoff that respects the server, and a raw-response cache
that doubles as a resume log (the frozen record of what the APIs returned). Successful responses
are keyed by URL and reused on a re-run, so a re-run only fetches what failed; failures are never
cached. Tests inject a fake `fetch`; nothing here is called in replay."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

RETRIABLE = (429, 503)                       # arXiv throttles with 429; 503 is a transient outage
BACKOFF = (15.0, 45.0, 120.0)                # seconds between retries when there is no Retry-After header


def urllib_fetch(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "noesis-lab/0.1 (research prototype)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _status(e: Exception) -> int | None:
    return getattr(e, "code", None) or getattr(e, "status", None)


def _retry_after(e: Exception) -> float | None:
    """Seconds the server asked us to wait, if it said so (integer form only)."""
    headers = getattr(e, "headers", None)
    try:
        value = headers.get("Retry-After") if headers else None
        return float(value) if value is not None and str(value).strip().isdigit() else None
    except Exception:  # noqa: BLE001
        return None


class RawCache:
    """Every raw API response is written to `raw/` with an index (url -> file). On construction the
    existing index is loaded, so a response fetched on an earlier run is reused rather than fetched
    again. One request is ever in flight (no concurrency); 429/503 back off and retry."""

    def __init__(self, raw_dir: Path, fetch: Callable[[str], str] = urllib_fetch,
                 min_interval_s: float = 5.0, sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 3, backoff: tuple[float, ...] = BACKOFF):
        self.dir, self.fetch, self.min_interval, self.sleep = raw_dir, fetch, min_interval_s, sleep
        self.max_retries, self.backoff = max_retries, backoff
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []
        self.failures: list[dict] = []
        self.cache: dict[str, str] = {}          # url -> file name of a successful response
        self.attempted = 0                       # requests that were not served from cache
        self.reused = 0
        self._last = 0.0
        idx = self.dir / "index.json"
        if idx.exists():                         # resume: reuse what a previous run already fetched
            for rec in json.loads(idx.read_text()):
                if (self.dir / rec["file"]).exists():
                    self.index.append(rec)
                    self.cache[rec["url"]] = rec["file"]

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def get(self, kind: str, url: str, ext: str) -> str | None:
        """Returns the body, or None once retries are exhausted (recorded in `failures`). A URL
        already in the cache is returned without a network call and never counts as attempted."""
        if url in self.cache:
            self.reused += 1
            return (self.dir / self.cache[url]).read_text()
        self.attempted += 1
        err, status = "", None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                body = self.fetch(url)
            except Exception as e:  # noqa: BLE001 - network failures are data (search_degraded)
                err, status = f"{type(e).__name__}: {e}"[:300], _status(e)
                if attempt >= self.max_retries:
                    break
                if status in RETRIABLE:
                    self.sleep(_retry_after(e) or self.backoff[min(attempt, len(self.backoff) - 1)])
                    continue
                if attempt >= 1:          # one quick retry for a transient non-HTTP error, then give up
                    break
                continue
            name = f"{len(self.index):03d}_{kind}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.{ext}"
            (self.dir / name).write_text(body)
            self.index.append({"kind": kind, "url": url, "file": name})
            self.cache[url] = name
            self.write_index()            # persist after every success so an interrupted run still resumes
            return body
        self.failures.append({"kind": kind, "url": url, "error": err, "status": status})
        return None

    @property
    def failure_fraction(self) -> float:
        """Failed requests over requests actually attempted (cache hits are not re-counted)."""
        return len(self.failures) / self.attempted if self.attempted else 0.0

    def write_index(self) -> None:
        (self.dir / "index.json").write_text(json.dumps(self.index, indent=2) + "\n")
