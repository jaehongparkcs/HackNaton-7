"""Evidence store (SQLite, WAL so the dashboard can read while the orchestrator writes).

SQLite is the single source of truth; the dashboard and the results bundle are derived from it.
Provenance is a generic edge table so any number can be walked back:
  decision -> analysis -> runs -> config, and hypothesis -> claim -> paper span.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import canonical_json, sha256_hex

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS papers(paper_id TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS claims(claim_id TEXT PRIMARY KEY, paper_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, seq INTEGER, kind TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS hypotheses(hypothesis_id TEXT PRIMARY KEY, fixture_id TEXT,
                                      status TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, config_hash TEXT, seed INTEGER,
                                run_order INTEGER, payload TEXT);
CREATE TABLE IF NOT EXISTS noise_floors(noise_id TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS analyses(analysis_id TEXT PRIMARY KEY, hypothesis_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, seq INTEGER, kind TEXT,
                                     hypothesis_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS derived_evidence(evidence_id TEXT PRIMARY KEY, claim_id TEXT,
                                            analysis_id TEXT, relation TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS links(src_type TEXT, src_id TEXT, rel TEXT, dst_type TEXT, dst_id TEXT,
                                 PRIMARY KEY(src_type, src_id, rel, dst_type, dst_id));
"""

# Tables hashed by state_digest (everything derived from measurements and LLM calls).
DIGEST_TABLES = ["papers", "claims", "events", "hypotheses", "runs", "noise_floors", "analyses",
                 "decisions", "derived_evidence", "links"]


def _dump(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return json.dumps(obj, sort_keys=True)


class Store:
    def __init__(self, path: str | Path, *, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row
        self._ev_seq = self._max("events", "seq")
        self._dec_seq = self._max("decisions", "seq")

    def _max(self, table: str, col: str) -> int:
        try:
            v = self.db.execute(f"SELECT MAX({col}) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            return 0
        return v or 0

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # ---------------------------------------------------------------- writers
    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, _dump(value)))
        self.db.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def put_paper(self, p: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO papers VALUES(?,?)", (p.paper_id, _dump(p)))
        self.db.commit()

    def put_claim(self, c: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO claims VALUES(?,?,?)",
                        (c.claim_id, c.paper_id, _dump(c)))
        self.db.commit()

    def add_event(self, kind: str, payload: dict) -> str:
        self._ev_seq += 1
        eid = f"evt_{self._ev_seq:04d}"
        self.db.execute("INSERT INTO events VALUES(?,?,?,?)",
                        (eid, self._ev_seq, kind, json.dumps({**payload, "event_id": eid}, sort_keys=True)))
        self.db.commit()
        return eid

    def put_hypothesis(self, hid: str, fixture_id: str, status: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO hypotheses VALUES(?,?,?,?) ON CONFLICT(hypothesis_id) DO UPDATE SET "
            "fixture_id=excluded.fixture_id, status=excluded.status, payload=excluded.payload",
            (hid, fixture_id, status, json.dumps(payload, sort_keys=True)))
        self.db.commit()

    def put_run(self, r: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?)",
                        (r.run_id, r.config_hash, r.seed, r.run_order, _dump(r)))
        self.db.commit()

    def put_noise(self, noise_id: str, n: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO noise_floors VALUES(?,?)", (noise_id, _dump(n)))
        self.db.commit()

    def put_analysis(self, a: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO analyses VALUES(?,?,?)",
                        (a.analysis_id, a.hypothesis_id, _dump(a)))
        self.db.commit()

    def add_decision(self, kind: str, hypothesis_id: str, payload: dict) -> str:
        self._dec_seq += 1
        did = f"dec_{self._dec_seq:03d}"
        self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?)",
                        (did, self._dec_seq, kind, hypothesis_id,
                         json.dumps({**payload, "decision_id": did}, sort_keys=True)))
        self.db.commit()
        return did

    def put_evidence(self, e: BaseModel) -> None:
        self.db.execute("INSERT OR REPLACE INTO derived_evidence VALUES(?,?,?,?,?)",
                        (e.evidence_id, e.claim_id, e.analysis_id, e.relation, _dump(e)))
        self.db.commit()

    def link(self, src_type: str, src_id: str, rel: str, dst_type: str, dst_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO links VALUES(?,?,?,?,?)",
                        (src_type, src_id, rel, dst_type, dst_id))
        self.db.commit()

    # ---------------------------------------------------------------- readers
    def _all(self, table: str, order: str) -> list[dict]:
        return [json.loads(r["payload"]) for r in
                self.db.execute(f"SELECT payload FROM {table} ORDER BY {order}")]

    def runs(self) -> list[dict]:
        return self._all("runs", "run_order, run_id")

    def run(self, run_id: str) -> dict | None:
        r = self.db.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def events(self) -> list[dict]:
        return self._all("events", "seq")

    def hypotheses(self) -> list[dict]:
        return [{**json.loads(r["payload"]), "hypothesis_id": r["hypothesis_id"],
                 "status": r["status"], "fixture_id": r["fixture_id"]}
                for r in self.db.execute("SELECT * FROM hypotheses ORDER BY rowid")]

    def analyses(self) -> list[dict]:
        return self._all("analyses", "rowid")

    def decisions(self) -> list[dict]:
        return [{**json.loads(r["payload"]), "kind": r["kind"], "hypothesis_id": r["hypothesis_id"]}
                for r in self.db.execute("SELECT * FROM decisions ORDER BY seq")]

    def noise_floors(self) -> dict[str, dict]:
        return {r["noise_id"]: json.loads(r["payload"])
                for r in self.db.execute("SELECT * FROM noise_floors")}

    def derived_evidence(self) -> list[dict]:
        return self._all("derived_evidence", "rowid")

    def claims(self) -> list[dict]:
        return self._all("claims", "claim_id")

    def papers(self) -> list[dict]:
        return self._all("papers", "paper_id")

    def edges(self, src_type: str | None = None, src_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM links", []
        if src_type:
            q += " WHERE src_type=? AND src_id=?"
            args = [src_type, src_id]
        return [dict(r) for r in self.db.execute(q + " ORDER BY rowid", args)]

    def provenance(self, node_type: str, node_id: str, _seen: set | None = None) -> dict:
        """Walk outgoing edges: decision -> analysis -> runs, hypothesis -> claim, etc."""
        seen = _seen if _seen is not None else set()
        key = (node_type, node_id)
        if key in seen:
            return {"type": node_type, "id": node_id, "cycle": True}
        seen.add(key)
        children = [{"rel": e["rel"], **self.provenance(e["dst_type"], e["dst_id"], seen)}
                    for e in self.edges(node_type, node_id)]
        return {"type": node_type, "id": node_id, "children": children}

    # ---------------------------------------------------------------- digest
    def state_digest(self) -> str:
        """Hash of the whole evidence state. Replay must reproduce this exactly."""
        h: dict[str, list] = {}
        for t in DIGEST_TABLES:
            rows = [list(r) for r in self.db.execute(f"SELECT * FROM {t}")]
            h[t] = sorted(rows, key=canonical_json)
        return sha256_hex(canonical_json(h))

    def iter_run_payloads(self) -> Iterable[dict]:
        return iter(self.runs())
