import json

import pytest

from noesis_lab.config import path_of
from noesis_lab.literature import Snapshot, SnapshotError


def test_all_spans_verbatim():
    s = Snapshot(path_of("papers"), path_of("claims"))
    assert s.size >= 6 and len(s.claims) >= 6
    for c in s.claims.values():
        assert c.source_span in s.papers[c.paper_id].abstract


def test_tampered_span_is_rejected(tmp_path):
    doc = json.loads(path_of("claims").read_text())
    doc["claims"][0]["source_span"] += " (embellished)"
    p = tmp_path / "claims.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(SnapshotError):
        Snapshot(path_of("papers"), p)


def test_retrieval_is_deterministic_and_finds_rmsnorm():
    s = Snapshot(path_of("papers"), path_of("claims"))
    q = "Replacing LayerNorm with RMSNorm lowers validation loss"
    a, b = s.search(q, 5), s.search(q, 5)
    assert [c.claim_id for c, _ in a] == [c.claim_id for c, _ in b]
    assert "c04" in [c.claim_id for c, _ in a][:2]


def test_fixtures_reference_real_claims():
    s = Snapshot(path_of("papers"), path_of("claims"))
    for f in json.loads(path_of("fixtures").read_text())["fixtures"]:
        assert set(f["cited_claim_ids"]) <= set(s.claims)
