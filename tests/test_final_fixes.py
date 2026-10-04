"""FINAL_FIXES: a bundle holds only what its manifest lists, and the polish / speed changes keep
old bundles replaying unchanged."""
import json

from noesis_lab.bundle import verify_manifest, write_manifest


# --------------------------------------------------------------------------- 0. unexpected files
def _bundle(tmp_path):
    d = tmp_path / "b"
    (d / "corpus" / "raw").mkdir(parents=True)
    (d / "state.json").write_text("{}\n")
    (d / "corpus" / "raw" / "001_arxiv.xml").write_text("<feed/>")
    write_manifest(d)
    return d


def test_verify_passes_on_a_clean_bundle(tmp_path):
    assert verify_manifest(_bundle(tmp_path)) == []


def test_verify_fails_on_files_the_manifest_does_not_list(tmp_path):
    d = _bundle(tmp_path)
    (d / "corpus" / "raw" / "001_arxiv 2.xml").write_text("<feed/>")      # an iCloud conflict copy
    (d / "state 2.json").write_text("{}\n")
    problems = verify_manifest(d)
    assert problems == ["corpus/raw/001_arxiv 2.xml: not in MANIFEST.json (unexpected file)",
                        "state 2.json: not in MANIFEST.json (unexpected file)"]


def test_verify_ignores_finder_metadata_but_not_the_manifest_contents(tmp_path):
    d = _bundle(tmp_path)
    (d / ".DS_Store").write_bytes(b"\0")
    assert verify_manifest(d) == []
    files = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert "MANIFEST.json" not in files and set(files) == {"state.json", "corpus/raw/001_arxiv.xml"}
