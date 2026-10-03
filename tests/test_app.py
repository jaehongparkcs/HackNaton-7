from streamlit.testing.v1 import AppTest

from prior_art.config import ROOT

from .test_session import smoke  # noqa: F401  (fixture)


def test_dashboard_renders_without_exceptions(smoke):  # noqa: F811
    name, _, d = smoke
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    import os
    os.environ["PRIOR_ART_NOTEBOOK"] = str(d / "notebook.sqlite")
    try:
        at.run()
    finally:
        os.environ.pop("PRIOR_ART_NOTEBOOK", None)
    assert not at.exception, [e.value for e in at.exception]
    assert any("REJECTED: PRIOR ART" in m.value for m in at.markdown)
    assert any("NOT CONFIRMED" in m.value for m in at.markdown)
    assert any("MOCK LLM" in w.value for w in at.warning)
