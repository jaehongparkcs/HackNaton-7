"""Synthetic full-text pages for the deep-read tests (LaTeXML structure, like arXiv HTML / ar5iv)."""
from pathlib import Path

DATA = Path(__file__).parent / "data" / "deep"


def page(pid: str, body: str, version: str = "v2") -> str:
    return (f"<html><head><title>{pid}</title></head><body><article class=\"ltx_document\">"
            f"<div class=\"ltx_abstract\"><h6 class=\"ltx_title\">Abstract</h6><p>Abstract of {pid}.</p></div>"
            f"{body}</article><footer class=\"ltx_page_footer\">arXiv:{pid}{version}</footer></body></html>")


def section(title: str, text: str, appendix: bool = False) -> str:
    cls = "ltx_appendix" if appendix else "ltx_section"
    return f"<section class=\"{cls}\"><h2 class=\"ltx_title\">{title}</h2><div class=\"ltx_para\"><p>{text}</p></div></section>"


GENERIC = (section("4 Experimental Setup", "We train Transformer language models with 125M parameters on OpenWebText.")
           + section("5 Results", "RMSNorm performs worse than LayerNorm in our runs.")
           + section("6 Limitations", "We did not evaluate dropout at small scale or on character-level models.")
           + section("Appendix A Hyperparameters", "The learning rate for Lion is typically 3-10x smaller than "
                     "that for AdamW, and the weight decay for Lion is 3-10x larger than that for AdamW.", True))
SMALL = (section("3 Experiments", "In a character-level experiment with 5M parameters, SwiGLU improves over GELU.")
         + section("7 Further results", "SwiGLU improves validation loss over GELU on every task."))


def fetch_factory(*, small: set[str] = frozenset(), missing: set[str] = frozenset(), calls: list | None = None):
    """arxiv.org/html/<id> serves a page (generic + optional small-scale evidence); ids in `missing`
    have neither arXiv HTML nor ar5iv. Any other URL fails (no live network in tests)."""
    def fetch(url: str) -> str:
        if calls is not None:
            calls.append(url)
        if "arxiv.org/html/" not in url:
            raise OSError(f"no fixture for {url}")
        pid = url.rsplit("/", 1)[1]
        if pid in missing:
            raise OSError("404")
        if "ar5iv" in url:
            raise OSError("404")
        return page(pid, GENERIC + (SMALL if pid in small else ""))
    return fetch
