"""Full-text parsing of arXiv's native HTML and ar5iv. Both are produced by LaTeXML, so one parser
covers them: sections are <section class="ltx_section|ltx_subsection|ltx_appendix ..."> with an
<h2..h6 class="ltx_title"> heading, the abstract is <div class="ltx_abstract">, references are
<section class="ltx_bibliography">. Pure code (stdlib html.parser), no network.

Figures and references are dropped; table text is kept (hyperparameter tables matter); MathML is
replaced by its LaTeX `alttext`, so a number written in math survives as text. Whitespace is
collapsed, and every quote check runs on this normalized text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

SECTIONS = ("abstract", "introduction", "method", "experimental_setup", "results", "limitations",
            "conclusion", "appendix_setup")
VOID = {"br", "img", "hr", "meta", "link", "input", "col", "area", "base", "embed", "source", "track", "wbr", "param"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "button", "svg"}
SKIP_CLASSES = ("ltx_bibliography", "ltx_figure", "ltx_page_footer", "ltx_page_header", "ltx_tag_section",
                "ltx_role_footnote", "ltx_dates", "ltx_authors", "ltx_TOC")
BLOCK = {"p", "div", "section", "table", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "figure",
         "figcaption", "blockquote", "br", "caption", "thead", "tbody"}
CELL = {"td", "th"}
SPECIFIC = ("limitations", "conclusion", "results", "experimental_setup", "appendix_setup")
SETUP_WORDS = ("setup", "set-up", "implementation", "hyperparameter", "hyper-parameter", "training detail",
               "experimental detail", "experiment", "training")


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    children: list = field(default_factory=list)

    @property
    def classes(self) -> list[str]:
        return (self.attrs.get("class") or "").split()


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, {k: v or "" for k, v in attrs})
        self.stack[-1].children.append(n)
        if tag not in VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {k: v or "" for k, v in attrs}))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):     # tolerate unclosed tags: pop to the match
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_tree(html: str) -> Node:
    b = _Builder()
    b.feed(html)
    b.close()
    return b.root


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _skip(n: Node) -> bool:
    return n.tag in SKIP_TAGS or any(c in n.classes for c in SKIP_CLASSES)


def _text(n: Node | str, out: list[str], *, nested_sections: bool = True) -> None:
    if isinstance(n, str):
        out.append(n)
        return
    if _skip(n):
        return
    if n.tag == "math":
        out.append(f" {n.attrs.get('alttext', '')} ")
        return
    if not nested_sections and n.tag == "section":
        return
    if n.tag in BLOCK:
        out.append(" ")
    for c in n.children:
        _text(c, out, nested_sections=nested_sections)
    if n.tag in CELL:
        out.append(" | ")
    elif n.tag in BLOCK:
        out.append(" ")


def text_of(n: Node, *, nested_sections: bool = True) -> str:
    """Text of `n`; with nested_sections=False, the section's own text without its subsections."""
    out: list[str] = []
    for c in ([n] if nested_sections else n.children):
        _text(c, out, nested_sections=nested_sections)
    return norm("".join(out))


def classify(heading: str, *, appendix: bool) -> str:
    """Canonical section for a heading. Appendix sections count only when they are about the setup."""
    h = re.sub(r"^[\dA-Z]+(\.\d+)*\s+", "", norm(heading)).lower()
    if appendix:
        return "appendix_setup" if any(w in h for w in SETUP_WORDS) else "other"
    if "limitation" in h:
        return "limitations"
    if any(w in h for w in ("conclusion", "future work", "discussion")):
        return "conclusion"
    if "result" in h or "evaluation" in h:
        return "results"
    if any(w in h for w in SETUP_WORDS):
        return "experimental_setup"
    if "introduction" in h:
        return "introduction"
    if any(w in h for w in ("method", "approach", "architecture", "model", "proposed", "algorithm")):
        return "method"
    return "other"


def _heading(n: Node) -> str:
    for c in n.children:
        if isinstance(c, Node) and re.fullmatch(r"h[1-6]", c.tag):
            return text_of(c)
    return ""


@dataclass
class ParsedPaper:
    sections: dict[str, str]                 # canonical name -> normalized text
    headings: list[dict]                     # [{"heading", "section", "chars"}] in document order
    version: str


def _walk(n: Node, parent_class: str, appendix: bool, acc: dict[str, list[str]], heads: list[dict]) -> None:
    for c in n.children:
        if not isinstance(c, Node) or _skip(c):
            continue
        if "ltx_abstract" in c.classes:
            t = re.sub(r"^Abstract\.?\s*", "", text_of(c))
            acc.setdefault("abstract", []).append(t)
            heads.append({"heading": "Abstract", "section": "abstract", "chars": len(t)})
            continue
        if c.tag == "section":
            app = appendix or "ltx_appendix" in c.classes
            head = _heading(c)
            name = classify(head, appendix=app)
            # a subsection inherits its section's class unless its own heading is more specific
            # ("3.1 Language modeling" under "Experiments" stays experimental_setup)
            if parent_class != "other" and "ltx_appendix" not in c.classes and name not in SPECIFIC:
                name = parent_class
            own = text_of(c, nested_sections=False)
            own = own[len(head):].strip() if head and own.startswith(head) else own
            heads.append({"heading": head, "section": name, "chars": len(own)})
            if name != "other" and own:
                acc.setdefault(name, []).append(own)
            _walk(c, name, app, acc, heads)
        else:
            _walk(c, parent_class, appendix, acc, heads)


def served_version(html: str, paper_id: str) -> str:
    """The arXiv version the page shows (e.g. 2302.06675v4), or "unknown"."""
    base = re.escape(paper_id.split("v")[0])
    visible = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    m = re.search(rf"{base}(v\d+)", visible)
    return f"{paper_id.split('v')[0]}{m.group(1)}" if m else "unknown"


def parse_paper(html: str, paper_id: str) -> ParsedPaper:
    root = parse_tree(html)
    acc: dict[str, list[str]] = {}
    heads: list[dict] = []
    _walk(root, "other", False, acc, heads)
    return ParsedPaper(sections={k: norm(" ".join(v)) for k, v in acc.items() if k in SECTIONS},
                       headings=heads, version=served_version(html, paper_id))


def quote_in(quote: str, section_text: str) -> bool:
    q = norm(quote)
    return bool(q) and q in section_text
