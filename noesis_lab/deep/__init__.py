"""Deep read (DEEP_READ.md): full-text reading of the papers closest to the queued hypotheses.

select.py  which papers (proximity in the gap graph; pure code)
html.py    arXiv HTML / ar5iv (LaTeXML) -> canonical sections (pure code, stdlib parser)
read.py    fetch, extract (LLM, recorded), quote check, recipes, stated gaps, freeze into corpus/deep/
apply.py   load a frozen deep read and apply it to a Snapshot (settings, claims, recipes, stated gaps)
"""
