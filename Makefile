# NOESIS LAB. `make setup` is the only install step. Replay/rederive need no API key, GPU or Mac.
SESSION ?= golden
NICHE ?=
CORPUS ?=
PY := uv run python
.PHONY: search setup test lint check verify-snapshot fetch-snapshot timing smoke golden replay rederive verify app clean

setup:                      ## install pinned deps from uv.lock
	uv sync --frozen

test:                       ## unit + end-to-end tests on CPU with a mocked LLM (no key)
	uv run pytest

lint:
	uv run ruff check .

verify-snapshot:            ## every claim span must be verbatim in its abstract
	$(PY) scripts/verify_snapshot.py

fetch-snapshot:             ## re-fetch arXiv abstracts (data/papers.json is committed; rarely needed)
	$(PY) scripts/fetch_snapshot.py

timing:                     ## measure throughput on this machine (budget is a fixed step count, so this only reports speed)
	$(PY) -m noesis_lab timing --profile full --steps 300

smoke:                      ## CPU pipeline check with the MOCK LLM (never a result)
	$(PY) -m noesis_lab session --session smoke --profile smoke --llm mock --force

golden:                     ## the real thing: live Claude + real MPS runs, recorded into results/$(SESSION)
	$(PY) -m noesis_lab session --session $(SESSION) --profile full --llm live $(if $(NICHE),--niche $(NICHE)) $(if $(CORPUS),--corpus $(CORPUS))

search:                     ## live literature search for NICHE (default niche.yaml), frozen to work/corpus-<niche>; no training
	$(PY) -m noesis_lab search --niche $(or $(NICHE),niche.yaml)

replay:                     ## R2: rebuild the full session from recordings, no key/GPU; asserts identical state
	$(PY) -m noesis_lab replay --session $(SESSION)

rederive:                   ## R1: recompute every statistic from stored runs; exact match required
	$(PY) -m noesis_lab rederive --session $(SESSION)

verify:                     ## SHA256 manifest of the results bundle
	$(PY) -m noesis_lab verify --session $(SESSION)

check: test verify-snapshot smoke   ## what CI would run
	$(PY) -m noesis_lab replay --session smoke
	$(PY) -m noesis_lab rederive --session smoke
	$(PY) -m noesis_lab verify --session smoke

app:                        ## dashboard (reads results/*/ and work/replay-*/ notebooks)
	uv run streamlit run app.py

clean:
	rm -rf work results/smoke results/pytest-smoke .pytest_cache .ruff_cache
