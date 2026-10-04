# NOESIS LAB. `make setup` is the only install step. Replay/rederive need no API key, GPU or Mac.
SESSION ?= golden
NICHE ?=
CORPUS ?=
PY := uv run python
# Long live commands run under `caffeinate -i` on macOS so the machine cannot sleep mid-session
# (explore2 lost ~57 minutes after its last run). Elsewhere it is a no-op.
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i)
LIVE := $(CAFFEINATE) $(PY)
.PHONY: deep-read verify-deep record search setup test lint check verify-snapshot fetch-snapshot timing smoke golden rehearse replay rederive verify app clean

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
	$(LIVE) -m noesis_lab session --session $(SESSION) --profile full --llm live $(if $(NICHE),--niche $(NICHE)) $(if $(CORPUS),--corpus $(CORPUS))

record:                     ## search (resumes from cache; aborts if >10% requests fail) -> deep read -> gate stability (aborts if UNSTABLE) -> session -> verify -> rederive -> replay
	@test -n "$(NICHE)" || (echo "usage: make record SESSION=<name> NICHE=niche.yaml" && exit 2)
	$(LIVE) -m noesis_lab search --niche $(NICHE) --out work/corpus-$(SESSION)
	$(LIVE) -m noesis_lab deep-read --corpus work/corpus-$(SESSION)      # no-op unless deep_read.enabled
	$(LIVE) -m noesis_lab lit-dryrun --corpus work/corpus-$(SESSION) --repeat 3 || \
	  (echo "ABORT: the prior-art gate is UNSTABLE on this corpus (see rows above). Nothing was recorded." && exit 1)
	$(LIVE) -m noesis_lab session --session $(SESSION) --profile full --llm live --corpus work/corpus-$(SESSION)
	$(PY) -m noesis_lab verify --session $(SESSION)
	$(PY) -m noesis_lab rederive --session $(SESSION)
	$(PY) -m noesis_lab replay --session $(SESSION)

search:                     ## live literature search for NICHE (default niche.yaml), frozen to work/corpus-<niche>; no training
	$(LIVE) -m noesis_lab search --niche $(or $(NICHE),niche.yaml)

deep-read:                  ## full text of the papers closest to the top hypotheses, frozen into CORPUS/deep/ (DEEP_READ.md)
	@test -n "$(CORPUS)" || (echo "usage: make deep-read CORPUS=work/corpus-<name>" && exit 2)
	$(LIVE) -m noesis_lab deep-read --corpus $(CORPUS)

verify-deep:                ## re-fetch the papers a bundle read; check HTML SHA256 and every quote (needs network)
	$(PY) -m noesis_lab verify-deep --session $(SESSION)

rehearse:                   ## ~5 min dry run of the live pipeline on a frozen CORPUS (500 steps, 1 short cycle). Never a result.
	@test -n "$(CORPUS)" || (echo "usage: make rehearse CORPUS=work/corpus-<name>" && exit 2)
	$(LIVE) -m noesis_lab session --session rehearse --profile rehearse --llm live --corpus $(CORPUS) --force

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
