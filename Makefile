# cgp-rsi — standard operations.
#
# Targets:
#   make test      Unit tests + CI mock end-to-end (CPU)
#   make build     Pack both .cgp packages via cpm
#   make verify    Verify the packed archives
#   make mock-e2e  Run the 2-generation MOCK pipeline
#   make ci        Reproduce the encapsulated Alpine CI build locally (requires docker)
#   make clean     Remove build artifacts

CPM ?= cpm
NAMES := base-model-v0 rsi-controller

.PHONY: all test build verify mock-e2e install clean ci

all: test build verify

test:
	python3 -m venv .venv && . .venv/bin/activate && pip install -q -e '.[dev]' && pytest -q

mock-e2e:
	python3 scripts/mock_e2e.py

build:
	cd packages/base-model-v0 && $(CPM) pack --manifest cognitive.json
	cd packages/rsi-controller && chmod 755 tools/rsi-controller && $(CPM) pack --bin tools

verify:
	cd packages/base-model-v0 && $(CPM) verify base-model-v0-*.cgp
	cd packages/rsi-controller && $(CPM) verify rsi-controller-*.cgp

install: build
	cd packages/base-model-v0 && $(CPM) install base-model-v0-*.cgp
	cd packages/rsi-controller && $(CPM) install rsi-controller-*.cgp

ci:
	docker build -t cgp-ci:local -f .github/docker/Dockerfile.ci .
	rm -rf out
	id=$$(docker create cgp-ci:local); \
	docker cp $$id:/out ./out; \
	docker rm $$id
	@echo "CI artifacts in ./out/"

clean:
	rm -f packages/*/*.cgp
	rm -rf runs .venv .pytest_cache **/__pycache__
