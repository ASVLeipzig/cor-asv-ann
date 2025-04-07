SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf1:v3.3.0
DOCKER_TAG ?= ocrd/cor-asv-ann
PYTEST_ARGS ?= -vv

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps        (install required Python packages)"
	@echo "    install     (install this Python package)"
	@echo "    install-dev (install in editable mode)"
	@echo "    build       (build Python source and binary dist)"
	@echo "    docker      (build Docker image)"
	@echo "    deps-test   (install Python packages required for test)"
	@echo "    test        (run tests via Pytest)"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PYTHON"
	@echo "    PYTEST_ARGS   Additional arguments for Pytest"
	@echo "    DOCKER_TAG    Docker image tag of result for the docker target"

# END-EVAL

# (install required Python packages)
deps:
	$(PIP) install -r requirements.txt

deps-test:
	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# deps-ubuntu:
# 	sudo apt-get install -y \
# 		...

# (install this Python package)
install:
	$(PIP) install .

install-dev:
	$(PIP) install -e .

build:
	$(PIP) install build wheel
	$(PYTHON) -m build .

models/%.h5:
	wget -P models https://git.informatik.uni-leipzig.de/ocr-d/cor-asv-ann-models/-/raw/master/$(@F)

# TODO: once core#1149 is fixed, remove this line (so the local copy can be used)
test: export OCRD_BASEURL=https://github.com/OCR-D/assets/raw/refs/heads/master/data/
test: models/s2s.dta19.Fraktur4.d2.w0512.adam.attention.stateless.variational-dropout.char.pretrained+retrained-conf.h5
# Run test
test: tests/assets
	$(PYTHON) -m pytest  tests --durations=0 $(PYTEST_ARGS)

coverage:
	coverage erase
	$(MAKE) test PYTHON="coverage run"
	coverage report -m

#
# Assets
#

# Update OCR-D/assets submodule
.PHONY: always-update tests/assets
testdata: always-update
	git submodule sync --recursive $@
	if git submodule status --recursive $@ | grep -qv '^ '; then \
		git submodule update --init --recursive $@ && \
		touch $@; \
	fi

# Setup test assets
tests/assets: testdata
	mkdir -p $@
	cp -a $</data/* $@

docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

.PHONY: help deps install install-dev build docker deps-test test coverage
