SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf1:v2.69.0
DOCKER_TAG ?= ocrd/cor-asv-ann

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     (install required Python packages)"
	@echo "    install  (install this Python package)"
	@echo "    docker   (build Docker image)"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PYTHON"
	@echo "    DOCKER_TAG    Docker image tag of result for the docker target"

# END-EVAL

# (install required Python packages)
deps:
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# deps-ubuntu:
# 	sudo apt-get install -y \
# 		...

# (install this Python package)
install: deps
	$(PIP) install .

docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

.PHONY: help deps install docker # deps-test test
