# BEGIN-EVAL makefile-parser --make-help Makefile

SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps       (install required Python packages)"
	#@echo "    deps-test  pip install -r requirements_test.txt"
	@echo ""
	@echo "    install    (install this Python package)"
	#@echo "    test       python -m pytest test"

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# deps-ubuntu:
# 	sudo apt-get install -y \
# 		...

install: deps
	$(PIP) install .

.PHONY: help deps install # deps-test test
