SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     (install required Python packages)"
	@echo "    install  (install this Python package)"
	@echo ""

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

.PHONY: help deps install # deps-test test
