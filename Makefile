SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps     pip install -r requirements"
	@echo "    install  pip install . (incl. deps)"

# END-EVAL

# pip install -r requirements
deps:
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# deps-ubuntu:
# 	sudo apt-get install -y \
# 		...

# pip install . (incl. deps)
install: deps
	$(PIP) install .

.PHONY: help deps install # deps-test test
