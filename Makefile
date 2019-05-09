# BEGIN-EVAL makefile-parser --make-help Makefile

SHELL = /bin/bash
PYTHON = python
PIP = pip

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps       pip install -r requirements.txt"
	#@echo "    deps-test  pip install -r requirements_test.txt"
	@echo ""
	@echo "    install    pip install -e ."
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

install:
	$(PIP) install -e .

.PHONY: help deps install # deps-test test
