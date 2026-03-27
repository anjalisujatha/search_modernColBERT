.PHONY: setup test

VENV := .venv
PYTHON := $(VENV)/bin/python

setup:
	python3.11 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e ".[notebook,dev]"

test:
	$(PYTHON) -m pytest tests
