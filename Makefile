.PHONY: setup test

VENV := .venv
PYTHON := $(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -e ".[notebook,dev]"

test:
	$(PYTHON) -m pytest tests
