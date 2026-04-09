.PHONY: setup test download-data

VENV := .venv
PYTHON := $(VENV)/bin/python

setup:
	python3.12 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e ".[notebook,dev]"

test:
	$(PYTHON) -m pytest tests

download-data:
	$(PYTHON) -m gdown --folder https://drive.google.com/drive/folders/1eveEvA5lljsbXcC9sEiPM4KOmGUw7StS -O . --remaining-ok
