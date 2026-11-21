ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

.PHONY: install baseline evaluate app test reproduce

install:
	python -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -v

baseline:
	$(PYTHON) -m src.baseline

evaluate:
	$(PYTHON) -m src.evaluate

app:
	$(PYTHON) -m streamlit run app/app.py

reproduce: baseline evaluate
