# Mutation Picker – common tasks
# Usage: make <target>
# Requires: Python venv at .venv/  (make install creates it)

PYTHON = .venv/Scripts/python.exe
PIP    = .venv/Scripts/pip.exe

.PHONY: install baseline evaluate app test reproduce

test:
	$(PYTHON) -m pytest -v


install:
	python -m venv .venv
	$(PIP) install -r requirements.txt

baseline:
	$(PYTHON) -m src.baseline

evaluate:
	$(PYTHON) -m src.evaluate

app:
	.venv/Scripts/streamlit.exe run app/app.py

# Regenerates every artifact from raw data in order.
reproduce: baseline evaluate
	@echo "All results regenerated."
