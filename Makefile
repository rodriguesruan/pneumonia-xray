.PHONY: audit split train train-debug eval-val eval-final app test lint format

audit:
	python -m src.data.audit

split:
	python -m src.data.make_splits

train:
	python -m src.training.train

train-debug:
	python -m src.training.train --debug

eval-val:
	python -m src.evaluation.run_evaluation --checkpoint "$(CHECKPOINT)" --split val

eval-final:
	python -m src.evaluation.run_evaluation --checkpoint "$(CHECKPOINT)" --split test --final-test

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest

lint:
	python -m ruff check src tests
	python -m black --check src tests

format:
	python -m ruff check --fix src tests
	python -m black src tests
