.PHONY: help setup setup-llm setup-api sample ingest run query eval test clean

help:
	@echo "Targets:"
	@echo "  setup      Install core (free, offline) dependencies"
	@echo "  setup-llm  Install optional local-LLM extras (sentence-transformers, flan-t5)"
	@echo "  setup-api  Install optional cloud API extras (OpenAI/Anthropic)"
	@echo "  sample     Generate synthetic sample business reports"
	@echo "  ingest     Build the FAISS index from data/reports"
	@echo "  run        Launch the Streamlit app"
	@echo "  query      Ask one question from the CLI (Q=\"...\")"
	@echo "  eval       Run the retrieval+grounding evaluation"
	@echo "  test       Run the test suite"
	@echo "  clean      Remove the built index and caches"

setup:
	pip install -r requirements.txt

setup-llm:
	pip install -r requirements-llm.txt

setup-api:
	pip install -r requirements-api.txt

sample:
	python scripts/generate_sample_reports.py

ingest:
	python scripts/ingest.py

run:
	streamlit run app.py

query:
	python scripts/query.py "$(Q)"

eval:
	python scripts/evaluate.py

test:
	pytest -q

clean:
	rm -rf data/index/* .pytest_cache **/__pycache__ eval/results*.json eval/results*.csv
