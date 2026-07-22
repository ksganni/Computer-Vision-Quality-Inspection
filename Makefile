.PHONY: setup test api ui docker-up docker-down sample evaluate

setup:
	rm -rf .venv
	python3.12 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements-dev.txt
	. .venv/bin/activate && python scripts/create_sample_image.py
	. .venv/bin/activate && python scripts/download_model.py || true

test:
	. .venv/bin/activate && pytest

api:
	. .venv/bin/activate && uvicorn app.main:app --reload --port 8000

ui:
	. .venv/bin/activate && streamlit run streamlit_app/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

sample:
	. .venv/bin/activate && python scripts/create_sample_image.py

evaluate:
	. .venv/bin/activate && python scripts/evaluate.py
