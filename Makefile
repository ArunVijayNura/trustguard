.PHONY: install test lint train evaluate api dashboard docker

install:
	pip install -e ".[dev,dashboard]"

test:
	pytest -q

lint:
	ruff check .

train:
	trustguard train

evaluate:
	trustguard evaluate

api:
	uvicorn trustguard.api:app --reload

dashboard:
	streamlit run dashboard/app.py

docker:
	docker build -t trustguard . && docker run -p 8000:8000 trustguard
