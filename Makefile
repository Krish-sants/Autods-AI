.PHONY: backend-install backend-dev backend-test frontend-install frontend-dev dev docker-up docker-down

backend-install:
	cd backend && python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

backend-dev:
	cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

backend-test:
	cd backend && .venv/Scripts/python -m pytest tests/ -v

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

dev:
	@echo "Run 'make backend-dev' and 'make frontend-dev' in two separate terminals."

docker-up:
	docker compose up --build

docker-down:
	docker compose down
