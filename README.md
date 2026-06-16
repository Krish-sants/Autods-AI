# AutoDS-AI (Phase 1)

An autonomous data-science agent. Upload a dataset (CSV/Excel/JSON/Parquet) or paste a URL, confirm the target
column when asked, and AutoDS-AI automatically profiles, cleans, engineers features, trains and compares 5
models, evaluates the best one, explains it with SHAP, and generates a report — no code or statistics required.

## What's in Phase 1

- **Pipeline**: understanding -> EDA -> cleaning -> target detection (human confirms) -> feature engineering ->
  training (5 models per problem type) -> evaluation -> SHAP explainability -> report.
- **Problem types**: classification, regression, and a KMeans fallback when no target is confirmed.
- **LLM**: Google Gemini (free tier) for narrative text — dataset description, model explanation, executive
  summary. Works with zero API keys too; narrative text falls back to deterministic templates.
- **Stack**: FastAPI + LangGraph + SQLite (backend), Next.js + Tailwind + Plotly (frontend).

Explicitly **not** in Phase 1 (see the implementation plan for the full list): time-series forecasting, full
Optuna hyperparameter search, Kubernetes/multi-cloud deployment, PDF/DOCX export, auto-generated predict
microservices, deep learning models.

## Running locally

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # or .venv/bin/python on macOS/Linux
cp .env.example .env   # optionally add GOOGLE_API_KEY from https://aistudio.google.com/apikey
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000, upload a dataset, confirm the target column when prompted, and explore the results
dashboard (EDA, leaderboard, metrics, feature importance, SHAP, report) once the run completes.

### Tests

```bash
cd backend
.venv/Scripts/python -m pytest tests/ -v
```

### Docker

```bash
docker compose up --build
```

## Project layout

See `backend/app/` for the FastAPI app (`api/`, `graph/`, `agents/`, `ml/`, `database/`, `storage/`) and
`frontend/` for the Next.js app (`app/`, `components/`, `lib/`).
