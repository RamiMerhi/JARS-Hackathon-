# Complaint Triage System - Full Project

This package contains:

- `backend/` - FastAPI backend with local ML models.
- `frontend/` - React/TanStack/Vite frontend.
- `data/` - Full complaints and establishments CSV files.

## Backend

From the project root:

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Open:

- http://localhost:8000/docs
- http://localhost:8000/api/summary
- http://localhost:8000/api/model-info

The backend uses local ML only. It does not use Claude/OpenAI/Gemini APIs.

## Frontend

Open a second terminal from the project root:

```cmd
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, usually:

- http://localhost:5173

The frontend calls the backend at `http://localhost:8000` by default.

## If backend is on another URL

Run frontend with:

```cmd
set VITE_API_BASE=http://localhost:8000
npm run dev
```

## What was verified

Backend endpoints tested:

- `GET /api/summary`
- `GET /api/model-info`
- `GET /api/complaints`
- `POST /api/triage`

The backend trains/loads:

- Category classifier from complaint labels.
- Priority classifier from weak-supervised AI priority labels.
- Mandatory Ministry zone rule layer after model prediction.

## Note

`node_modules/` and Python `.venv/` are not included in this zip. Install them locally using the commands above.
