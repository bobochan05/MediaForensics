# Deepfake Detection & Intelligence Console

Production-oriented monorepo for:

- Flask backend APIs and auth
- AI Layer 1/2/3 pipelines
- Next.js dashboard frontend

## Project Structure

```text
backend/                  Flask app, auth, routes, schemas, services
ai/                       Layer 1, Layer 2, Layer 3, shared ML utilities
frontend/dashboard-app/   Next.js dashboard
data/                     Local datasets and embeddings (ignored in git)
scripts/                  Training and evaluation scripts
tests/                    Automated tests
docs/                     Architecture and technical docs
```

## Quick Start

1. Install backend dependencies

```powershell
python -m pip install -r requirements.txt
```

2. Run backend

```powershell
python app.py
```

3. Run frontend (optional dashboard app)

```powershell
cd frontend/dashboard-app
npm install
npm run dev
```

## Deploy (Render)

This repo’s Flask WSGI app is exposed as `backend.app:app`.

- **Start command**: `gunicorn backend.app:app --bind 0.0.0.0:$PORT`
- If you use Render’s native Python runtime, the included `Procfile` already defines this.

If your deploy logs show `ModuleNotFoundError: No module named 'your_application'`, Render is still using a placeholder start command (common in tutorials). Update your Render service **Settings → Start Command** to the command above (or clear it and rely on the `Procfile`), then redeploy.

## Main Endpoints

- `POST /api/analyze`
- `POST /api/chat`
- `POST /api/predict`
- `POST /api/discover`
- `POST /reverse-search`

## Docs

- Technical architecture: [`docs/TECHNICAL_README.md`](docs/TECHNICAL_README.md)
- Layer notes:
  - [`docs/layers/LAYER1_README.md`](docs/layers/LAYER1_README.md)
  - [`docs/layers/LAYER2_README.md`](docs/layers/LAYER2_README.md)
