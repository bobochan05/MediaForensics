# Tracelyt Dashboard App

Production-oriented Next.js dashboard for post-login intelligence workflows.

## Run

```bash
npm install
npm run dev
```

## Routes

- `/dashboard` - protected dashboard workspace
- `/login` - fallback when auth cookie is missing

## Stack

- Next.js 14 + React 18
- Tailwind CSS
- Framer Motion
- Recharts
- Zustand

## Backend contract

- `GET /api/auth/session`
- `POST /api/analyze`

## Notes

Set `NEXT_PUBLIC_USE_DUMMY_DATA=true` to test the UI without backend analysis responses.

### Required env (Firebase/Render)

- `NEXT_PUBLIC_API_BASE_URL` = your backend base URL (Render), e.g. `https://YOUR-SERVICE.onrender.com`
- `NEXT_PUBLIC_AUTH_ENTRY_URL` (optional) = where the “Go To Login” button should send users; defaults to `NEXT_PUBLIC_API_BASE_URL`
