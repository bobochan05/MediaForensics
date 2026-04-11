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
