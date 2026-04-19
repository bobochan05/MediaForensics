## Tracelyt Handoff Context

Date: 2026-04-14

This file captures the current project state so work can continue smoothly in a new chat.

### Current high-priority user request

The user wants:

1. The Gemini model to be visible to the user in the right-side UI.
2. The user to be able to interact with the Gemini model directly from the dashboard.
3. The `Related Web Sources` section to show a fuller set of public leads, with the user's expectation being around 5 visible matches when enough leads exist.

### Current status

#### Gemini model visibility

- The right-side `Investigation Snapshot` panel already shows:
  - `AI Provider`
  - `Model`
- Backend already passes these values into the template.

Relevant files:

- `backend/app.py`
- `backend/templates/dashboard.html`

#### Gemini chat backend

Backend support already exists.

Routes / logic already present:

- `/api/chat`
- `/api/agent`
- `_agent_system_prompt()`
- `_build_agent_context(...)`
- `_build_gemini_payload(...)`
- `_generate_agent_reply(...)`
- `_stream_agent_reply(...)`
- `_agent_query_in_scope(...)`
- `_agent_policy_reply(...)`

This means the backend can already answer scoped questions using Gemini, with fallback behavior if Gemini is unavailable.

#### Missing piece

There is currently **no visible chat UI** in the dashboard for users to interact with Gemini.

The previous AI assistant panel was removed earlier and replaced with `Investigation Snapshot`.

### Related Web Sources issue

The user reported that `Related Web Sources` is too low or empty even when public leads exist.

Important finding:

- `Top Matches` currently consumes strong exact/visual matches.
- `Related Web Sources` only renders the embedding/related bucket.
- Because of that, the UI may show only 1 related source even when more public leads exist overall.

Backend fallback logic was already improved to top up related sources in some cases, but the frontend still may not surface enough of them clearly.

Relevant file:

- `backend/app.py`
- `backend/templates/dashboard.html`

### Latest inspected UI state

Right rail currently contains:

- `Investigation Snapshot`
- Provider/model information

What is missing:

- a visible Gemini chat box
- quick question buttons
- message history
- user input box
- send button

### Recommended next implementation steps

1. Restore a compact Gemini chat panel **under** `Investigation Snapshot` in the right rail.
2. Wire that chat UI to `/api/agent`.
3. Send current Layer 1 / Layer 2 / Layer 3 data as context.
4. Add simple quick prompts:
   - `Explain result`
   - `Why is this fake?`
   - `Where did this come from?`
   - `Explain risk`
5. Keep answers scoped to Tracelyt analysis only.
6. Top up `Related Web Sources` in the UI so at least the first 5 public leads are visible when available.

### Important recent fixes already completed

- Duplicate top-bar/session info was removed from the right side.
- Gemini provider/model display was added to the right-side snapshot.
- `_safe_float(...)` crash in `backend/app.py` was fixed.
- duplicate Gemini env entries were cleaned.
- multiple Layer 1 CLIP compatibility issues were patched.
- Layer 2/Layer 3 progressive rendering and enrichment were improved.

### Current files most relevant for the next chat

- `C:\Users\soham\Desktop\googlesolution\MediaForensics\backend\templates\dashboard.html`
- `C:\Users\soham\Desktop\googlesolution\MediaForensics\backend\app.py`
- `C:\Users\soham\Desktop\googlesolution\MediaForensics\.env`

### User preference notes

- The user wants direct, practical fixes.
- The user prefers visible working UI over backend-only wiring.
- The user specifically wants the Gemini model to be both visible and interactive.
- The user expects the dashboard to feel complete from the user perspective, not just technically configured.
