# `backend/app/stt/AGENTS.md`

Purpose:
- Local speech-to-text adapters and wrappers for backend-owned voice sessions.

Rules:
- Prefer explicit, configurable buffering over hidden provider defaults.
- Keep provider-specific behavior isolated here so the bridge layer stays transport-focused.
