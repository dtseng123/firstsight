# Backend

Minimal backend scaffold for the planned smart-glasses realtime agent platform.

## Purpose

- expose a simple FastAPI service teammates can run immediately
- provide a Vision Agents starter example with config-based provider selection
- establish folders for future processors, tools, and RAG wiring

## Public Entry Points

- `app.main:app`
- `app.main:create_app`
- `app.examples.basic_video_agent`

## Current HTTP / WebSocket Surfaces

- `GET /health`
- `GET /bootstrap`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/frame`
- `WS /sessions/{session_id}/stream`

Important:

- the session WebSocket is live and tested for app-to-backend ingest
- realtime mode forwards live media into the Vision Agents realtime provider bridge
- the viewer can poll session state and fetch the latest annotated preview frame
- the Fast Whisper pipeline path still exists, but the quickest demo path is Gemini realtime plus the pose overlay processor

## Minimal Example

From the `backend/` directory:

```bash
make setup
make dev
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Vision Agents starter:

```bash
make example
```

Tests:

```bash
make test
```

Live ingest smoke test:

```bash
make smoke-stream
```

## Observability (Logfire)

The backend is instrumented with [Pydantic Logfire](https://logfire.pydantic.dev) for tracing FastAPI requests and pydantic-ai agent calls.

### First-time setup

1. Create an org at [logfire.pydantic.dev](https://logfire.pydantic.dev)
2. Authenticate the CLI:

```bash
pip install logfire
logfire auth
```

3. Create a project inside your org (one-time):

```bash
logfire projects new droopdetection
```

4. On first `make dev`, logfire will pick up the linked project and start sending traces automatically.

If no project is linked, logfire falls back to a local no-op mode — the backend still runs, you just won't see traces.

## Notes

- Use Python `3.11` to `3.13`. `vision-agents` currently pulls native dependencies that are not smooth on Python `3.14`.
- The FastAPI service is intentionally lightweight and does not require a live Stream or model session to boot.
- The Vision Agents example is the starting point for the realtime backend, not the finished architecture.
- The face droop processor is currently a scaffold seam for your real model.
- The pose overlay processor uses the Vision Agents ultralytics plugin and stores the latest annotated JPEG in memory for the viewer.
- `make clean` removes the backend virtualenv and Python caches.
