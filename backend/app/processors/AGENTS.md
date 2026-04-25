# `processors/AGENTS.md`

Scope:
- backend demo processors
- structured processor signals
- viewer preview frame generation

Rules:
- keep processors small and session-scoped
- prefer emitting simple numeric or boolean signals alongside human-readable messages
- if a processor powers the viewer, publish only the latest preview frame in memory rather than storing histories
- avoid introducing extra network services when a local processor plugin already exists
