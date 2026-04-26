# STT

Purpose:
- Speech-to-text adapters used by the backend voice pipeline.

Entrypoints:
- `fast_whisper_live.py`

Example:
```python
from app.stt.fast_whisper_live import FastWhisperLiveSTT

stt = FastWhisperLiveSTT(model_size="base", language="en", device="cpu")
```

How to test:
- `make backend-test`
- run the backend and verify `/sessions/{id}` shows `input_transcription` debug events while the session is still live
