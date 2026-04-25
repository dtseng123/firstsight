import sys
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "heart_rate_detection"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.detector import HeadDetector
from server.tracker import HeadTracker
from server.pipeline import HeartRatePipeline

BASE = Path(__file__).parent.parent.parent / "heart_rate_detection"

app = FastAPI()


def build_pipeline(mode: str = "adult") -> HeartRatePipeline:
    detector = HeadDetector(
        cfg_path=str(BASE / "config/yolor_p6_head.cfg"),
        weights_path=str(BASE / "weights/yolor_head.pt"),
        device="cpu",
    )
    tracker = HeadTracker(config_path=str(BASE / "config/deep_sort.yaml"))
    return HeartRatePipeline(detector=detector, tracker=tracker, mode=mode)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    mode = websocket.query_params.get("mode", "adult")
    pipeline = build_pipeline(mode=mode)

    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            for result in pipeline.process_frame(frame):
                await websocket.send_text(json.dumps({
                    "track_id": result.track_id,
                    "bpm": result.bpm,
                    "confidence": result.confidence,
                    "alert": result.alert,
                }))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
