import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "vendor"))

from server.detector import HeadDetector
from server.tracker import HeadTracker
from server.pipeline import HeartRatePipeline, BUFFER_SIZE

WEIGHTS_PATH = BASE / "weights/yolor_head.pt"
VIDEO_PATH = Path("/home/safi/heart_rate_detection/demo/baby.mp4")


@pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="Weights not present — copy from heart_rate_detection/weights/ or re-download"
)
def test_neonatal_bpm_in_expected_range():
    detector = HeadDetector(
        cfg_path=str(BASE / "config/yolor_p6_head.cfg"),
        weights_path=str(WEIGHTS_PATH),
        device="cpu",
    )
    tracker = HeadTracker(config_path=str(BASE / "config/deep_sort.yaml"))
    pipeline = HeartRatePipeline(detector=detector, tracker=tracker,
                                  fps=30.0, mode="neonate")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    bpm_readings = []

    for _ in range(BUFFER_SIZE + 30):
        ret, frame = cap.read()
        if not ret:
            break
        for result in pipeline.process_frame(frame):
            if result.confidence > 0.1:
                bpm_readings.append(result.bpm)

    cap.release()

    assert len(bpm_readings) > 0, "No BPM readings produced — check head detection"
    avg_bpm = sum(bpm_readings) / len(bpm_readings)
    assert 100 <= avg_bpm <= 175, f"BPM {avg_bpm:.1f} outside expected neonatal range"
