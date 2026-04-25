import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "heart_rate_detection"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.detector import HeadDetector
from server.tracker import HeadTracker
from server.pipeline import HeartRatePipeline, BUFFER_SIZE

WEIGHTS_BASE = Path("/home/safi/heart_rate_detection")
VIDEO_PATH = WEIGHTS_BASE / "demo/baby.mp4"


@pytest.mark.skipif(
    not WEIGHTS_BASE.exists(),
    reason="heart_rate_detection project not downloaded"
)
def test_neonatal_bpm_in_expected_range():
    detector = HeadDetector(
        cfg_path=str(WEIGHTS_BASE / "config/yolor_p6_head.cfg"),
        weights_path=str(WEIGHTS_BASE / "weights/yolor_head.pt"),
        device="cpu",
    )
    tracker = HeadTracker(config_path=str(WEIGHTS_BASE / "config/deep_sort.yaml"))
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
                print(f"BPM: {result.bpm}, confidence: {result.confidence}")

    cap.release()

    assert len(bpm_readings) > 0, "No BPM readings were produced"
    avg_bpm = sum(bpm_readings) / len(bpm_readings)
    print(f"Average BPM: {avg_bpm:.1f}")
    assert 100 <= avg_bpm <= 175, f"BPM {avg_bpm:.1f} outside expected neonatal range"
