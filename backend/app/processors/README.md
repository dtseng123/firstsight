# `processors/README.md`

Purpose:
- hold demo and custom video processors used by the backend bridges
- expose structured signals for the app and viewer
- optionally publish annotated preview frames for the React dashboard

Entrypoints:
- `face_droop.py` - placeholder seam for the stroke droopiness model
- `pose_overlay.py` - Vision Agents ultralytics pose overlay processor for the realtime demo

Example:
```python
from app.processors.pose_overlay import PoseOverlayProcessor

processor = PoseOverlayProcessor(
    session_id="demo",
    model_path="yolo11n-pose.pt",
    conf_threshold=0.5,
    device="cpu",
    fps=10,
    enable_hand_tracking=True,
)
```

How to test:
- run `make backend-test`
- start the backend in realtime mode
- stream video from the Android app
- open the viewer and confirm the preview frame and pose signal update together
