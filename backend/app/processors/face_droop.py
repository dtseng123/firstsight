from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vision_agents.core.processors import VideoProcessor

from ..session_manager import session_manager

if TYPE_CHECKING:
    import aiortc
    from av import VideoFrame
    from vision_agents.core import Agent
    from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceDroopSignal:
    score: float
    threshold: float
    over_threshold: bool
    message: str
    severity: str
    face_detected: bool
    asymmetry_score: float | None


class FaceDroopProcessor(VideoProcessor):
    name = "face_droop_processor"

    def __init__(
        self,
        *,
        session_id: str,
        fps: int = 2,
        model_path: str,
        threshold_path: str,
        face_landmarker_path: str,
        image_size: int = 224,
    ) -> None:
        self.fps = fps
        self.session_id = session_id
        self._model_path = model_path
        self._threshold_path = threshold_path
        self._face_landmarker_path = face_landmarker_path
        self._image_size = image_size
        self._model = None
        self._agent: Agent | None = None
        self._forwarder: VideoForwarder | None = None
        self.latest_signal = FaceDroopSignal(
            score=0.0,
            threshold=0.5,
            over_threshold=False,
            message="Waiting for first frame.",
            severity="none",
            face_detected=False,
            asymmetry_score=None,
        )

    def _get_model(self):
        if self._model is None:
            from pathlib import Path
            from .droop_inference import DroopModel

            if not Path(self._model_path).exists():
                raise FileNotFoundError("ONNX model not found")
            if not Path(self._threshold_path).exists():
                raise FileNotFoundError("Threshold file not found")
            self._model = DroopModel(
                model_path=self._model_path,
                threshold_path=self._threshold_path,
                image_size=self._image_size,
                face_landmarker_path=self._face_landmarker_path,
            )
        return self._model

    def attach_agent(self, agent: "Agent") -> None:
        self._agent = agent

    async def process_video(
        self,
        track: "aiortc.VideoStreamTrack",
        participant_id: str | None,
        shared_forwarder: "VideoForwarder" | None = None,
    ) -> None:
        del track
        del participant_id
        self._forwarder = shared_forwarder
        if self._forwarder is None:
            return

        self._forwarder.add_frame_handler(
            self._handle_frame,
            fps=float(self.fps),
            name=self.name,
        )

    async def stop_processing(self) -> None:
        if self._forwarder is not None:
            await self._forwarder.remove_frame_handler(self._handle_frame)
        self._forwarder = None

    async def close(self) -> None:
        return None

    def _predict_sync(self, jpeg_bytes: bytes) -> dict:
        return self._model.predict(jpeg_bytes)  # type: ignore[union-attr]

    async def _handle_frame(self, frame: "VideoFrame") -> None:
        try:
            model = self._get_model()
        except FileNotFoundError as exc:
            logger.warning("FaceDroopProcessor: model files missing — %s", exc)
            self.latest_signal = FaceDroopSignal(
                score=0.0,
                threshold=0.5,
                over_threshold=False,
                message="Face droop model unavailable. Contact administrator.",
                severity="none",
                face_detected=False,
                asymmetry_score=None,
            )
            session_manager.update_processor_signal(
                self.session_id, self.name, self._signal_dict()
            )
            return

        try:
            buf = io.BytesIO()
            frame.to_image().save(buf, format="JPEG", quality=85)
            result = await asyncio.to_thread(self._predict_sync, buf.getvalue())
        except Exception:
            logger.exception("FaceDroopProcessor: inference failed")
            return

        face_detected: bool = result.get("face_detected", False)
        prob: float = result.get("droop_probability") or 0.0
        is_drooping: bool = result.get("is_drooping") or False
        severity: str = result.get("severity") or "none"
        asymmetry = result.get("asymmetry_score")

        if not face_detected:
            message = "No face detected in frame."
        elif is_drooping:
            message = f"Droop detected — severity: {severity}, probability: {prob:.2f}."
        else:
            message = f"No droop detected. Probability: {prob:.2f}."

        self.latest_signal = FaceDroopSignal(
            score=prob,
            threshold=model.threshold,
            over_threshold=is_drooping,
            message=message,
            severity=severity,
            face_detected=face_detected,
            asymmetry_score=round(asymmetry, 4) if asymmetry is not None else None,
        )
        session_manager.update_processor_signal(
            self.session_id, self.name, self._signal_dict()
        )

    def _signal_dict(self) -> dict:
        s = self.latest_signal
        return {
            "name": self.name,
            "score": s.score,
            "threshold": s.threshold,
            "over_threshold": s.over_threshold,
            "message": s.message,
            "severity": s.severity,
            "face_detected": s.face_detected,
            "asymmetry_score": s.asymmetry_score,
        }
