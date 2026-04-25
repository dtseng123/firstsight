from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image
from vision_agents.plugins.ultralytics import YOLOPoseProcessor

from ..session_manager import session_manager


@dataclass(slots=True)
class PoseOverlaySignal:
    score: float
    threshold: float
    over_threshold: bool
    message: str
    person_count: int


class PoseOverlayProcessor(YOLOPoseProcessor):
    name = "yolo_pose_overlay"

    def __init__(
        self,
        *,
        session_id: str,
        model_path: str,
        conf_threshold: float,
        device: str,
        fps: int,
        enable_hand_tracking: bool,
    ) -> None:
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            fps=fps,
            enable_hand_tracking=enable_hand_tracking,
        )
        self.session_id = session_id
        self.latest_signal = PoseOverlaySignal(
            score=0.0,
            threshold=conf_threshold,
            over_threshold=False,
            message="Pose processor initialized. Waiting for video frames.",
            person_count=0,
        )

    async def add_pose_to_ndarray(
        self,
        frame_array: Any,
    ) -> tuple[Any, dict[str, Any]]:
        annotated_array, pose_data = await super().add_pose_to_ndarray(frame_array)

        persons = pose_data.get("persons", []) if isinstance(pose_data, dict) else []
        person_count = len(persons)
        confidences = [
            float(person.get("confidence", 0.0))
            for person in persons
            if isinstance(person, dict)
        ]
        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )
        self.latest_signal = PoseOverlaySignal(
            score=avg_confidence,
            threshold=self.conf_threshold,
            over_threshold=person_count > 0 and avg_confidence >= self.conf_threshold,
            message=(
                f"Detected {person_count} person(s) with average confidence "
                f"{avg_confidence:.2f}."
                if person_count
                else "No pose detected in the latest frame."
            ),
            person_count=person_count,
        )
        session_manager.update_processor_signal(
            self.session_id,
            self.name,
            {
                "name": self.name,
                "score": avg_confidence,
                "threshold": self.conf_threshold,
                "over_threshold": person_count > 0 and avg_confidence >= self.conf_threshold,
                "message": self.latest_signal.message,
                "person_count": person_count,
            },
        )

        output = io.BytesIO()
        Image.fromarray(annotated_array).save(output, format="JPEG", quality=80)
        session_manager.update_preview_frame(
            self.session_id,
            output.getvalue(),
            mime_type="image/jpeg",
        )
        return annotated_array, pose_data
