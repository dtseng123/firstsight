from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    realtime_provider: Literal["gemini", "openai"] = "gemini"
    speech_pipeline: Literal["realtime", "fast_whisper_pipeline"] = "realtime"
    realtime_video_fps: int = 10
    processor_fps: int = 10
    enable_face_droop_processor: bool = True
    droop_model_path: str = "../model/droop_model.onnx"
    droop_threshold_path: str = "../checkpoints/threshold.json"
    droop_face_landmarker_path: str = "../model/face_landmarker.task"
    droop_image_size: int = 224
    enable_pose_processor: bool = False
    pose_model_path: str = "yolo11n-pose.pt"
    pose_conf_threshold: float = 0.5
    pose_device: Literal["cpu", "cuda"] = "cpu"
    pose_enable_hand_tracking: bool = True
    stream_api_key: str = ""
    stream_api_secret: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3-flash-preview"
    fast_whisper_model_size: str = "small.en"
    fast_whisper_language: str = "en"
    fast_whisper_device: Literal["cpu", "cuda"] = "cpu"
    fast_whisper_min_buffer_ms: int = 400
    fast_whisper_process_interval_ms: int = 800
    fast_whisper_max_buffer_ms: int = 3000
    pipeline_turn_delay_ms: int = 1200
    backend_tts_enabled: bool = True
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "VR6AewLTigWG4xSOukaG"
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    enable_heart_rate_processor: bool = False
    heart_rate_mode: str = "adult"
    heart_rate_fps: float = 10.0

    rag_enabled: bool = False
    rag_index_dir: str = "rag_index"
    rag_top_k: int = 6
    rag_epub_path: str = "../data/jrcalc-clinical-guidelines-2022.epub"
    rag_max_context_tokens: int = 1200

    agent_name: str = "DroopDetection Agent"
    agent_user_id: str = "agent"
    agent_instructions: str = (
        "You are a guidance-only first-aid assistant. Use visible evidence, "
        "processor signals, and retrieved guidance to help the wearer assess "
        "urgent situations. Never pretend to place calls or take external "
        "actions on the user's behalf."
    )


    def droop_model_path_resolved(self) -> str:
        return str(Path(self.droop_model_path).resolve())

    def droop_threshold_path_resolved(self) -> str:
        return str(Path(self.droop_threshold_path).resolve())

    def droop_face_landmarker_path_resolved(self) -> str:
        return str(Path(self.droop_face_landmarker_path).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
