from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_processors(settings: Settings, *, session_id: str | None = None) -> list[object]:
    processors: list[object] = []
    if settings.enable_pose_processor:
        from .processors.pose_overlay import PoseOverlayProcessor

        processors.append(
            PoseOverlayProcessor(
                session_id=session_id or "preview",
                model_path=settings.pose_model_path,
                conf_threshold=settings.pose_conf_threshold,
                device=settings.pose_device,
                fps=settings.processor_fps,
                enable_hand_tracking=settings.pose_enable_hand_tracking,
            )
        )
    if settings.enable_face_droop_processor:
        from .processors.face_droop import FaceDroopProcessor

        processors.append(
            FaceDroopProcessor(
                session_id=session_id or "preview",
                fps=settings.processor_fps,
                model_path=settings.droop_model_path_resolved(),
                threshold_path=settings.droop_threshold_path_resolved(),
                face_landmarker_path=settings.droop_face_landmarker_path_resolved(),
                image_size=settings.droop_image_size,
            )
        )
    if settings.enable_heart_rate_processor:
        from .processors.heart_rate import HeartRateProcessor

        processors.append(
            HeartRateProcessor(
                fps=settings.heart_rate_fps,
                mode=settings.heart_rate_mode,
            )
        )
    return processors


def build_realtime_llm(settings: Settings) -> object:
    if settings.realtime_provider == "openai":
        from vision_agents.plugins import openai

        return openai.Realtime(
            fps=settings.realtime_video_fps,
            api_key=settings.openai_api_key or None,
        )

    from vision_agents.plugins import gemini

    return gemini.Realtime(
        fps=settings.realtime_video_fps,
        api_key=settings.gemini_api_key or None,
    )


def build_text_llm(settings: Settings) -> object:
    from vision_agents.plugins import gemini

    return gemini.LLM(
        model=settings.gemini_llm_model,
        api_key=settings.gemini_api_key or None,
    )


def build_stt(settings: Settings) -> object:
    from .stt.fast_whisper_live import FastWhisperLiveSTT

    return FastWhisperLiveSTT(
        model_size=settings.fast_whisper_model_size,
        language=settings.fast_whisper_language,
        device=settings.fast_whisper_device,
        min_buffer_duration_ms=settings.fast_whisper_min_buffer_ms,
        process_interval_ms=settings.fast_whisper_process_interval_ms,
        max_buffer_duration_ms=settings.fast_whisper_max_buffer_ms,
    )


def build_tts(settings: Settings) -> object | None:
    if not settings.backend_tts_enabled or not settings.elevenlabs_api_key:
        return None

    from vision_agents.plugins import elevenlabs

    return elevenlabs.TTS(
        api_key=settings.elevenlabs_api_key or None,
        voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model_id,
    )


def _build_embed_fn(settings: Settings) -> Callable[[list[str]], list[list[float]]]:
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    return embed


def build_retriever(settings: Settings) -> object | None:
    if not settings.rag_enabled:
        return None
    try:
        from .rag.graph import ClinicalGraph
        from .rag.index import FaissIndex
        from .rag.retriever import GraphRetriever

        embed_fn = _build_embed_fn(settings)
        graph = ClinicalGraph.load(settings.rag_index_dir)
        index = FaissIndex.load(settings.rag_index_dir)
        logger.info(
            "rag retriever loaded index_dir=%s nodes=%s vectors=%s top_k=%s",
            settings.rag_index_dir,
            graph.node_count(),
            index.total(),
            settings.rag_top_k,
        )
        return GraphRetriever(graph, index, embed_fn, top_k=settings.rag_top_k)
    except Exception:
        logger.exception("rag retriever failed to load index_dir=%s", settings.rag_index_dir)
        return None


def build_agent(settings: Settings | None = None) -> object:
    active_settings = settings or get_settings()

    from vision_agents.core import Agent, User
    from vision_agents.plugins import getstream

    processors: Sequence[object] = _build_processors(active_settings)
    if active_settings.speech_pipeline == "fast_whisper_pipeline":
        return Agent(
            edge=getstream.Edge(),
            agent_user=User(name=active_settings.agent_name, id=active_settings.agent_user_id),
            instructions=active_settings.agent_instructions,
            llm=build_text_llm(active_settings),
            stt=build_stt(active_settings),
            tts=build_tts(active_settings),
            processors=list(processors),
        )

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name=active_settings.agent_name, id=active_settings.agent_user_id),
        instructions=active_settings.agent_instructions,
        llm=build_realtime_llm(active_settings),
        processors=list(processors),
    )
