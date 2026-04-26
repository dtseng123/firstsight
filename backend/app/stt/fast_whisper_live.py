from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from getstream.video.rtc.track_util import AudioFormat, PcmData
from numpy.typing import NDArray
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import TranscriptResponse
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

RATE = 16000


class FastWhisperLiveSTT(stt.STT, Warmable[Optional[WhisperModel]]):
    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large"] = "base",
        language: Optional[str] = "en",
        device: Literal["cpu", "cuda"] = "cpu",
        min_buffer_duration_ms: int = 400,
        process_interval_ms: int = 800,
        max_buffer_duration_ms: int = 3000,
        client: Optional[WhisperModel] = None,
    ) -> None:
        super().__init__(provider_name="faster_whisper_live")
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = "int8"
        self.min_buffer_duration_ms = min_buffer_duration_ms
        self.process_interval_ms = process_interval_ms
        self.max_buffer_duration_ms = max_buffer_duration_ms
        self.whisper = client
        self._audio_buffer = PcmData(
            sample_rate=RATE,
            channels=1,
            format=AudioFormat.F32,
        )
        self._last_process_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._buffer_lock = asyncio.Lock()

    async def on_warmup(self) -> Optional[WhisperModel]:
        if self.whisper is None:
            logger.info("Loading faster-whisper model=%s device=%s", self.model_size, self.device)
            loop = asyncio.get_running_loop()
            whisper = await loop.run_in_executor(
                self._executor,
                lambda: WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                ),
            )
            logger.info("Faster-whisper loaded model=%s", self.model_size)
            return whisper
        return None

    def on_warmed_up(self, whisper: Optional[WhisperModel]) -> None:
        if self.whisper is None:
            self.whisper = whisper

    async def process_audio(self, pcm_data: PcmData, participant: Participant) -> None:
        if self.closed:
            return
        if self.whisper is None:
            raise ValueError("Whisper model not loaded, call warmup() first")
        if pcm_data.samples.size == 0:
            return

        try:
            audio_data = pcm_data.resample(RATE).to_float32()
            async with self._buffer_lock:
                self._audio_buffer = self._audio_buffer.append(audio_data)
                current_time = time.time()
                buffer_duration_ms = self._audio_buffer.duration_ms
                time_since_last_process = (current_time - self._last_process_time) * 1000
                should_process = (
                    buffer_duration_ms >= self.min_buffer_duration_ms
                    and (
                        time_since_last_process >= self.process_interval_ms
                        or buffer_duration_ms >= self.max_buffer_duration_ms
                    )
                )

            if should_process:
                await self._process_buffer(participant)
        except Exception as exc:
            logger.exception("Error buffering audio for faster-whisper live")
            self._emit_error_event(exc, context="buffering_audio", participant=participant)

    async def flush(self, participant: Participant) -> None:
        async with self._buffer_lock:
            has_audio = self._audio_buffer.samples.size > 0
        if has_audio:
            await self._process_buffer(participant)

    async def _process_buffer(self, participant: Participant) -> None:
        async with self._buffer_lock:
            buffer_to_process = self._audio_buffer
            if buffer_to_process.samples.size == 0:
                return
            self._audio_buffer = PcmData(
                sample_rate=RATE,
                channels=1,
                format=AudioFormat.F32,
            )
            self._last_process_time = time.time()

        pcm = buffer_to_process.resample(RATE).to_float32()
        audio_array = pcm.samples
        if audio_array.size == 0:
            return

        start_time = time.time()
        try:
            segments, info = await self._transcribe(audio_array=audio_array)
        except Exception as exc:
            logger.exception("Error transcribing faster-whisper live buffer")
            self._emit_error_event(exc, context="transcription", participant=participant)
            return

        processing_time_ms = (time.time() - start_time) * 1000
        text_parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            text_parts.append(text)
            response = TranscriptResponse(
                confidence=getattr(segment, "avg_logprob", None),
                language=getattr(info, "language", self.language),
                processing_time_ms=processing_time_ms,
                audio_duration_ms=buffer_to_process.duration_ms,
                model_name=f"faster-whisper-{self.model_size}",
            )
            self._emit_partial_transcript_event(text, participant, response)

        if text_parts:
            full_text = " ".join(text_parts).strip()
            response = TranscriptResponse(
                confidence=None,
                language=getattr(info, "language", self.language),
                processing_time_ms=processing_time_ms,
                audio_duration_ms=buffer_to_process.duration_ms,
                model_name=f"faster-whisper-{self.model_size}",
            )
            self._emit_transcript_event(full_text, participant, response)

    async def close(self) -> None:
        await super().close()
        self._executor.shutdown(wait=False)

    async def _transcribe(
        self,
        audio_array: NDArray,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        if self.whisper is None:
            raise ValueError("Whisper model not loaded, call warmup() first")

        whisper = self.whisper

        def worker() -> tuple[list[Segment], TranscriptionInfo]:
            segments, info = whisper.transcribe(
                audio_array,
                language=self.language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return list(segments), info

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, worker)
