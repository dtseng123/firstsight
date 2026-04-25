import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HeartRateResult:
    bpm: float
    confidence: float
    alert: Optional[str]


class SignalProcessor:
    def __init__(self, fps: float = 30.0, buffer_size: int = 150,
                 min_freq: float = 1.0, max_freq: float = 2.0):
        self.fps = fps
        self.buffer_size = buffer_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._no_signal_count = 0
        self._no_signal_threshold = int(10 * fps)

    def compute(self, buffer: np.ndarray) -> HeartRateResult:
        if len(buffer) < self.buffer_size:
            return HeartRateResult(bpm=0.0, confidence=0.0, alert=None)

        fft_result = np.fft.fft(buffer, axis=0)
        avg_spectrum = np.abs(fft_result).mean(axis=(1, 2, 3))

        frequencies = (self.fps * np.arange(self.buffer_size)) / self.buffer_size
        mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)

        bandpass = avg_spectrum.copy()
        bandpass[~mask] = 0
        peak_idx = int(np.argmax(bandpass))
        peak_freq = frequencies[peak_idx]
        bpm = round(float(peak_freq * 60), 1)

        peak_power = avg_spectrum[peak_idx]
        bg_mean = avg_spectrum[mask].mean()
        bg_mean = bg_mean if bg_mean > 0 else 1e-10
        confidence = round(min(float(peak_power / bg_mean) / 10.0, 1.0), 3)

        if bpm < 40 or bpm > 180 or confidence < 0.3:
            self._no_signal_count += 1
        else:
            self._no_signal_count = 0

        alert = "no_pulse" if self._no_signal_count >= self._no_signal_threshold else None
        return HeartRateResult(bpm=bpm, confidence=confidence, alert=alert)
