import numpy as np
from dataclasses import dataclass
from typing import Optional

ALERT_NO_PULSE = "no_pulse"


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
        if buffer.ndim != 4 or len(buffer) < self.buffer_size:
            return HeartRateResult(bpm=0.0, confidence=0.0, alert=None)

        fft_result = np.fft.fft(buffer, axis=0, n=self.buffer_size)
        avg_spectrum = np.abs(fft_result).mean(axis=(1, 2, 3))

        frequencies = (self.fps * np.arange(self.buffer_size)) / self.buffer_size
        mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)

        bandpass = avg_spectrum.copy()
        bandpass[~mask] = 0
        peak_idx = int(np.argmax(bandpass))
        peak_freq = frequencies[peak_idx]
        bpm = round(float(peak_freq * 60), 1)

        peak_power = bandpass[peak_idx]
        # Out-of-band power is the true noise floor — everything the signal competes against
        out_of_band = avg_spectrum[~mask]
        noise = float(out_of_band.mean()) if len(out_of_band) > 0 else 0.0
        noise = noise if noise > 0 else 1e-10
        # SNR / 5.0 calibrated for real video: clean pulse → confidence > 0.5, flat noise → confidence < 0.3
        confidence = round(min(float(peak_power / noise) / 5.0, 1.0), 3)

        if bpm < 40 or bpm > 180 or confidence < 0.3:
            self._no_signal_count += 1
        else:
            self._no_signal_count = 0

        alert = ALERT_NO_PULSE if self._no_signal_count >= self._no_signal_threshold else None
        return HeartRateResult(bpm=bpm, confidence=confidence, alert=alert)
