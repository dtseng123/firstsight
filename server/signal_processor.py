import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

ALERT_NO_PULSE = "no_pulse"
EMA_ALPHA = 0.3        # BPM smoothing weight — lower is smoother, ~1s lag at 30fps
CONF_WINDOW = 60       # sliding window length in compute() calls
CONF_BAD_FRACTION = 0.7  # fraction of window that must be bad to fire the alert


@dataclass
class HeartRateResult:
    bpm: float
    confidence: float
    alert: Optional[str]


class SignalProcessor:
    def __init__(self, fps: float = 30.0, buffer_size: int = 150,
                 min_freq: float = 1.0, max_freq: float = 2.0,
                 conf_window: int = CONF_WINDOW):
        self.fps = fps
        self.buffer_size = buffer_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._bpm_ema: Optional[float] = None
        self._conf_window: deque = deque(maxlen=conf_window)

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
        raw_bpm = float(peak_freq * 60)

        peak_power = bandpass[peak_idx]
        out_of_band = avg_spectrum[~mask]
        noise = float(out_of_band.mean()) if len(out_of_band) > 0 else 0.0
        noise = noise if noise > 0 else 1e-10
        confidence = round(min(float(peak_power / noise) / 5.0, 1.0), 3)

        # EMA smoothing — prevents BPM jumping by full FFT bin widths frame-to-frame
        if self._bpm_ema is None:
            self._bpm_ema = raw_bpm
        else:
            self._bpm_ema = EMA_ALPHA * raw_bpm + (1 - EMA_ALPHA) * self._bpm_ema
        bpm = round(self._bpm_ema, 1)

        # Sliding window alert — requires sustained poor signal, not just consecutive frames.
        # One good frame in a run of bad ones no longer resets the counter.
        is_bad = raw_bpm < 40 or raw_bpm > 180 or confidence < 0.3
        self._conf_window.append(is_bad)
        window_full = len(self._conf_window) == self._conf_window.maxlen
        bad_fraction = sum(self._conf_window) / len(self._conf_window)
        alert = ALERT_NO_PULSE if window_full and bad_fraction >= CONF_BAD_FRACTION else None

        return HeartRateResult(bpm=bpm, confidence=confidence, alert=alert)
