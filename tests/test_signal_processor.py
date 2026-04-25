import numpy as np
import pytest
from server.signal_processor import SignalProcessor, HeartRateResult


def make_buffer(fps: float, n: int, target_hz: float, h: int = 8, w: int = 15) -> np.ndarray:
    t = np.linspace(0, n / fps, n)
    signal = np.sin(2 * np.pi * target_hz * t)
    buf = np.zeros((n, h, w, 3), dtype=np.float32)
    for i in range(n):
        buf[i] = signal[i]
    return buf


def test_detects_known_frequency():
    buf = make_buffer(fps=30.0, n=150, target_hz=1.2)  # 72 BPM
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    result = p.compute(buf)
    assert abs(result.bpm - 72.0) < 5.0


def test_confidence_high_for_clean_signal():
    buf = make_buffer(fps=30.0, n=150, target_hz=1.2)
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    result = p.compute(buf)
    assert result.confidence > 0.5


def test_returns_zero_for_incomplete_buffer():
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    small = np.zeros((50, 8, 15, 3), dtype=np.float32)
    result = p.compute(small)
    assert result.bpm == 0.0
    assert result.confidence == 0.0


def test_no_pulse_alert_after_sustained_low_confidence():
    noise = np.random.rand(150, 8, 15, 3).astype(np.float32) * 0.001
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    p._no_signal_threshold = 1  # trigger immediately
    result = p.compute(noise)
    assert result.alert == "no_pulse"


def test_no_alert_for_clean_signal():
    buf = make_buffer(fps=30.0, n=150, target_hz=1.2)
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    result = p.compute(buf)
    assert result.alert is None


def test_result_is_heartrate_result_dataclass():
    buf = make_buffer(fps=30.0, n=150, target_hz=1.2)
    p = SignalProcessor(fps=30.0, buffer_size=150, min_freq=1.0, max_freq=2.0)
    result = p.compute(buf)
    assert isinstance(result, HeartRateResult)
    assert hasattr(result, "bpm")
    assert hasattr(result, "confidence")
    assert hasattr(result, "alert")
