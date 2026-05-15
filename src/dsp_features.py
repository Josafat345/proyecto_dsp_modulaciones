from __future__ import annotations

import numpy as np


BASE_FEATURE_NAMES = [
    "amp_mean",
    "amp_std",
    "amp_max",
    "amp_power_mean",
    "papr",
    "i_std",
    "q_std",
    "abs_i_mean",
    "abs_q_mean",
    "iq_cross_mean",
    "phase_diff_std",
    "phase_diff_abs_mean",
    "amp_p25",
    "amp_p50",
    "amp_p75",
    "spectral_entropy",
    "effective_bandwidth",
    "spectral_centroid",
    "spectral_spread",
    "spectral_peak",
    "spectral_center_energy",
    "amp_delta_abs_mean",
    "i_delta_abs_mean",
    "q_delta_abs_mean",
]

DELAY_PHASE_FEATURE_NAMES = [
    f"delay_phase_lag{lag}_order{order}" for lag in (1, 2, 4, 8) for order in (2, 4, 8)
]

FEATURE_NAMES = BASE_FEATURE_NAMES + DELAY_PHASE_FEATURE_NAMES

FEATURE_GROUPS = {
    "amplitud": [0, 1, 2, 3, 4, 12, 13, 14, 21],
    "iq": [5, 6, 7, 8, 9, 22, 23],
    "fase": [10, 11, *range(24, 36)],
    "espectro": [15, 16, 17, 18, 19, 20],
    "todos": list(range(len(FEATURE_NAMES))),
}


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if np.isfinite(value) else 0.0


def spectral_entropy(power_spectrum: np.ndarray) -> float:
    p = np.maximum(power_spectrum, 1e-12)
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log2(p))
    return float(entropy / np.log2(len(p)))


def effective_bandwidth(power_spectrum: np.ndarray, threshold: float = 0.9) -> float:
    p = np.maximum(power_spectrum, 1e-12)
    p = p / np.sum(p)
    center = len(p) // 2
    order = np.argsort(np.abs(np.arange(len(p)) - center))
    cumulative = np.cumsum(p[order])
    used_bins = int(np.searchsorted(cumulative, threshold) + 1)
    return used_bins / len(p)


def extract_features(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(complex)
    signal = signal - np.mean(signal)
    rms = np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-12
    signal = signal / rms

    amplitude = np.abs(signal)
    phase = np.unwrap(np.angle(signal))
    phase_diff = np.diff(phase)
    i = np.real(signal)
    q = np.imag(signal)

    fft = np.fft.fftshift(np.fft.fft(signal * np.hanning(len(signal))))
    power = np.abs(fft) ** 2
    power = power / (np.sum(power) + 1e-12)
    freqs = np.linspace(-0.5, 0.5, len(power), endpoint=False)
    centroid = float(np.sum(freqs * power))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power)))

    delay_phase_features: list[float] = []
    for lag in (1, 2, 4, 8):
        delayed_product = signal[lag:] * np.conj(signal[:-lag])
        delayed_phase = np.angle(delayed_product)
        for order in (2, 4, 8):
            delay_phase_features.append(float(np.abs(np.mean(np.exp(1j * order * delayed_phase)))))

    features = [
        float(np.mean(amplitude)),
        _safe_std(amplitude),
        float(np.max(amplitude)),
        float(np.mean(amplitude**2)),
        float(np.max(amplitude**2) / (np.mean(amplitude**2) + 1e-12)),
        _safe_std(i),
        _safe_std(q),
        float(np.mean(np.abs(i))),
        float(np.mean(np.abs(q))),
        float(np.mean(i * q)),
        _safe_std(phase_diff),
        float(np.mean(np.abs(phase_diff))),
        float(np.percentile(amplitude, 25)),
        float(np.percentile(amplitude, 50)),
        float(np.percentile(amplitude, 75)),
        spectral_entropy(power),
        effective_bandwidth(power),
        centroid,
        spread,
        float(np.max(power)),
        float(np.sum(power[len(power) // 2 - 4 : len(power) // 2 + 5])),
        float(np.mean(np.abs(np.diff(amplitude)))),
        float(np.mean(np.abs(np.diff(i)))),
        float(np.mean(np.abs(np.diff(q)))),
        *delay_phase_features,
    ]
    return np.asarray(features, dtype=float)


def extract_feature_matrix(signals: np.ndarray) -> np.ndarray:
    return np.vstack([extract_features(signal) for signal in signals])


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-9
    return (x_train - mean) / std, (x_test - mean) / std, mean, std
