from __future__ import annotations

import numpy as np


MODULATIONS = ("BPSK", "QPSK", "8PSK", "16QAM", "2FSK", "4ASK")


def _normalize_power(signal: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(signal) ** 2)
    if power <= 1e-12:
        return signal
    return signal / np.sqrt(power)


def _repeat_symbols(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    return np.repeat(symbols, samples_per_symbol)


def _psk(order: int, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    indexes = rng.integers(0, order, size=n_symbols)
    phase = 2 * np.pi * indexes / order
    return np.exp(1j * phase)


def _qam16(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=float)
    i = rng.choice(levels, size=n_symbols)
    q = rng.choice(levels, size=n_symbols)
    return _normalize_power(i + 1j * q)


def _ask4(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=float)
    symbols = rng.choice(levels, size=n_symbols)
    return _normalize_power(symbols.astype(complex))


def _fsk2(n_symbols: int, samples_per_symbol: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 2, size=n_symbols)
    freqs = np.where(bits == 0, -0.18, 0.18)
    chunks = []
    t = np.arange(samples_per_symbol)
    phase = rng.uniform(0, 2 * np.pi)
    for freq in freqs:
        chunks.append(np.exp(1j * (2 * np.pi * freq * t + phase)))
        phase = (phase + 2 * np.pi * freq * samples_per_symbol) % (2 * np.pi)
    return np.concatenate(chunks)


def generate_clean_signal(
    modulation: str,
    n_symbols: int,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if modulation == "BPSK":
        symbols = _psk(2, n_symbols, rng)
        signal = _repeat_symbols(symbols, samples_per_symbol)
    elif modulation == "QPSK":
        symbols = _psk(4, n_symbols, rng)
        signal = _repeat_symbols(symbols, samples_per_symbol)
    elif modulation == "8PSK":
        symbols = _psk(8, n_symbols, rng)
        signal = _repeat_symbols(symbols, samples_per_symbol)
    elif modulation == "16QAM":
        symbols = _qam16(n_symbols, rng)
        signal = _repeat_symbols(symbols, samples_per_symbol)
    elif modulation == "4ASK":
        symbols = _ask4(n_symbols, rng)
        signal = _repeat_symbols(symbols, samples_per_symbol)
    elif modulation == "2FSK":
        signal = _fsk2(n_symbols, samples_per_symbol, rng)
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    return _normalize_power(signal.astype(complex))


def apply_channel(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    max_freq_offset: float = 0.025,
    channel_model: str = "awgn",
) -> np.ndarray:
    n = len(signal)
    t = np.arange(n)
    phase_offset = rng.uniform(0, 2 * np.pi)
    freq_offset = rng.uniform(-max_freq_offset, max_freq_offset)
    gain = rng.uniform(0.65, 1.35)

    impaired = gain * signal * np.exp(1j * (2 * np.pi * freq_offset * t + phase_offset))
    if channel_model == "awgn":
        pass
    elif channel_model == "rayleigh":
        rayleigh_gain = (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
        impaired = impaired * rayleigh_gain
    elif channel_model == "multipath":
        taps = np.zeros(7, dtype=complex)
        taps[0] = 1.0
        taps[2] = 0.42 * np.exp(1j * rng.uniform(0, 2 * np.pi))
        taps[5] = 0.22 * np.exp(1j * rng.uniform(0, 2 * np.pi))
        impaired = np.convolve(impaired, taps, mode="same")
    else:
        raise ValueError(f"Unsupported channel model: {channel_model}")

    signal_power = np.mean(np.abs(impaired) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    return impaired + noise


def generate_dataset(
    samples_per_class: int = 300,
    n_symbols: int = 128,
    samples_per_symbol: int = 8,
    snr_range_db: tuple[float, float] = (4.0, 24.0),
    channel_model: str = "awgn",
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    for label, modulation in enumerate(MODULATIONS):
        for _ in range(samples_per_class):
            clean = generate_clean_signal(modulation, n_symbols, samples_per_symbol, rng)
            snr_db = rng.uniform(*snr_range_db)
            signals.append(apply_channel(clean, snr_db, rng, channel_model=channel_model))
            labels.append(label)

    return np.asarray(signals), np.asarray(labels, dtype=int), list(MODULATIONS)
