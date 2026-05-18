from __future__ import annotations

"""Fase 1: diagnostico de cambio de dominio.

Este es el script central del enfoque nuevo del proyecto. Entrena el modelo en
un dominio fuente controlado y lo evalua en dominios objetivo mas dificiles.
La idea es medir cuanto cae el rendimiento antes de aplicar adaptacion.
"""

import csv
import json
from pathlib import Path

import numpy as np

from dsp_features import extract_feature_matrix, standardize_train_test
from signal_generator import MODULATIONS, apply_channel, generate_clean_signal
from simple_mlp import SimpleMLP, accuracy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "domain_adaptation"


# Cada escenario representa una condicion de prueba. El modelo se entrena solo
# en el dominio fuente AWGN y despues se mide en estas condiciones.
DOMAIN_SCENARIOS = [
    {
        "scenario": "source_holdout_awgn",
        "channel_model": "awgn",
        "snr_range_db": (8.0, 24.0),
        "max_freq_offset": 0.025,
        "description": "Mismo dominio que entrenamiento; sirve como referencia alta.",
    },
    {
        "scenario": "target_awgn_low_snr",
        "channel_model": "awgn",
        "snr_range_db": (0.0, 14.0),
        "max_freq_offset": 0.050,
        "description": "Mismo tipo de canal, pero con menor SNR y mayor offset.",
    },
    {
        "scenario": "target_rayleigh",
        "channel_model": "rayleigh",
        "snr_range_db": (4.0, 18.0),
        "max_freq_offset": 0.035,
        "description": "Canal Rayleigh: fading plano sin componente dominante.",
    },
    {
        "scenario": "target_multipath",
        "channel_model": "multipath",
        "snr_range_db": (4.0, 18.0),
        "max_freq_offset": 0.035,
        "description": "Canal multipath: mezcla copias retardadas y deforma simbolos.",
    },
    {
        "scenario": "target_harsh_multipath",
        "channel_model": "multipath",
        "snr_range_db": (0.0, 12.0),
        "max_freq_offset": 0.060,
        "description": "Dominio objetivo mas dificil: multipath, bajo SNR y offset alto.",
    },
]


def generate_domain_dataset(
    samples_per_class: int,
    snr_range_db: tuple[float, float],
    channel_model: str,
    max_freq_offset: float,
    seed: int,
    n_symbols: int = 128,
    samples_per_symbol: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Genera datos para un dominio especifico.

    A diferencia de generate_dataset(), aqui se controla tambien el maximo
    offset de frecuencia. Eso permite crear dominios objetivo mas severos.
    """
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    for label, modulation in enumerate(MODULATIONS):
        for _ in range(samples_per_class):
            # Primero se crea la senal ideal y despues se pasa por el canal.
            clean = generate_clean_signal(modulation, n_symbols, samples_per_symbol, rng)
            snr_db = rng.uniform(*snr_range_db)
            received = apply_channel(
                clean,
                snr_db=snr_db,
                rng=rng,
                max_freq_offset=max_freq_offset,
                channel_model=channel_model,
            )
            signals.append(received)
            labels.append(label)

    return np.asarray(signals), np.asarray(labels, dtype=int), list(MODULATIONS)


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 2027,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide el dominio fuente en entrenamiento y holdout de referencia."""
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(x))
    test_size = int(len(x) * test_fraction)
    test_idx = indexes[:test_size]
    train_idx = indexes[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def nearest_centroid_fit(x_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Entrena el baseline DSP calculando un centroide por modulacion."""
    return np.vstack([x_train[y_train == label].mean(axis=0) for label in range(n_classes)])


def nearest_centroid_predict(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Predice la clase usando distancia euclidiana al centroide mas cercano."""
    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def round_float(value: float) -> float:
    """Redondea resultados numericos para reportes legibles."""
    return round(float(value), 4)


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    """Guarda resultados de escenarios en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Entrena en dominio fuente y evalua en dominios objetivo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fase 1: diagnosticando cambio de dominio sintetico -> objetivo realista")
    print("Entrenando solo en dominio fuente AWGN...")

    # Dominio fuente: condiciones relativamente controladas. Aqui el modelo
    # aprende la relacion entre rasgos DSP y tipo de modulacion.
    source_signals, source_y, labels = generate_domain_dataset(
        samples_per_class=260,
        snr_range_db=(8.0, 24.0),
        channel_model="awgn",
        max_freq_offset=0.025,
        seed=710,
    )
    source_x = extract_feature_matrix(source_signals)
    x_train, x_source_holdout, y_train, y_source_holdout = train_test_split(source_x, source_y)
    x_train_std, x_source_holdout_std, mean, std = standardize_train_test(x_train, x_source_holdout)

    # MLP principal y baseline clasico entrenados con exactamente el mismo
    # dominio fuente. Asi la comparacion es justa.
    model = SimpleMLP(input_dim=x_train_std.shape[1], hidden_dim=48, output_dim=len(labels), seed=710)
    model.fit(x_train_std, y_train, epochs=120, learning_rate=0.035, batch_size=64, seed=711)
    centroids = nearest_centroid_fit(x_train_std, y_train, len(labels))

    source_ai_accuracy = accuracy(y_source_holdout, model.predict(x_source_holdout_std))
    source_baseline_accuracy = accuracy(y_source_holdout, nearest_centroid_predict(x_source_holdout_std, centroids))

    rows = []
    for index, scenario in enumerate(DOMAIN_SCENARIOS):
        print(f"Evaluando {scenario['scenario']}...")
        # Dominio objetivo: no se reentrena el modelo, solo se prueba cuanto
        # resiste al cambio de condiciones.
        target_signals, target_y, _ = generate_domain_dataset(
            samples_per_class=120,
            snr_range_db=scenario["snr_range_db"],
            channel_model=scenario["channel_model"],
            max_freq_offset=scenario["max_freq_offset"],
            seed=900 + index,
        )
        target_x = extract_feature_matrix(target_signals)
        # Punto importante: el objetivo se normaliza con media/std del dominio
        # fuente. Esto representa "sin adaptacion" y sirve como Fase 1.
        target_x_std = (target_x - mean) / std
        target_ai_accuracy = accuracy(target_y, model.predict(target_x_std))
        target_baseline_accuracy = accuracy(target_y, nearest_centroid_predict(target_x_std, centroids))

        rows.append(
            {
                "scenario": scenario["scenario"],
                "train_domain": "awgn_snr_8_24_offset_0.025",
                "test_channel": scenario["channel_model"],
                "test_snr_min_db": scenario["snr_range_db"][0],
                "test_snr_max_db": scenario["snr_range_db"][1],
                "test_max_freq_offset": scenario["max_freq_offset"],
                "source_ai_accuracy": round_float(source_ai_accuracy),
                "target_ai_accuracy": round_float(target_ai_accuracy),
                "ai_accuracy_drop": round_float(source_ai_accuracy - target_ai_accuracy),
                "source_baseline_accuracy": round_float(source_baseline_accuracy),
                "target_baseline_accuracy": round_float(target_baseline_accuracy),
                "baseline_accuracy_drop": round_float(source_baseline_accuracy - target_baseline_accuracy),
                "description": scenario["description"],
            }
        )

    write_csv(
        OUTPUT_DIR / "domain_shift_baseline.csv",
        [
            "scenario",
            "train_domain",
            "test_channel",
            "test_snr_min_db",
            "test_snr_max_db",
            "test_max_freq_offset",
            "source_ai_accuracy",
            "target_ai_accuracy",
            "ai_accuracy_drop",
            "source_baseline_accuracy",
            "target_baseline_accuracy",
            "baseline_accuracy_drop",
            "description",
        ],
        rows,
    )

    payload = {
        "title": "Fase 1 - Diagnostico de cambio de dominio para AMC",
        "objective": "Medir cuanto cae un clasificador entrenado en AWGN cuando se evalua en dominios objetivo mas realistas.",
        "source_domain": {
            "channel_model": "awgn",
            "snr_range_db": [8.0, 24.0],
            "max_freq_offset": 0.025,
            "samples_per_class": 260,
        },
        "model": {
            "type": "SimpleMLP",
            "input_features": x_train.shape[1],
            "hidden_units": 48,
            "classes": labels,
        },
        "results": rows,
        "next_phases": [
            "Fase 2: probar normalizacion con estadisticas del dominio objetivo sin etiquetas.",
            "Fase 3: implementar CORAL para alinear covarianzas entre fuente y objetivo.",
            "Fase 4: usar pocas muestras etiquetadas del objetivo para fine-tuning.",
            "Fase 5: reemplazar el dominio objetivo sintetico por RadioML o capturas SDR reales.",
        ],
    }
    (OUTPUT_DIR / "domain_shift_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResultados Fase 1")
    print(f"Accuracy IA en fuente AWGN: {source_ai_accuracy:.3f}")
    for row in rows:
        print(f"{row['scenario']}: target IA={row['target_ai_accuracy']:.3f}, caida={row['ai_accuracy_drop']:.3f}")
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
