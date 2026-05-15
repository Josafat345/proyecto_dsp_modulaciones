from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from dsp_features import FEATURE_GROUPS, FEATURE_NAMES, extract_feature_matrix, standardize_train_test
from signal_generator import MODULATIONS, generate_dataset
from simple_mlp import SimpleMLP, accuracy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "research"


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.28,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(x))
    test_size = int(len(x) * test_fraction)
    test_idx = indexes[:test_size]
    train_idx = indexes[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def nearest_centroid_fit(x_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    centroids = []
    for label in range(n_classes):
        centroids.append(x_train[y_train == label].mean(axis=0))
    return np.vstack(centroids)


def nearest_centroid_predict(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def fit_models(
    x: np.ndarray,
    y: np.ndarray,
    feature_indexes: list[int] | None = None,
    seed: int = 2026,
    epochs: int = 110,
) -> dict:
    if feature_indexes is not None:
        x = x[:, feature_indexes]

    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=seed)
    x_train_std, x_test_std, mean, std = standardize_train_test(x_train, x_test)

    baseline_centroids = nearest_centroid_fit(x_train_std, y_train, len(MODULATIONS))
    baseline_pred = nearest_centroid_predict(x_test_std, baseline_centroids)

    model = SimpleMLP(input_dim=x_train_std.shape[1], hidden_dim=48, output_dim=len(MODULATIONS), seed=seed)
    model.fit(x_train_std, y_train, epochs=epochs, learning_rate=0.035, batch_size=64, seed=seed + 1)
    ai_pred = model.predict(x_test_std)

    return {
        "model": model,
        "mean": mean,
        "std": std,
        "feature_indexes": feature_indexes,
        "baseline_centroids": baseline_centroids,
        "ai_accuracy": accuracy(y_test, ai_pred),
        "baseline_accuracy": accuracy(y_test, baseline_pred),
        "x_test": x_test,
        "y_test": y_test,
    }


def evaluate_trained_models(
    bundle: dict,
    signals: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    x = extract_feature_matrix(signals)
    indexes = bundle["feature_indexes"]
    if indexes is not None:
        x = x[:, indexes]
    x_std = (x - bundle["mean"]) / bundle["std"]
    ai_pred = bundle["model"].predict(x_std)
    baseline_pred = nearest_centroid_predict(x_std, bundle["baseline_centroids"])
    return accuracy(y, ai_pred), accuracy(y, baseline_pred)


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def round_row(row: dict) -> dict:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded


def summarize_metric(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando dataset base AWGN para comparacion IA vs baseline...")
    signals, y, labels = generate_dataset(samples_per_class=240, snr_range_db=(4.0, 24.0), channel_model="awgn", seed=100)
    x = extract_feature_matrix(signals)
    full_bundle = fit_models(x, y, feature_indexes=FEATURE_GROUPS["todos"], epochs=120)

    baseline_rows = [
        round_row({"modelo": "Baseline DSP distancia-centroide", "accuracy": full_bundle["baseline_accuracy"]}),
        round_row({"modelo": "MLP IA con features DSP", "accuracy": full_bundle["ai_accuracy"]}),
    ]
    write_csv(OUTPUT_DIR / "baseline_vs_ai.csv", ["modelo", "accuracy"], baseline_rows)

    print("Repitiendo validacion con multiples semillas...")
    repeated_rows = []
    repeated_seeds = [410, 411, 412, 413, 414]
    for seed in repeated_seeds:
        repeated_signals, repeated_y, _ = generate_dataset(
            samples_per_class=180,
            snr_range_db=(4.0, 24.0),
            channel_model="awgn",
            seed=seed,
        )
        repeated_x = extract_feature_matrix(repeated_signals)
        bundle = fit_models(
            repeated_x,
            repeated_y,
            feature_indexes=FEATURE_GROUPS["todos"],
            seed=seed + 10_000,
            epochs=100,
        )
        repeated_rows.append(
            round_row(
                {
                    "seed": seed,
                    "baseline_accuracy": bundle["baseline_accuracy"],
                    "ai_accuracy": bundle["ai_accuracy"],
                    "ai_minus_baseline": bundle["ai_accuracy"] - bundle["baseline_accuracy"],
                }
            )
        )
    write_csv(
        OUTPUT_DIR / "repeated_trials.csv",
        ["seed", "baseline_accuracy", "ai_accuracy", "ai_minus_baseline"],
        repeated_rows,
    )

    print("Evaluando robustez por SNR...")
    snr_rows = []
    for snr in [-5, 0, 5, 10, 15, 20, 25]:
        test_signals, test_y, _ = generate_dataset(
            samples_per_class=90,
            snr_range_db=(float(snr), float(snr)),
            channel_model="awgn",
            seed=200 + int(snr + 10),
        )
        ai_acc, baseline_acc = evaluate_trained_models(full_bundle, test_signals, test_y)
        snr_rows.append(round_row({"snr_db": snr, "baseline_accuracy": baseline_acc, "ai_accuracy": ai_acc}))
    write_csv(OUTPUT_DIR / "snr_sweep.csv", ["snr_db", "baseline_accuracy", "ai_accuracy"], snr_rows)

    print("Ejecutando ablation study por grupos de features...")
    ablation_rows = []
    for group_name, indexes in FEATURE_GROUPS.items():
        bundle = fit_models(x, y, feature_indexes=list(indexes), seed=310 + len(ablation_rows), epochs=95)
        ablation_rows.append(
            round_row(
                {
                    "grupo_features": group_name,
                    "n_features": len(indexes),
                    "baseline_accuracy": bundle["baseline_accuracy"],
                    "ai_accuracy": bundle["ai_accuracy"],
                }
            )
        )
    write_csv(OUTPUT_DIR / "ablation_features.csv", ["grupo_features", "n_features", "baseline_accuracy", "ai_accuracy"], ablation_rows)

    print("Evaluando generalizacion a canales no ideales...")
    channel_rows = []
    for channel_model in ["awgn", "rayleigh", "multipath"]:
        test_signals, test_y, _ = generate_dataset(
            samples_per_class=90,
            snr_range_db=(10.0, 18.0),
            channel_model=channel_model,
            seed=500 + len(channel_rows),
        )
        ai_acc, baseline_acc = evaluate_trained_models(full_bundle, test_signals, test_y)
        channel_rows.append(round_row({"train_channel": "awgn", "test_channel": channel_model, "baseline_accuracy": baseline_acc, "ai_accuracy": ai_acc}))
    write_csv(OUTPUT_DIR / "channel_generalization.csv", ["train_channel", "test_channel", "baseline_accuracy", "ai_accuracy"], channel_rows)

    payload = {
        "title": "Experimentos de robustez y explicabilidad para clasificacion de modulaciones",
        "feature_count": len(FEATURE_NAMES),
        "feature_groups": {name: [FEATURE_NAMES[i] for i in indexes] for name, indexes in FEATURE_GROUPS.items()},
        "baseline_vs_ai": baseline_rows,
        "repeated_trials": repeated_rows,
        "repeated_trial_summary": {
            "baseline_accuracy": summarize_metric(repeated_rows, "baseline_accuracy"),
            "ai_accuracy": summarize_metric(repeated_rows, "ai_accuracy"),
            "ai_minus_baseline": summarize_metric(repeated_rows, "ai_minus_baseline"),
        },
        "snr_sweep": snr_rows,
        "ablation_features": ablation_rows,
        "channel_generalization": channel_rows,
        "interpretacion": [
            "El baseline de distancia a centroides funciona como referencia clasica usando los mismos rasgos DSP.",
            "La validacion con multiples semillas reduce el riesgo de defender resultados dependientes de una sola particion del dataset.",
            "El barrido por SNR mide la robustez ante ruido y permite identificar el punto donde el clasificador deja de ser confiable.",
            "El ablation study estima que grupos de rasgos aportan mas informacion discriminante.",
            "La prueba de canales evalua si el modelo aprende modulacion o si depende demasiado de las condiciones AWGN del simulador.",
        ],
    }
    (OUTPUT_DIR / "research_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResultados principales")
    for row in baseline_rows:
        print(f"{row['modelo']}: {row['accuracy']:.3f}")
    repeated_summary = summarize_metric(repeated_rows, "ai_accuracy")
    print(f"MLP IA multisemilla: {repeated_summary['mean']:.3f} +/- {repeated_summary['std']:.3f}")
    print(f"Archivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
