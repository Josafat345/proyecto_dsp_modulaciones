from __future__ import annotations

"""Experimento base de clasificacion de modulaciones.

Este script demuestra que el pipeline completo funciona:
1. Generar senales sinteticas.
2. Extraer rasgos DSP.
3. Entrenar un MLP.
4. Guardar metricas, matriz de confusion y parametros.

En la investigacion actual este archivo queda como baseline historico; el
aporte principal esta en los experimentos de cambio de dominio.
"""

import csv
import json
from pathlib import Path

import numpy as np

from dsp_features import extract_feature_matrix, standardize_train_test
from signal_generator import generate_dataset
from simple_mlp import SimpleMLP, accuracy, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "baseline"


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide datos en entrenamiento y prueba de forma reproducible."""
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(x))
    test_size = int(len(x) * test_fraction)
    test_idx = indexes[:test_size]
    train_idx = indexes[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def save_confusion_csv(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    """Guarda la matriz de confusion en CSV para revisar errores por clase."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["real/predicho", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_summary(
    labels: list[str],
    matrix: np.ndarray,
    train_accuracy: float,
    test_accuracy: float,
    path: Path,
) -> None:
    """Guarda resumen JSON con accuracy global y accuracy por modulacion."""
    per_class = {}
    for i, label in enumerate(labels):
        total = matrix[i].sum()
        correct = matrix[i, i]
        per_class[label] = {
            "correctas": int(correct),
            "total": int(total),
            "accuracy": round(float(correct / total), 4) if total else 0.0,
        }

    payload = {
        "descripcion": "Clasificador MLP en NumPy para reconocimiento de modulaciones digitales.",
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "clases": labels,
        "per_class": per_class,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Ejecuta el experimento base completo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando dataset sintetico...")
    signals, y, labels = generate_dataset(samples_per_class=320, seed=42)
    print(f"Senales: {signals.shape[0]}, muestras por senal: {signals.shape[1]}")

    print("Extrayendo rasgos DSP...")
    x = extract_feature_matrix(signals)
    # Primero se separa entrenamiento/prueba y luego se normaliza usando solo
    # estadisticas del entrenamiento.
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train, x_test, mean, std = standardize_train_test(x_train, x_test)

    print("Entrenando MLP en NumPy...")
    model = SimpleMLP(input_dim=x_train.shape[1], hidden_dim=48, output_dim=len(labels))
    history = model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    matrix = confusion_matrix(y_test, test_pred, len(labels))

    # Estos archivos alimentan el notebook, la demo y los documentos.
    save_confusion_csv(matrix, labels, OUTPUT_DIR / "confusion_matrix.csv")
    save_summary(labels, matrix, train_acc, test_acc, OUTPUT_DIR / "summary.json")
    np.savez(
        OUTPUT_DIR / "model_parameters.npz",
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        mean=mean,
        std=std,
        labels=np.asarray(labels),
        loss=np.asarray(history),
    )

    print("\nResultados")
    print(f"Accuracy entrenamiento: {train_acc:.3f}")
    print(f"Accuracy prueba:        {test_acc:.3f}")
    print("\nMatriz de confusion")
    print("real/predicho", *labels)
    for label, row in zip(labels, matrix):
        print(label, *row.tolist())
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
