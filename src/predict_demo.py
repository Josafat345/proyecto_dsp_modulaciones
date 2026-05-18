from __future__ import annotations

"""Demo de inferencia usando el modelo base ya entrenado.

Este script sirve para mostrar rapidamente el uso practico del modelo:
cargar pesos guardados, generar senales nuevas y predecir su modulacion.
No entrena; solo hace inferencia.
"""

from pathlib import Path

import numpy as np

from dsp_features import extract_features
from signal_generator import MODULATIONS, apply_channel, generate_clean_signal
from simple_mlp import SimpleMLP


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "baseline" / "model_parameters.npz"


def load_model() -> tuple[SimpleMLP, np.ndarray, np.ndarray, list[str]]:
    """Carga pesos, normalizacion y etiquetas guardadas por experiment.py."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Primero ejecuta experiment.py para entrenar y guardar el modelo base.")

    data = np.load(MODEL_PATH)
    labels = [str(label) for label in data["labels"].tolist()]
    # Se crea una red con la misma arquitectura y luego se reemplazan sus pesos.
    model = SimpleMLP(input_dim=data["w1"].shape[0], hidden_dim=data["w1"].shape[1], output_dim=data["w2"].shape[1])
    model.w1 = data["w1"]
    model.b1 = data["b1"]
    model.w2 = data["w2"]
    model.b2 = data["b2"]
    return model, data["mean"], data["std"], labels


def main() -> None:
    """Genera una senal nueva por modulacion y muestra la prediccion."""
    model, mean, std, labels = load_model()
    rng = np.random.default_rng(2026)

    print("Demo de prediccion con senales nuevas")
    print("-" * 54)
    for modulation in MODULATIONS:
        # La senal nueva no viene del set de entrenamiento; se genera con otra
        # semilla y se degrada con canal para simular una recepcion.
        clean = generate_clean_signal(modulation, n_symbols=128, samples_per_symbol=8, rng=rng)
        noisy = apply_channel(clean, snr_db=10.0, rng=rng)
        # La normalizacion debe usar los mismos mean/std usados en entrenamiento.
        features = (extract_features(noisy) - mean) / std
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        prediction = labels[int(np.argmax(probabilities))]
        confidence = float(np.max(probabilities))
        print(f"Real: {modulation:5s} | Predicho: {prediction:5s} | Confianza: {confidence:0.3f}")


if __name__ == "__main__":
    main()
