from __future__ import annotations

"""MLP ligero implementado solo con NumPy.

Este archivo contiene el modelo de IA del proyecto. Se implementa sin PyTorch
ni TensorFlow para que el asesor pueda ver claramente que ocurre: pesos,
forward pass, softmax, perdida y actualizacion por gradiente.
"""

import numpy as np


class SimpleMLP:
    """Red neuronal de una capa oculta para clasificacion multiclase."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 123,
    ) -> None:
        """Inicializa pesos y sesgos.

        input_dim: numero de rasgos DSP.
        hidden_dim: neuronas de la capa oculta.
        output_dim: numero de modulaciones a clasificar.
        """
        rng = np.random.default_rng(seed)
        # Inicializacion tipo He para trabajar bien con activacion ReLU.
        self.w1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagacion hacia adelante: features -> probabilidades."""
        z1 = x @ self.w1 + self.b1
        # ReLU deja pasar valores positivos y apaga valores negativos.
        h = np.maximum(z1, 0)
        logits = h @ self.w2 + self.b2
        # Restar el maximo mejora estabilidad numerica antes del softmax.
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return z1, h, probs

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Devuelve probabilidad estimada para cada modulacion."""
        return self._forward(x)[2]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Devuelve la clase con mayor probabilidad."""
        return np.argmax(self.predict_proba(x), axis=1)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 160,
        learning_rate: float = 0.035,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        seed: int = 321,
    ) -> list[float]:
        """Entrena el MLP con mini-batches y descenso por gradiente.

        El entrenamiento usa entropia cruzada multiclase y regularizacion L2.
        Devuelve el historial de perdida para poder graficar o auditar.
        """
        rng = np.random.default_rng(seed)
        n = len(x)
        history: list[float] = []

        for _ in range(epochs):
            # Mezclar muestras en cada epoca reduce dependencia del orden.
            indexes = rng.permutation(n)
            x_shuffled = x[indexes]
            y_shuffled = y[indexes]
            epoch_losses: list[float] = []

            for start in range(0, n, batch_size):
                xb = x_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                z1, h, probs = self._forward(xb)

                # Perdida de clasificacion: penaliza baja probabilidad en la clase real.
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
                loss += weight_decay * (np.sum(self.w1**2) + np.sum(self.w2**2))
                epoch_losses.append(float(loss))

                # Gradiente de softmax + entropia cruzada.
                grad_logits = probs
                grad_logits[np.arange(len(yb)), yb] -= 1
                grad_logits /= len(yb)

                # Backpropagation: calculo de gradientes capa por capa.
                grad_w2 = h.T @ grad_logits + 2 * weight_decay * self.w2
                grad_b2 = grad_logits.sum(axis=0)
                grad_h = grad_logits @ self.w2.T
                grad_z1 = grad_h * (z1 > 0)
                grad_w1 = xb.T @ grad_z1 + 2 * weight_decay * self.w1
                grad_b1 = grad_z1.sum(axis=0)

                # Actualizacion de parametros.
                self.w2 -= learning_rate * grad_w2
                self.b2 -= learning_rate * grad_b2
                self.w1 -= learning_rate * grad_w1
                self.b1 -= learning_rate * grad_b1

            history.append(float(np.mean(epoch_losses)))

        return history


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Porcentaje de predicciones correctas."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Construye matriz de confusion: filas reales, columnas predichas."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix
