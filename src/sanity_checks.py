from __future__ import annotations

import numpy as np

from dsp_features import FEATURE_NAMES, extract_feature_matrix, standardize_train_test
from experiment import train_test_split
from signal_generator import MODULATIONS, generate_dataset
from simple_mlp import SimpleMLP, accuracy


def main() -> None:
    signals, y, labels = generate_dataset(
        samples_per_class=8,
        n_symbols=32,
        samples_per_symbol=4,
        snr_range_db=(8.0, 16.0),
        seed=2026,
    )

    expected_signal_count = len(MODULATIONS) * 8
    assert signals.shape == (expected_signal_count, 128)
    assert labels == list(MODULATIONS)
    assert set(y.tolist()) == set(range(len(MODULATIONS)))

    x = extract_feature_matrix(signals)
    assert x.shape == (expected_signal_count, len(FEATURE_NAMES))
    assert np.isfinite(x).all()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_fraction=0.3, seed=2026)
    x_train, x_test, _, _ = standardize_train_test(x_train, x_test)

    model = SimpleMLP(input_dim=x_train.shape[1], hidden_dim=16, output_dim=len(MODULATIONS), seed=2026)
    history = model.fit(x_train, y_train, epochs=8, learning_rate=0.03, batch_size=16, seed=2027)
    pred = model.predict(x_test)

    assert len(history) == 8
    assert pred.shape == y_test.shape
    assert 0.0 <= accuracy(y_test, pred) <= 1.0

    print("Sanity checks completados correctamente.")
    print(f"Senales: {signals.shape[0]} | Features: {x.shape[1]} | Clases: {', '.join(labels)}")


if __name__ == "__main__":
    main()
