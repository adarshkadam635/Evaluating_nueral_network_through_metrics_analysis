"""Checks for leakage protection, metric interpretation, and model serialization."""
import numpy as np
import pytest
import tensorflow as tf

from train import build_model, evaluate, prepare_splits


def test_prepare_splits_removes_overlaps_duplicates_and_conflicts():
    raw = {
        "train": (["unique", " UNIQUE ", "conflict", "conflict", "validation", "test", "another"], np.array([0, 0, 0, 2, 1, 2, 1])),
        "val": (["validation", " VALIDATION ", "test"], np.array([1, 1, 2])),
        "test": (["test", "test"], np.array([2, 2])),
    }
    clean, audit = prepare_splits(raw)
    assert clean["train"][0] == ["unique", "another"]
    assert clean["val"][0] == ["validation"]
    assert clean["test"][0] == raw["test"][0]
    assert audit["train"]["removed_rows"] == 5


def test_metrics_use_three_classes():
    metrics = evaluate(np.array([0, 1, 2]), np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == 1.0
    assert metrics["confusion_matrix"] == np.eye(3, dtype=int).tolist()


@pytest.mark.parametrize("name", ["dense", "cnn", "lstm"])
def test_model_round_trip_includes_vocabulary(name, tmp_path):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=20, output_sequence_length=6)
    text = tf.constant(["I love this", "I hate this", "today is Tuesday"])
    vectorizer.adapt(text)
    model = build_model(name, len(vectorizer.get_vocabulary()), 6, 8)
    # Exercise an actual optimization step for every architecture.
    model.train_on_batch(vectorizer(text), np.array([2, 0, 1]))
    inputs = tf.keras.Input(shape=(), dtype="string")
    inference = tf.keras.Model(inputs, model(vectorizer(inputs)))
    path = tmp_path / f"{name}.keras"
    inference.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(inference(text).numpy(), restored(text).numpy(), atol=1e-6)
