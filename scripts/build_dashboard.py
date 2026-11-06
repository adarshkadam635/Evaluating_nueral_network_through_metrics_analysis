"""Export recorded evaluation and the selected dense model for a static dashboard."""
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "dashboard" / "data"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "summary", "dataset_manifest", "vocabulary"):
        shutil.copyfile(ARTIFACTS / f"{name}.json", OUTPUT / f"{name}.json")
    config = json.loads((ARTIFACTS / "config.json").read_text())
    # Machine-local paths are irrelevant to the public dashboard.
    config.pop("data_dir", None)
    config.pop("output", None)
    write_json(OUTPUT / "config.json", config)
    histories, roc = {}, {}
    for name in config["models"]:
        histories[name] = json.loads((ARTIFACTS / f"{name}_history.json").read_text())
        predictions = np.load(ARTIFACTS / f"{name}_test_predictions.npz")
        curves = []
        for i, label in enumerate(config["labels"]):
            fpr, tpr, _ = roc_curve(predictions["labels"] == i, predictions["probabilities"][:, i])
            selected = np.unique(np.linspace(0, len(fpr) - 1, min(200, len(fpr)), dtype=int))
            curves.append({"label": label, "auc": float(auc(fpr, tpr)), "points": [[float(fpr[j]), float(tpr[j])] for j in selected]})
        roc[name] = curves
    write_json(OUTPUT / "histories.json", histories)
    write_json(OUTPUT / "roc.json", roc)
    model = tf.keras.models.load_model(ARTIFACTS / "dense.keras", compile=False)
    network = model.get_layer("dense")
    vectorizer = model.get_layer("text_vectorization")
    tensors = []
    offset = 0
    weights = []
    for layer in network.layers:
        for i, weight in enumerate(layer.get_weights()):
            flattened = np.asarray(weight, dtype="<f4").ravel()
            tensors.append({"layer": layer.name, "index": i, "shape": list(weight.shape), "offset": offset, "length": int(flattened.size)})
            weights.append(flattened)
            offset += flattened.size
    np.concatenate(weights).astype("<f4").tofile(OUTPUT / "dense_weights.bin")
    write_json(OUTPUT / "inference.json", {"model": "dense", "labels": config["labels"], "sequence_length": config["sequence_length"], "embedding_dim": config["embedding_dim"], "tensors": tensors, "vectorizer": vectorizer.get_config()})
    examples = ["I really loved this experience!", "This was disappointing and awful.", "The meeting starts at three.", "I don't like this!!!", "hello\nworld", "unknownwordxyz"]
    # Include numerical reference fixtures, not raw training/test tweets.
    probabilities = model(tf.constant(examples), training=False).numpy()
    tokens = vectorizer(tf.constant(examples)).numpy()
    write_json(OUTPUT / "inference_fixtures.json", [{"text": text, "token_ids": ids.tolist(), "probabilities": probs.tolist()} for text, ids, probs in zip(examples, tokens, probabilities)])
    print(f"Exported dashboard data and {offset:,} float32 model weights to {OUTPUT}")


if __name__ == "__main__":
    main()
