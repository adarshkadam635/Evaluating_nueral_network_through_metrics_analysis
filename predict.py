"""Predict sentiment using a saved model with its bundled text vectorizer."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

LABELS = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="Quote each text separately")
    parser.add_argument("--model", type=Path, help="Defaults to the validation-selected model")
    args = parser.parse_args()
    artifacts = Path(__file__).resolve().parent / "artifacts"
    model_path = args.model
    if model_path is None:
        summary = json.loads((artifacts / "summary.json").read_text())
        model_path = artifacts / f"{summary['best_model']}.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model(tf.constant(args.text), training=False).numpy()
    print(json.dumps([{"text": text, "sentiment": LABELS[int(probs.argmax())], "probabilities": dict(zip(LABELS, map(float, probs)))} for text, probs in zip(args.text, probabilities)], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
