"""Reproducible TweetEval dense/CNN/LSTM comparison. Run python train.py --help."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
DATA_REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
DATA_URL = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{DATA_REVISION}/datasets/sentiment"
LABELS = ["negative", "neutral", "positive"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_data(cache):
    """Fetch immutable upstream files, verify available checksums, validate rows."""
    cache.mkdir(parents=True, exist_ok=True)
    checksum_path = ROOT / "dataset_checksums.json"
    expected = json.loads(checksum_path.read_text()) if checksum_path.exists() else {}
    manifest = {"source": DATA_URL, "revision": DATA_REVISION, "files": {}}
    splits = {}
    for split in ("train", "val", "test"):
        contents = {}
        for kind in ("text", "labels"):
            name = f"{split}_{kind}.txt"
            path = cache / name
            if not path.exists():
                with urlopen(f"{DATA_URL}/{name}", timeout=90) as response:
                    payload = response.read()
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if name in expected and digest != expected[name]:
                raise ValueError(f"Checksum mismatch for {path}; remove the cached file and retry")
            manifest["files"][name] = {"sha256": digest, "bytes": len(payload)}
            contents[kind] = payload.decode("utf-8").splitlines()
        texts, labels = contents["text"], np.asarray(contents["labels"], dtype=np.int64)
        if len(texts) != len(labels) or not texts or not set(labels) <= {0, 1, 2}:
            raise ValueError(f"Invalid {split} data")
        if any(not text.strip() for text in texts):
            raise ValueError(f"Empty text in {split}")
        splits[split] = (texts, labels)
    return splits, manifest


def canonical(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def prepare_splits(raw):
    """Keep test untouched; remove text overlaps and conflicting fitting examples."""
    result = {"test": raw["test"]}
    blocked = {canonical(t) for t in raw["test"][0]}
    audit = {}
    for split in ("val", "train"):
        texts, labels = raw[split]
        label_sets = {}
        for text, label in zip(texts, labels):
            label_sets.setdefault(canonical(text), set()).add(int(label))
        seen, kept_texts, kept_labels = set(), [], []
        for text, label in zip(texts, labels):
            key = canonical(text)
            if key in blocked or key in seen or len(label_sets[key]) != 1:
                continue
            seen.add(key)
            kept_texts.append(text)
            kept_labels.append(label)
        result[split] = (kept_texts, np.asarray(kept_labels, dtype=np.int64))
        blocked.update(seen)
        audit[split] = {"raw_rows": len(texts), "kept_rows": len(kept_texts), "removed_rows": len(texts) - len(kept_texts)}
    audit["test"] = {"raw_rows": len(raw["test"][0]), "kept_rows": len(raw["test"][0]), "removed_rows": 0}
    return result, audit


def evaluate(y, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_precision": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "roc_auc_ovr_macro": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(y, predictions, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
    }


def build_model(name, vocabulary_size, sequence_length, embedding_dim):
    inputs = keras.Input(shape=(sequence_length,), dtype="int64", name="token_ids")
    x = keras.layers.Embedding(vocabulary_size, embedding_dim, mask_zero=name != "cnn", name="embedding")(inputs)
    if name == "dense":
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
    elif name == "cnn":
        x = keras.layers.Conv1D(64, 3, activation="relu")(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
    elif name == "lstm":
        x = keras.layers.LSTM(32)(x)
    else:
        raise ValueError(name)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(3, activation="softmax", name="sentiment")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_plots(name, history, metrics, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes[:2], ("accuracy", "loss")):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.set(xlabel="Epoch", ylabel=key.title(), title=f"{name.upper()} {key}")
        ax.legend()
    matrix = np.asarray(metrics["confusion_matrix"])
    axes[2].imshow(matrix, cmap="Blues")
    axes[2].set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output / f"{name}_evaluation.png", dpi=150)
    plt.close(fig)


def run(args):
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw, manifest = load_data(args.data_dir)
    splits, audit = prepare_splits(raw)
    if args.limit_train:
        # Balanced random subsample only for smoke tests; never truncate held-out data.
        from sklearn.model_selection import train_test_split
        texts, y = splits["train"]
        selected, _ = train_test_split(np.arange(len(y)), train_size=args.limit_train, stratify=y, random_state=args.seed)
        splits["train"] = ([texts[i] for i in selected], y[selected])
    vectorizer = keras.layers.TextVectorization(max_tokens=args.max_tokens, output_sequence_length=args.sequence_length, name="text_vectorization")
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(splits["train"][0]).batch(args.batch_size))
    encoded = {}
    for split, (texts, _) in splits.items():
        batches = tf.data.Dataset.from_tensor_slices(texts).batch(args.batch_size)
        encoded[split] = np.concatenate([vectorizer(batch).numpy() for batch in batches])
    write_json(output / "dataset_manifest.json", {**manifest, "deduplication": audit, "effective_class_counts": {s: {LABELS[int(k)]: v for k, v in sorted(Counter(y.tolist()).items())} for s, (_, y) in splits.items()}})
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"labels": LABELS, "vocabulary_size": len(vectorizer.get_vocabulary()), "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("tensorflow", "keras", "numpy", "scikit-learn", "matplotlib")}, "devices": [d.device_type for d in tf.config.list_physical_devices()]})
    write_json(output / "config.json", config)
    write_json(output / "vocabulary.json", vectorizer.get_vocabulary())
    results = {}
    for name in args.models:
        keras.utils.set_random_seed(args.seed)
        model = build_model(name, len(vectorizer.get_vocabulary()), args.sequence_length, args.embedding_dim)
        started = time.perf_counter()
        print(f"\nTraining {name}: {len(splits['train'][1])} examples", flush=True)
        history = model.fit(encoded["train"], splits["train"][1], validation_data=(encoded["val"], splits["val"][1]), epochs=args.epochs, batch_size=args.batch_size, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)], verbose=2)
        elapsed = time.perf_counter() - started
        probabilities = model.predict(encoded["val"], batch_size=args.batch_size, verbose=0)
        val_metrics = evaluate(splits["val"][1], probabilities)
        probabilities = model.predict(encoded["test"], batch_size=args.batch_size, verbose=0)
        test_metrics = evaluate(splits["test"][1], probabilities)
        # Persist the text preprocessing with the model for standalone inference.
        text_input = keras.Input(shape=(), dtype="string", name="text")
        inference_model = keras.Model(text_input, model(vectorizer(text_input)), name=f"{name}_sentiment")
        inference_model.save(output / f"{name}.keras")
        reloaded = keras.models.load_model(output / f"{name}.keras", compile=False)
        check = reloaded(tf.constant(splits["test"][0][:8]), training=False).numpy()
        np.testing.assert_allclose(check, probabilities[:8], atol=1e-6, rtol=1e-5)
        np.savez_compressed(output / f"{name}_test_predictions.npz", labels=splits["test"][1], probabilities=probabilities)
        values = {"validation": val_metrics, "test": test_metrics, "epochs_run": len(history.history["loss"]), "best_epoch": int(np.argmin(history.history["val_loss"])) + 1, "training_seconds": elapsed, "parameters": model.count_params()}
        results[name] = values
        write_json(output / f"{name}_history.json", history.history)
        write_json(output / "metrics.json", results)
        save_plots(name, history.history, test_metrics, output)
        print(f"{name}: test accuracy={test_metrics['accuracy']:.4f}, macro F1={test_metrics['macro_f1']:.4f}", flush=True)
    best = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    majority = int(np.bincount(splits["train"][1]).argmax())
    baseline_predictions = np.full(len(splits["test"][1]), majority)
    summary = {"best_model": best, "selection_metric": "validation macro F1", "majority_baseline": {"label": LABELS[majority], "test_accuracy": float(accuracy_score(splits["test"][1], baseline_predictions)), "test_macro_f1": float(f1_score(splits["test"][1], baseline_predictions, average="macro", zero_division=0))}}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "tweeteval" / DATA_REVISION)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--models", nargs="+", choices=("dense", "cnn", "lstm"), default=["dense", "cnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=15000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=0, help="Smoke-test subsample; 0 uses all cleaned training data")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "max_tokens", "sequence_length", "embedding_dim", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.sequence_length < 3 or args.max_tokens < 3 or args.limit_train < 0:
        parser.error("sequence length and max tokens must be >=3; limit train must be >=0")
    return args


if __name__ == "__main__":
    run(parse_args())
