"""Small, real-data TensorFlow/WFDB smoke test for a locally prepared MIT-BIH copy.

Run from the repository root after installing the optional test dependencies:
    python scripts/smoke_test.py --dataset mit-bih-arrhythmia-database-1.0.0

This checks integration, not the scientific performance of an eight-class model.
No generated images, models, or logs are added to the repository.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("mit-bih-arrhythmia-database-1.0.0"),
        help="Folder with all 48 MIT-BIH v1.0.0 .atr/.dat/.hea record triplets",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the official PhysioNet dataset when --dataset is missing",
    )
    parser.add_argument(
        "--beats", type=int, default=2,
        help="Maximum ECG events for each chosen record (default: 2)",
    )
    args = parser.parse_args()
    if args.beats < 1:
        parser.error("--beats must be positive")

    print("Python:", platform.python_version())
    for dependency in ("tensorflow", "wfdb", "numpy", "opencv-python-headless",
                       "matplotlib", "scikit-learn", "imutils"):
        try:
            print(f"{dependency}: {version(dependency)}")
        except PackageNotFoundError:
            print(f"{dependency}: NOT INSTALLED")

    # Imports are deferred, allowing --help to work without ML dependencies.
    import numpy as np
    import tensorflow as tf
    import wfdb  # noqa: F401 - WFDB is used by the imported data module.

    from ecg_classification.data import (
        download_mit_bih, generate_record_images, load_images, record_class_labels,
        record_ids, split_records, verify_mit_bih_records,
    )
    from ecg_classification.evaluation import evaluate_predictions
    from ecg_classification.model import predict_model, train_model

    print("TensorFlow devices:", tf.config.list_physical_devices())
    if not args.dataset.exists() and args.download:
        print("Downloading MIT-BIH v1.0.0 and checking SHA-256 hashes...")
        download_mit_bih(args.dataset)
    verify_mit_bih_records(args.dataset)
    records = record_ids(args.dataset)
    if len(records) != 48:
        raise ValueError("Expected all 48 MIT-BIH records")
    labels = record_class_labels(args.dataset, records)
    partitions = split_records(records, seed=0, record_labels=labels)
    print("Record split sizes:", {key: len(value) for key, value in partitions.items()})
    with tempfile.TemporaryDirectory(prefix="ecg-smoke-") as temp:
        temporary = Path(temp)
        images_directory = temporary / "images"
        samples = {}
        for split, split_records_list in partitions.items():
            chosen = split_records_list[0]
            generated = generate_record_images(
                args.dataset, images_directory, chosen, split,
                augmentation=False, maximum_beats=args.beats,
            )
            samples[split] = load_images(images_directory, split, [chosen])
            print(f"{split}: record={chosen}; generated={generated}; "
                  f"base images={len(samples[split])}")
            if not samples[split]:
                raise RuntimeError(f"The {split} smoke subset is empty")
        checkpoint = temporary / "cnn.keras"
        train_model(
            samples["train"], samples["validation"],
            epochs=1, batch_size=2, checkpoint=checkpoint,
            augmentation=False, seed=0,
        )
        if not checkpoint.is_file():
            raise RuntimeError("CNN training did not write a checkpoint")
        model = tf.keras.models.load_model(checkpoint)
        predictions = predict_model(model, samples["test"], batch_size=2)
        if predictions.shape != (len(samples["test"]), 8):
            raise RuntimeError("Unexpected CNN prediction dimensions")
        if not np.isfinite(predictions).all():
            raise RuntimeError("Non-finite CNN predictions")
        results = evaluate_predictions(predictions, samples["test"])
        print("Test images:", len(samples["test"]))
        print("Class support:", {name: metric["support"]
                                 for name, metric in results["per_class"].items()})
        print("Smoke accuracy (NOT scientific performance):", results["accuracy"])
        print("PASS: integrity, WFDB reading, image generation, TensorFlow training, "
              "checkpoint loading, inference, evaluation")


if __name__ == "__main__":
    main()
