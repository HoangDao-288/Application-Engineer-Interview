#!/usr/bin/env python3
"""Run the Q2 HOG + SVM classifier over every supplied Part2-3 crop.

Examples:
    python script/run_q2_batch.py
    python script/run_q2_batch.py --retrain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# This file lives in <project>/script; import the Q2 module from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from q2_classify_type import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    IMAGE_EXTENSIONS,
    hog_features,
    label_from_filename,
    read_image,
    train_and_save,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Q2 classification and validation on every Part2-3 image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 101 labelled Part2-3 images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "q2",
        help="Directory for the JSON batch report",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained joblib model path",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train a fresh model before running the batch",
    )
    return parser.parse_args()


def image_paths(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )
    if len(paths) != 101:
        raise ValueError(f"Expected 101 Part2-3 images, found {len(paths)} in {input_dir}")
    return paths


def main() -> int:
    args = parse_args()
    paths = image_paths(args.input_dir)

    if args.retrain or not args.model.is_file():
        print("Training model before batch evaluation...")
        train_and_save(args.input_dir, args.model)

    artifact: Dict[str, Any] = joblib.load(args.model)
    classifier = artifact["model"]
    predictions: List[str] = []
    expected_labels: List[str] = []
    report_rows: List[Dict[str, Any]] = []

    print(f"Input : {args.input_dir}")
    print(f"Model : {args.model}")
    print("-" * 76)
    for index, path in enumerate(paths, start=1):
        expected = label_from_filename(path)
        features = hog_features(read_image(path)).reshape(1, -1)
        probabilities = classifier.predict_proba(features)[0]
        classes = classifier.classes_
        best_index = int(np.argmax(probabilities))
        predicted = str(classes[best_index])
        confidence = float(probabilities[best_index])
        correct = predicted == expected

        predictions.append(predicted)
        expected_labels.append(expected)
        report_rows.append(
            {
                "input": str(path),
                "expected_label": expected,
                "predicted_label": predicted,
                "confidence": confidence,
                "correct": correct,
                "probabilities": {
                    str(label): float(probability)
                    for label, probability in zip(classes, probabilities)
                },
            }
        )
        status = "OK" if correct else "MISMATCH"
        print(
            f"[{index:03d}/{len(paths)}] {path.name}: {predicted:<12} "
            f"{confidence:6.2%} | expected={expected:<12} {status}"
        )

    accuracy = float(accuracy_score(expected_labels, predictions))
    matrix = confusion_matrix(expected_labels, predictions, labels=CLASS_NAMES)
    print("-" * 76)
    print(f"Accuracy: {accuracy:.2%} ({sum(a == b for a, b in zip(expected_labels, predictions))}/{len(paths)})")
    print(classification_report(expected_labels, predictions, labels=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=expected, columns=predicted):")
    print(f"labels: {', '.join(CLASS_NAMES)}")
    print(matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "batch_predictions.json"
    report_path.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "input_dir": str(args.input_dir),
                "num_images": len(paths),
                "accuracy": accuracy,
                "class_names": list(CLASS_NAMES),
                "confusion_matrix": matrix.tolist(),
                "predictions": report_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Batch report: {report_path}")
    return 0 if accuracy == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
