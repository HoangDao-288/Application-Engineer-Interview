#!/usr/bin/env python3
"""Q2 - Classify a cropped microswitch as ``no_level`` or ``roller_level``.

The Part2-3 samples are visually ordered by type: 001--051 contain the
external round roller/lever mechanism (``roller_level``), while 052--101 do
not (``no_level``).  The classifier uses grayscale HOG descriptors and an RBF
SVM.  During training each image is rotated to make HOG robust to the freely
rotated crops in the supplied data and in Q3.

Examples:
    python q2_classify_type.py --train --evaluate
    python q2_classify_type.py --input data/Part2-3/001.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Part2-3"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "q2_hog_svm.joblib"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_NAMES = ("no_level", "roller_level")

# HOG needs a fixed canvas.  96 is divisible by the block/cell sizes below.
IMAGE_SIZE = (96, 96)
HOG_CELL_SIZE = 8
HOG_BINS = 9
# The foreground can be at any angle, so expose the SVM to rotations at train
# time.  The original is included by the zero-degree angle.
TRAIN_ROTATIONS = tuple(range(0, 360, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a cropped microswitch with HOG + SVM."
    )
    parser.add_argument("--input", help="One cropped switch image to classify")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and save a model using --train-dir",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Print an 80/20 held-out evaluation before training the final model",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain before prediction even when --model already exists",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 101 Part2-3 images",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used to save/load the trained joblib model",
    )
    args = parser.parse_args()
    if not args.input and not args.train and not args.evaluate:
        parser.error("provide --input and/or --train (use --help for examples)")
    return args


def label_from_filename(path: Path) -> str:
    """Return the manually verified Part2-3 label encoded by file ordering."""

    try:
        image_id = int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric Part2-3 filename, got: {path.name}") from exc

    if 1 <= image_id <= 51:
        return "roller_level"
    if 52 <= image_id <= 101:
        return "no_level"
    raise ValueError(
        f"{path.name} is outside the labelled Part2-3 range 001.png--101.png"
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the image centre without changing the HOG canvas size."""

    if angle == 0:
        return image
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def hog_features(image: np.ndarray) -> np.ndarray:
    """Convert one BGR image to a normalized, fixed-length HOG feature row."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Equalization reduces the strong sample-to-sample illumination variation.
    gray = cv2.equalizeHist(gray)
    # Implement the small HOG descriptor with NumPy rather than OpenCV's
    # HOGDescriptor: this keeps the script compatible with both OpenCV 4 and
    # OpenCV 5, whose Python build no longer exposes that class.
    gray_float = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, orientation = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    orientation = np.mod(orientation, 180.0)  # unsigned gradient directions

    cells_per_axis = IMAGE_SIZE[0] // HOG_CELL_SIZE
    histograms = np.zeros((cells_per_axis, cells_per_axis, HOG_BINS), dtype=np.float32)
    bin_position = orientation / (180.0 / HOG_BINS)
    lower_bin = np.floor(bin_position).astype(np.intp) % HOG_BINS
    upper_weight = bin_position - np.floor(bin_position)
    upper_bin = (lower_bin + 1) % HOG_BINS

    cell_y, cell_x = np.indices(gray.shape)
    cell_y = (cell_y // HOG_CELL_SIZE).reshape(-1)
    cell_x = (cell_x // HOG_CELL_SIZE).reshape(-1)
    magnitude = magnitude.reshape(-1)
    lower_bin = lower_bin.reshape(-1)
    upper_bin = upper_bin.reshape(-1)
    upper_weight = upper_weight.reshape(-1)
    np.add.at(
        histograms,
        (cell_y, cell_x, lower_bin),
        magnitude * (1.0 - upper_weight),
    )
    np.add.at(histograms, (cell_y, cell_x, upper_bin), magnitude * upper_weight)

    blocks: List[np.ndarray] = []
    for y in range(cells_per_axis - 1):
        for x in range(cells_per_axis - 1):
            block = histograms[y : y + 2, x : x + 2].reshape(-1)
            blocks.append(block / np.sqrt(np.dot(block, block) + 1e-6))
    return np.concatenate(blocks).astype(np.float32)


def load_labeled_images(train_dir: Path) -> List[Tuple[Path, str]]:
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")

    samples = [
        (path, label_from_filename(path))
        for path in sorted(train_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(samples) != 101:
        raise ValueError(
            f"Expected exactly 101 labelled Part2-3 images, found {len(samples)} in {train_dir}"
        )
    return samples


def build_features(
    samples: Iterable[Tuple[Path, str]], rotations: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    for path, label in samples:
        image = read_image(path)
        for angle in rotations:
            features.append(hog_features(rotate_image(image, angle)))
            labels.append(label)
    return np.vstack(features), np.asarray(labels)


def build_classifier() -> Any:
    """Create the requested probability-enabled HOG + SVM pipeline."""

    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    )


def evaluate(samples: Sequence[Tuple[Path, str]]) -> float:
    """Evaluate with images held out before rotation augmentation."""

    indices = np.arange(len(samples))
    labels = np.asarray([label for _, label in samples])
    train_indices, test_indices = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=42
    )
    train_samples = [samples[index] for index in train_indices]
    test_samples = [samples[index] for index in test_indices]

    x_train, y_train = build_features(train_samples, TRAIN_ROTATIONS)
    x_test, y_test = build_features(test_samples, (0,))
    classifier = build_classifier()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    print(f"Held-out accuracy: {accuracy:.3f} ({len(y_test)} images)")
    print(classification_report(y_test, predictions, labels=CLASS_NAMES, zero_division=0))
    return accuracy


def train_and_save(train_dir: Path, model_path: Path) -> None:
    samples = load_labeled_images(train_dir)
    features, labels = build_features(samples, TRAIN_ROTATIONS)
    classifier = build_classifier()
    classifier.fit(features, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "feature_extractor": "grayscale + equalizeHist + HOG(96x96, 8px cells, 9 bins)",
            "training_samples": len(samples),
            "rotations": TRAIN_ROTATIONS,
        },
        model_path,
    )
    print(f"Trained on {len(samples)} images ({len(features)} augmented samples).")
    print(f"Saved model: {model_path}")


def predict(image_path: Path, model_path: Path, train_dir: Path, retrain: bool) -> None:
    if retrain or not model_path.is_file():
        print("No trained model found; training HOG + SVM model first.")
        train_and_save(train_dir, model_path)

    artifact = joblib.load(model_path)
    classifier = artifact["model"]
    feature_row = hog_features(read_image(image_path)).reshape(1, -1)
    probabilities = classifier.predict_proba(feature_row)[0]
    classes = classifier.classes_
    best_index = int(np.argmax(probabilities))

    print(f"Input: {image_path}")
    print(f"Predicted type: {classes[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.2%}")
    print(
        "Probabilities: "
        + ", ".join(f"{label}={probability:.2%}" for label, probability in zip(classes, probabilities))
    )


def main() -> int:
    args = parse_args()
    samples: List[Tuple[Path, str]] | None = None
    if args.evaluate:
        samples = load_labeled_images(args.train_dir)
        evaluate(samples)
    if args.train:
        train_and_save(args.train_dir, args.model)
    if args.input:
        predict(Path(args.input), args.model, args.train_dir, args.force_retrain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
