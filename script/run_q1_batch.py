#!/usr/bin/env python3
"""Run Q1 switch detection on all images in data/Part1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# run_q1_batch.py is inside <project>/script/
# q1_detect_contours.py is inside <project>/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from q1_detect_contours import detect_switches, draw_detections


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Q1 detection on all images in a folder."
    )

    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "data" / "Part1"),
        help="Input image folder. Default: <project>/data/Part1",
    )

    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "q1"),
        help="Output folder. Default: <project>/output/q1",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist:\n  {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )

    if not image_paths:
        print(f"ERROR: No images found in:\n  {input_dir}")
        return 1

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(image_paths)} image(s)")
    print("-" * 60)

    success_count = 0
    total_detections = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(
                f"[{index:02d}/{len(image_paths)}] "
                f"ERROR: Cannot read {image_path.name}"
            )
            continue

        try:
            detections, _, roi = detect_switches(image)
            result = draw_detections(image, detections, roi)

            result_path = output_dir / f"{image_path.stem}_detected.png"

            if not cv2.imwrite(str(result_path), result):
                raise RuntimeError("cv2.imwrite() failed")

            success_count += 1
            total_detections += len(detections)

            print(
                f"[{index:02d}/{len(image_paths)}] "
                f"{image_path.name}: "
                f"{len(detections)} detection(s) "
                f"-> {result_path.name}"
            )

        except Exception as exc:
            print(
                f"[{index:02d}/{len(image_paths)}] "
                f"ERROR processing {image_path.name}: {exc}"
            )

    print("-" * 60)
    print(f"Processed successfully: {success_count}/{len(image_paths)}")
    print(f"Total detections      : {total_detections}")
    print(f"Results saved to      : {output_dir}")

    return 0 if success_count == len(image_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
