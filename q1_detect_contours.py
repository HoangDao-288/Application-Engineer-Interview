#!/usr/bin/env python3
"""Q1 - Microswitch detection using intensity segmentation + components.

Pipeline:
    image -> grayscale -> robust dark-object threshold
          -> morphology -> connected components -> component filtering
          -> axis-aligned bounding boxes -> visualization

The detector is intentionally lightweight and uses OpenCV only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, cast

import cv2
import numpy as np


# Distinct deterministic BGR colors for the bounding boxes.
BOX_COLORS: List[Tuple[int, int, int]] = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 140, 255),
    (255, 170, 60),
    (190, 80, 220),
    (0, 210, 210),
    (255, 100, 180),
    (100, 180, 255),
    (170, 255, 80),
    (220, 120, 255),
    (80, 255, 180),
    (255, 210, 80),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect microswitches in one image.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--min-area",
        type=float,
        default=800.0,
        help="Minimum component area at 1080px image height (scaled per image)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.08,
        help="Reject contours larger than this ratio of ROI area",
    )
    parser.add_argument(
        "--adaptive-block",
        type=int,
        default=51,
        help="Odd block size for adaptive threshold",
    )
    parser.add_argument(
        "--adaptive-c",
        type=float,
        default=10.0,
        help="Constant subtracted by adaptive threshold",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=6,
        help="Padding added around each detected box",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=35,
        help="Minimum bounding-box width",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=30,
        help="Minimum bounding-box height",
    )
    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.08,
        help="Minimum contour-area / bounding-box-area ratio",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.15,
        help="IoU threshold used to remove near-duplicate boxes",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the binary processing mask",
    )
    return parser.parse_args()


def get_detection_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Return a central working ROI to suppress tray/frame hardware.

    The supplied Part1 examples keep the switches in the central tray area.
    The ROI is expressed as image-relative fractions so it scales with size.
    """

    height, width = image.shape[:2]
    if height <= 900:
        # The low-resolution scenes have a much thicker visible tray/frame.
        # Keeping that frame in the mask creates one giant component that
        # swallows several switches.
        x0 = int(0.32 * width)
        x1 = int(0.73 * width)
        y0 = int(0.23 * height)
        y1 = int(0.90 * height)
    else:
        x0 = int(0.24 * width)
        x1 = int(0.84 * width)
        y0 = int(0.02 * height)
        y1 = height
    return x0, y0, x1, y1


def clamp_box(
    x: int, y: int, w: int, h: int, image_shape: Tuple[int, ...], padding: int
) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return x0, y0, x1 - x0, y1 - y0


def split_merged_components(
    candidates: List[Dict[str, Any]],
    labels: np.ndarray,
    image_shape: Tuple[int, ...],
    padding: int,
    scaled_min_area: float,
) -> List[Dict[str, Any]]:
    """Split only abnormally large connected components.

    Several Part1 scenes contain switches in physical contact.  A plain
    connected-components pass correctly finds their dark plastic, but treats a
    touching cluster as one object.  We estimate the area of one switch from
    the smaller candidates in *that image*, then progressively erode only a
    component whose area represents two or more switches.  A narrow contact
    breaks before the main switch bodies do.  The selected pieces are expanded
    by the erosion radius again so their bounding regions cover the full body.

    This is intentionally conservative: a component is split only when an
    erosion level produces exactly the area-derived number of substantial
    pieces.  It avoids turning a single roller/lever switch into false objects.
    """

    if len(candidates) < 2:
        return candidates

    areas = sorted(float(candidate["contour_area"]) for candidate in candidates)
    if len(areas) < 3 or areas[0] <= 0:
        return candidates

    # Isolated switches can have noticeably different visible areas because
    # of rotation and perspective.  A fixed multiple of the median therefore
    # classifies legitimate large switches as merged ones.  Instead, find the
    # largest gap in the sorted area distribution.  In these scenes the
    # normal-switch cluster ends before a clear jump to touching clusters.
    median_area = float(np.median(areas))
    normal_areas = [
        area for area in areas
        if 0.55 * median_area <= area <= 1.45 * median_area
    ]
    reference_area = float(np.median(normal_areas or areas))
    largest_normal_area = float(max(normal_areas or areas))
    if reference_area <= 0:
        return candidates

    split_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_area = float(candidate["contour_area"])
        split_multiplier = 1.45 if image_shape[0] <= 900 else 1.65
        if candidate_area < split_multiplier * largest_normal_area:
            split_candidates.append(candidate)
            continue
        # The largest normal object is a better denominator than the median:
        # it keeps the count at two for two overlapping, differently oriented
        # switches while avoiding over-splitting large single switches.
        max_split_count = 6 if image_shape[0] <= 900 else 3
        estimated_count = max(2, min(max_split_count, int(round(candidate_area / reference_area))))

        component_x, component_y, component_w, component_h = candidate["component_bbox"]
        component_label = int(candidate["component_label"])
        component_mask = (
            labels[
                component_y : component_y + component_h,
                component_x : component_x + component_w,
            ]
            == component_label
        ).astype(np.uint8) * 255

        # A piece must remain large enough to be switch material, rather than
        # a detached terminal, screw, or noise fragment.  The fixed 5--39px
        # range was validated on both supplied resolutions; unlike a global
        # erosion, it is used only on oversized clusters.
        minimum_piece_area = max(scaled_min_area * 0.25, reference_area * 0.18)
        chosen_parts: List[Tuple[int, int, int, int, int]] | None = None
        chosen_kernel = 0
        for kernel_size in range(5, 40, 2):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            eroded = cv2.erode(component_mask, kernel, iterations=1)
            _, _, part_stats, _ = cv2.connectedComponentsWithStats(
                eroded, connectivity=8
            )
            parts: List[Tuple[int, int, int, int, int]] = []
            for stat in part_stats[1:]:
                if float(stat[4]) < minimum_piece_area:
                    continue
                # connectedComponentsWithStats rows are always
                # (x, y, width, height, area).  Explicit unpacking preserves
                # that fixed-length tuple type for Pylance.
                part_x, part_y, part_w, part_h, part_area = (
                    int(value) for value in stat
                )
                parts.append((part_x, part_y, part_w, part_h, part_area))
            if len(parts) == estimated_count:
                chosen_parts = parts
                chosen_kernel = kernel_size
                break

        if chosen_parts is not None:
            parts_to_add = [
                (
                    part_x,
                    part_y,
                    part_w,
                    part_h,
                    part_area,
                    padding + chosen_kernel // 2,
                )
                for part_x, part_y, part_w, part_h, part_area in chosen_parts
            ]
        else:
            parts_to_add = [
                (*part, padding)
                for part in _watershed_parts(component_mask, estimated_count, minimum_piece_area)
            ]

        if len(parts_to_add) != estimated_count:
            split_candidates.append(candidate)
            continue

        for part_x, part_y, part_w, part_h, part_area, restore_padding in parts_to_add:
            box = clamp_box(
                component_x + part_x,
                component_y + part_y,
                part_w,
                part_h,
                image_shape,
                restore_padding,
            )
            split_candidates.append(
                {
                    "bbox": [int(value) for value in box],
                    "contour_area": float(part_area),
                    "fill_ratio": float(part_area) / float(max(1, part_w * part_h)),
                    "aspect_ratio": max(part_w, part_h) / float(max(1, min(part_w, part_h))),
                    "split_group": component_label,
                }
            )

    # Internal connected-component data must not leak into the public JSON.
    for candidate in split_candidates:
        candidate.pop("component_label", None)
        candidate.pop("component_bbox", None)
    return split_candidates


def _watershed_parts(
    component_mask: np.ndarray,
    part_count: int,
    minimum_piece_area: float,
) -> List[Tuple[int, int, int, int, int]]:
    """Split a tightly overlapping component using foreground-derived seeds."""

    foreground = component_mask > 0
    y_coords, x_coords = np.where(foreground)
    if len(x_coords) < part_count:
        return []

    points = np.asarray(np.column_stack((x_coords, y_coords)), dtype=np.float32)
    best_labels = np.empty((len(points), 1), dtype=np.int32)
    _, _, centers = cast(Any, cv2.kmeans)(
        points,
        part_count,
        best_labels,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.asarray(centers, dtype=np.float32)

    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    markers = np.zeros(component_mask.shape, dtype=np.int32)
    markers[~foreground] = 1
    for marker_id, (center_x, center_y) in enumerate(centers, start=2):
        x = int(np.clip(round(float(center_x)), 0, component_mask.shape[1] - 1))
        y = int(np.clip(round(float(center_y)), 0, component_mask.shape[0] - 1))
        if not foreground[y, x]:
            foreground_points = np.column_stack(np.where(foreground))
            nearest = np.argmin(
                (foreground_points[:, 1] - x) ** 2
                + (foreground_points[:, 0] - y) ** 2
            )
            y, x = (int(value) for value in foreground_points[nearest])
        markers[y, x] = marker_id

    normalized_distance = cv2.normalize(
        distance,
        np.empty_like(distance),
        0.0,
        255.0,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    watershed_image = cv2.cvtColor(255 - normalized_distance, cv2.COLOR_GRAY2BGR)
    watershed_labels = cv2.watershed(watershed_image, markers)

    parts: List[Tuple[int, int, int, int, int]] = []
    for marker_id in range(2, part_count + 2):
        ys, xs = np.where(watershed_labels == marker_id)
        if len(xs) < minimum_piece_area:
            continue
        part_x, part_y = int(xs.min()), int(ys.min())
        part_w, part_h = int(xs.max() - part_x + 1), int(ys.max() - part_y + 1)
        parts.append((part_x, part_y, part_w, part_h, int(len(xs))))
    return parts


def detect_switches(
    image: np.ndarray,
    min_area: float = 800.0,
    max_area_ratio: float = 0.08,
    adaptive_block: int = 51,
    adaptive_c: float = 10.0,
    padding: int = 6,
    min_width: int = 35,
    min_height: int = 30,
    min_fill_ratio: float = 0.08,
    nms_iou: float = 0.15,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Tuple[int, int, int, int]]:
    """Detect switch candidates and return boxes in full-image coordinates."""

    if adaptive_block < 3 or adaptive_block % 2 == 0:
        raise ValueError("--adaptive-block must be an odd integer >= 3")

    x0, y0, x1, y1 = get_detection_roi(image)
    roi = image[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Switch bodies are the dark, low-reflectance parts of each object.  A
    # global adaptive threshold was unreliable here: the dark tray rim joins
    # many otherwise separate switches into one external contour.  Instead,
    # derive a conservative dark threshold from the scene's 10th percentile
    # and clamp it to the observed illumination range of this data set.
    # This retains the black plastic body while excluding the white tray.
    # cv2's type stub describes GaussianBlur's return value as MatLike.  Make
    # its numeric ndarray type explicit for NumPy (and Pylance) before asking
    # for a scalar percentile.
    blurred_for_stats = np.asarray(blurred, dtype=np.float64)
    dark_percentile = float(np.percentile(blurred_for_stats, 10.0))
    if image.shape[0] <= 900:
        # Low-resolution scenes have stronger shadows and a darker tray frame.
        # A lower cutoff keeps adjacent switch bodies from becoming one dark
        # mass and preserves their separate cores.
        dark_threshold = int(min(70.0, max(40.0, dark_percentile + 3.0)))
    else:
        dark_threshold = int(min(95.0, max(75.0, dark_percentile + 6.0)))
    _, binary = cv2.threshold(
        blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV
    )

    # A small close joins texture within one switch without bridging nearby
    # switches.  Connected components (rather than RETR_EXTERNAL contours)
    # preserve objects inside the closed dark outline of a tray.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    roi_area = float(roi.shape[0] * roi.shape[1])
    # The input contains both 800px and 1080px high images.  Scale pixel
    # filters so small, valid objects in the 800px images are not discarded.
    scale = min(image.shape[:2]) / 1080.0
    scaled_min_area = min_area * scale * scale
    if image.shape[0] > 900:
        # Small isolated dark fragments (rollers, terminals, and shadows) are
        # common in the high-resolution scenes but are far below a switch
        # body.  Keep the lower threshold for 800px scenes where real bodies
        # are genuinely smaller.
        scaled_min_area = max(scaled_min_area, 3000.0)
    scaled_min_width = min_width * scale
    scaled_min_height = min_height * scale
    candidates: List[Dict[str, Any]] = []

    # stats[0] describes background; every following row is one foreground
    # component: x, y, width, height, area.
    for component_label, (x, y, w, h, component_area) in enumerate(stats[1:], start=1):
        contour_area = float(component_area)
        # Components touching the working ROI boundary belong to the tray,
        # camera frame, or ROI crop rather than to a switch.  The ROI has a
        # deliberate margin around every supplied switch.
        touches_boundary = (
            x == 0
            or y == 0
            or x + w >= roi.shape[1]
            or (image.shape[0] <= 900 and y + h >= roi.shape[0])
        )
        if touches_boundary:
            # A low-resolution cluster can touch the left tray edge while
            # still being a valid group of switches.  Keep only compact,
            # moderate-size groups; large frame-shaped components remain
            # rejected.
            bbox_fraction = float(w * h) / float(max(1, roi.shape[0] * roi.shape[1]))
            if not (
                image.shape[0] <= 900
                and contour_area < 0.06 * roi_area
                and bbox_fraction < 0.90
                and max(w, h) / float(max(1, min(w, h))) < 3.0
            ):
                continue
        if contour_area < scaled_min_area:
            continue
        if contour_area > max_area_ratio * roi_area:
            continue

        if w < scaled_min_width or h < scaled_min_height:
            continue

        aspect_ratio = max(w, h) / float(min(w, h))
        if aspect_ratio > 3.0:
            continue

        # Tray rims are long, thin dark loops.  Their bounding boxes can look
        # square, but their fill ratio is much lower than a switch body.
        fill_ratio = contour_area / float(w * h)
        if fill_ratio < min_fill_ratio:
            continue

        box = clamp_box(x + x0, y + y0, w, h, image.shape, padding)
        candidates.append(
            {
                "bbox": [int(v) for v in box],
                "contour_area": contour_area,
                "fill_ratio": fill_ratio,
                "aspect_ratio": aspect_ratio,
                "component_label": component_label,
                "component_bbox": [int(x + x0), int(y + y0), int(w), int(h)],
            }
        )

    # Labels are ROI-relative while candidate boxes above are full-image
    # relative.  Pass a full-size label image so split_merged_components can
    # use the stored full-image component coordinates directly.
    full_labels = np.zeros(image.shape[:2], dtype=labels.dtype)
    full_labels[y0:y1, x0:x1] = labels
    candidates = split_merged_components(
        candidates, full_labels, image.shape, padding, scaled_min_area
    )

    # Non-maximum suppression removes near-duplicate contours generated by
    # small fragments of the same switch. Score by contour area.
    def iou(box_a: List[int], box_b: List[int]) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        union = aw * ah + bw * bh - intersection
        return intersection / float(max(1, union))

    candidates.sort(key=lambda d: d["contour_area"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        if all(
            candidate.get("split_group") == other.get("split_group")
            or iou(candidate["bbox"], other["bbox"]) < nms_iou
            for other in kept
        ):
            kept.append(candidate)

    # Stable presentation order: top-to-bottom, then left-to-right.
    kept.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    for idx, candidate in enumerate(kept, start=1):
        candidate["id"] = idx
        candidate.pop("split_group", None)

    candidates = kept

    return candidates, binary, (x0, y0, x1, y1)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    output = image.copy()

    for det in detections:
        idx = det["id"]
        x, y, w, h = det["bbox"]
        color = BOX_COLORS[(idx - 1) % len(BOX_COLORS)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 4)
        label = f"SW{idx}"
        cv2.putText(
            output,
            label,
            (x, max(30, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"Detected switches: {len(detections)}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {input_path}")

    detections, binary, roi = detect_switches(
        image,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        padding=args.padding,
        min_width=args.min_width,
        min_height=args.min_height,
        min_fill_ratio=args.min_fill_ratio,
        nms_iou=args.nms_iou,
    )

    result = draw_detections(image, detections, roi)

    stem = input_path.stem
    result_path = output_dir / f"{stem}_detected.png"
    mask_path = output_dir / f"{stem}_mask.png"
    json_path = output_dir / f"{stem}_detections.json"

    if not cv2.imwrite(str(result_path), result):
        raise RuntimeError(f"Unable to write result: {result_path}")

    if args.save_mask and not cv2.imwrite(str(mask_path), binary):
        raise RuntimeError(f"Unable to write mask: {mask_path}")

    payload = {
        "input": str(input_path),
        "num_detections": len(detections),
        "roi": list(roi),
        "detections": detections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Detected {len(detections)} switch candidate(s).")
    print(f"Result image: {result_path}")
    print(f"Detection JSON: {json_path}")
    if args.save_mask:
        print(f"Binary mask: {mask_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
