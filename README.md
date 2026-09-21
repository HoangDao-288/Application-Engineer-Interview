# Application Engineer Interview Assignment

Python solution for the three computer-vision tasks in the Application
Engineer interview test:

- Q1: detect all microswitches in a Part1 tray scene;
- Q2: classify one cropped switch as `roller_level` or `no_level`;
- Q3: inspect detections and classifications interactively in a PyQt5 app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Question 1: switch contour detection

Run the detector on one image:

```bash
python q1_detect_contours.py --input data/Part1/01.png --output output/q1 --save-mask
```

The detector combines dark-body segmentation, connected components, geometric
filtering, and resolution-aware tray regions. Oversized components from
touching switches are split with adaptive erosion and, when needed, seeded
watershed. It writes a labelled detection image and a JSON file containing the
bounding regions; `--save-mask` also writes the processing mask. Bounding
regions are expanded slightly so the Q2 crop keeps attached roller/lever parts.

## Question 2: switch type classification

Exploration of Part2-3 identifies two variants: `roller_level` has the round
external roller/lever, while `no_level` does not. Images `001`--`051` are the
roller variant and `052`--`101` are the no-level variant. The HOG + RBF-SVM
classifier augments the training images with rotations and reports a predicted
label and probability for a single crop.

```bash
python q2_classify_type.py --train --evaluate
python q2_classify_type.py --input data/Part2-3/001.png
```

The repository includes the trained model at `models/q2_hog_svm.joblib`. If it
is absent, prediction and Q3 train it automatically from `data/Part2-3/`.

The held-out evaluation splits original images before rotation augmentation,
avoiding augmented versions of a test image entering the training split.

## Question 3: interactive desktop application

Install the dependencies (including PyQt5), then launch:

```bash
python q3_app.py
```

Use **Open Image** (or drag a scene image onto the image area), press **Run
Detection**, and click a labelled bounding box or an item in the detected
switch list.  The app crops the selected Q1 region and sends that crop to the
same HOG + SVM classifier used in Q2.  The result panel shows the switch ID,
predicted type, and confidence. Clicking outside every box explicitly reports
that no switch was selected. If the saved Q2 model is absent, it is trained
automatically from `data/Part2-3/` on the first selected switch.

Scroll the mouse wheel over the image to zoom around the pointer. Hold the
left mouse button and drag to pan; use **Fit Image** to restore the full-image
view.

The app reuses the public `detect_switches` and `classify_image` APIs from Q1
and Q2. It also selects PyQt5's Qt platform plugins after importing OpenCV to
avoid the common OpenCV/PyQt `xcb` plugin conflict on Linux.

## Project files

```text
q1_detect_contours.py  # Q1 command-line detector
q2_classify_type.py    # Q2 train, evaluate, and predict command-line tool
q3_app.py              # Q3 PyQt5 desktop application
models/q2_hog_svm.joblib
requirements.txt
```
