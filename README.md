# Application Engineer Interview Assignment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The supplied interview images must be placed in `data/Part1/`.

## Question 1: switch contour detection

Run the detector on one image:

```bash
python q1_detect_contours.py --input data/Part1/01.png --output output/q1 --save-mask
```

The detector segments dark switch bodies, uses connected components to retain
objects enclosed by tray contours, and scales its geometric filters for both
image resolutions in the supplied set. Results are written to `output/q1/`.

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

The model is saved as `models/q2_hog_svm.joblib`. The second command trains it
automatically if it has not been created yet.

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

The app explicitly selects PyQt5's Qt platform plugins after importing
OpenCV, so it avoids the common OpenCV/PyQt `xcb` plugin conflict on Linux.
