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

Run it on the full Part1 dataset:

```bash
python script/run_q1_batch.py
```

The detector segments dark switch bodies, uses connected components to retain
objects enclosed by tray contours, and scales its geometric filters for both
image resolutions in the supplied set. Results are written to `output/q1/`.
