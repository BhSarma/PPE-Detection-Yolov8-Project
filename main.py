"""
YOLO PPE Detection (reflective_jacket + safety_helmet)

Put this file in the "Yolo Files" folder, next to:
    vest_helmet/            (train, valid, test, data.yaml)
    ppe_sample_video_test.mp4
    ppe_sample_video_test_2.mp4

Run:  python main.py
"""

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ---------------- Settings (change these if you want) ----------------
EPOCHS = 10          # use 30-50 for better accuracy
IMG_SIZE = 416       # smaller = much faster on CPU (640 is slowest)
BATCH = 8            # lower to 4 if you get out-of-memory errors
RUN_NAME = "ppe_project5"
VIDEOS = ["ppe_sample_video_test.mp4", "ppe_sample_video_test_2.mp4"]
# ----------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
DATASET = BASE / "vest_helmet"


def make_fixed_yaml() -> Path:
    """Write data_fixed.yaml with absolute paths so YOLO always finds the images."""
    with open(DATASET / "data.yaml") as f:
        cfg = yaml.safe_load(f)

    fixed = {
        "path": DATASET.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": cfg.get("nc", 2),
        "names": cfg.get("names", ["reflective_jacket", "safety_helmet"]),
    }
    out = DATASET / "data_fixed.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(fixed, f)
    return out


def main():
    # Pick GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available(), "| using device:", device)

    data_yaml = make_fixed_yaml()
    print("Using dataset config:", data_yaml)

    # ---------- 1. Train ----------
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(BASE / "runs"),
        name=RUN_NAME,
        device=device,
        workers=2,       # if you get a Windows error, change this to 0
        cache=False,
        freeze=10,       # freeze backbone: faster on CPU
        val=False,       # skip per-epoch validation (validated at the end)
        verbose=True,
    )

    # Load the best weights saved by training
    best = Path(model.trainer.best)
    print("Best weights:", best)
    model = YOLO(str(best))

    # ---------- 2. Validate ----------
    metrics = model.val(
        data=str(data_yaml),
        device=device,
        workers=0,
        batch=BATCH,
    )
    print("mAP50-95 :", metrics.box.map)
    print("mAP50    :", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall   :", metrics.box.mr)

    # ---------- 3. Predict on videos ----------
    for video in VIDEOS:
        video_path = BASE / video
        if not video_path.exists():
            print("Skipping (not found):", video_path)
            continue
        model.predict(
            source=str(video_path),
            save=True,
            conf=0.25,
            device=device,
            project=str(BASE / "runs" / "detect"),
            name="predict",
            exist_ok=True,
        )

    print("\nDone! Annotated videos are in:", BASE / "runs" / "detect" / "predict")


if __name__ == "__main__":
    # This guard is required on Windows
    main()