from pathlib import Path

from ultralytics import YOLO

# ---------------- Settings ----------------
TIME_HOURS = 6      # 4 = faster, 8-10 = better (run overnight)
IMG_SIZE = 512      # 416 = faster, 640 = best accuracy but slowest on CPU
BATCH = 8           # use 4 if "out of memory" error comes
# ------------------------------------------

BASE = Path(__file__).resolve().parent
DATA = BASE / "vest_helmet" / "data_fixed.yaml"

# Start from your previously trained model
found = sorted(
    (p for p in (BASE / "runs").rglob("best.pt") if "ppe_final" not in str(p)),
    key=lambda p: p.stat().st_mtime,
)
start = found[-1] if found else BASE / "yolov8n.pt"
print("Starting from:", start)


def main():
    model = YOLO(str(start))
    model.train(
        data=str(DATA),
        time=TIME_HOURS,
        epochs=100,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=2,            # if a Windows error comes, change to 0
        project=str(BASE / "runs"),
        name="ppe_final",
        exist_ok=True,
        val=False,
        patience=0,
        cos_lr=True,
        close_mosaic=3,
    )

    # Final test of the new model
    best = BASE / "runs" / "ppe_final" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA), imgsz=IMG_SIZE, device="cpu", workers=0, batch=BATCH)
    print("\nmAP50-95 :", metrics.box.map)
    print("mAP50    :", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall   :", metrics.box.mr)
    print("\nDone! New model: runs/ppe_final/weights/best.pt")


if __name__ == "__main__":
    main()