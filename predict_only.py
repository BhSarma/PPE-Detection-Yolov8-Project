from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# Latest trained best.pt ni automatic ga vethukutundi
candidates = sorted(
    (BASE / "runs").rglob("best.pt"),
    key=lambda p: p.stat().st_mtime,
)
if not candidates:
    raise SystemExit("best.pt dorakaledu. runs folder check cheyandi.")
weights = candidates[-1]
print("Using weights:", weights)

model = YOLO(str(weights))

for video in ["ppe_sample_video_test.mp4", "ppe_sample_video_test_2.mp4"]:
    for _ in model.predict(
        source=str(BASE / video),
        stream=True,          # one frame at a time (low memory)
        save=True,
        conf=0.25,
        device="cpu",
        project=str(BASE / "runs" / "detect"),
        name="predict",
        exist_ok=True,
    ):
        pass

print("Done! Check runs/detect/predict")