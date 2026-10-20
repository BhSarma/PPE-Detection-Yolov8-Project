import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="PPE Detection", page_icon="🦺")
st.title("🦺 PPE Detection (Helmet + Reflective Jacket)")
st.write("Upload an image or video. The AI will find safety helmets and reflective jackets.")

BASE = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    # Finds the newest best.pt inside the runs folder automatically
    found = sorted((BASE / "runs").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        st.error("best.pt not found. Keep app.py in 'Yolo Files' folder (next to 'runs').")
        st.stop()
    return YOLO(str(found[-1])), found[-1]


model, weights_path = load_model()
st.caption(f"Model: {weights_path.relative_to(BASE)}")

conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)


def count_objects(result):
    names = result.names
    counts = {}
    for c in result.boxes.cls.tolist():
        n = names[int(c)]
        counts[n] = counts.get(n, 0) + 1
    return counts


if mode == "Image":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        result = model.predict(np.array(img), conf=conf, device="cpu")[0]
        annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
        st.image(annotated, caption="Detection result", use_container_width=True)
        counts = count_objects(result)
        if counts:
            for name, n in counts.items():
                st.success(f"{name}: {n}")
        else:
            st.warning("Nothing detected.")

else:
    file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if file and st.button("Start detection"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_path = str(BASE / "web_output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_box = st.empty()
        info_box = st.empty()
        bar = st.progress(0)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, device="cpu", imgsz=416, verbose=False)[0]
            annotated = result.plot()
            writer.write(annotated)
            i += 1
            if i % 3 == 0:  # update screen every 3rd frame (faster)
                frame_box.image(annotated[:, :, ::-1], use_container_width=True)
                info_box.write(count_objects(result))
            bar.progress(min(i / total, 1.0))
        cap.release()
        writer.release()

        st.success("Done! Download the annotated video below.")
        with open(out_path, "rb") as f:
            st.download_button("⬇ Download video", f, file_name="ppe_result.mp4", mime="video/mp4")
