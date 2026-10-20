# PPE Detection using YOLOv8

Detects *safety helmets* and *reflective jackets* in images and videos, with a Streamlit web app for easy testing.

## Results
| Metric | Value |
|---|---|
| mAP50 | 0.94 |
| mAP50-95 | 0.69 |
| Precision | 0.89 |
| Recall | 0.88 |

Validated on 1,427 images (reflective_jacket and safety_helmet classes).

## Demo


![Detection 1](Project_Results/IMAGE_NAME_1.png)


Demo video: [Watch here](https://github.com/BhSarma/PPE-Detection-Yolov8-Project/raw/main/Project_Results/Video_Result_small.mp4)
## Tech Stack
Python, YOLOv8 (Ultralytics), OpenCV, Streamlit

## Dataset
Roboflow Universe: [vest-helmet](https://universe.roboflow.com/uhhh/vest-helmet-wlbch/dataset/1) (CC BY 4.0).
The dataset is not included in this repo because of its size.

## How to Run
bash
pip install -r requirements.txt
streamlit run app.py

Upload an image or video. The app loads the trained model from runs/ppe_final/weights/best.pt.

## Training
- main.py: initial training, validation and prediction.
- train_better.py: fine-tunes the trained model (full model unfrozen, 512 px, time-limited) for better accuracy on CPU.
- predict_only.py: runs prediction on sample videos.

## Limitations
- Works best on ground-level / eye-level views.
- Aerial (top-view) footage can cause confusion, since the training data has few aerial images.

## Future Work
- Add aerial images to the dataset.
- Add a no_helmet class and violation alerts.
- Live camera / CCTV support.
