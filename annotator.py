import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_PATH = "hand_landmarker.task"


def _ensure_hand_model():
    if not os.path.exists(_MODEL_PATH):
        print("[Annotator] Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class Annotator:
    def __init__(self, yolo_model: str = "yolov8n.pt", conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self.yolo = YOLO(yolo_model)
        _ensure_hand_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> tuple:
        results = self.yolo(frame, verbose=False, conf=self.conf_threshold)
        yolo_results = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                if cls_name == "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_results.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name,
                    "confidence": conf,
                    "class_id": cls_id,
                })

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = self.hand_landmarker.detect(mp_image)

        hand_results = []
        if mp_result.hand_landmarks and mp_result.handedness:
            for lm_list, handedness in zip(mp_result.hand_landmarks, mp_result.handedness):
                label = handedness[0].display_name
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
                hand_results.append({"label": label, "landmarks": landmarks})

        return yolo_results, hand_results

    def close(self):
        self.hand_landmarker.close()
