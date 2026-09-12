"""MobileNet-SSD object detector using OpenCV DNN.

A real pretrained object detector. Runs on CPU.
Replaces the color-segmentation detector from `visual_detector.py`.
"""
from __future__ import annotations
import os
from typing import List

import cv2
import numpy as np


# MobileNet-SSD class labels (VOC-trained subset used by OpenCV DNN).
# We only care about the subset that maps to our grid world's objects.
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Map detector class names to our world's object types.
# MobileNet-SSD doesn't know "Mug" or "Apple" — we handle that separately.
CLASS_MAP = {
    "diningtable": "Table",
    "sofa":        "Sofa",
    "tvmonitor":   "TV",
    "chair":       "Chair",
    "bottle":      "Mug",     # bottle is the closest match to our Mug
}


class MobileNetDetector:
    def __init__(
        self,
        prototxt_path: str = "models/MobileNetSSD_deploy.prototxt",
        weights_path: str = "models/mobilenet_iter_73000.caffemodel",
        confidence_threshold: float = 0.35,
    ):
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(
                f"Prototxt not found: {prototxt_path}. "
                f"Run the download commands from the README."
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. "
                f"Run the download commands from the README."
            )

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        self.threshold = confidence_threshold

    def detect(self, img: np.ndarray) -> List[dict]:
        """
        Run detection on an (H, W, 3) uint8 RGB image.

        Returns list of dicts: {class_name, object_type, confidence,
                                pixel_x, pixel_y, pixel_w, pixel_h}
        """
        H, W = img.shape[:2]

        # MobileNet-SSD expects 300x300 input with mean 127.5, scale 1/127.5
        # Our renderer returns RGB; OpenCV's Caffe pipeline expects BGR.
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 127.5,
            size=(300, 300), mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(CLASSES):
                continue
            class_name = CLASSES[class_id]
            if class_name == "background":
                continue

            # Bounding box in normalized coords → pixel coords
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            results.append({
                "class_name": class_name,
                "object_type": CLASS_MAP.get(class_name, class_name),
                "confidence": conf,
                "pixel_x": cx,
                "pixel_y": cy,
                "pixel_w": w,
                "pixel_h": h,
            })

        return results