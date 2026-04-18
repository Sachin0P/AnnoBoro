import cv2
import numpy as np
from typing import List, Dict

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_CLASS_COLORS: dict = {}


def _class_color(cls_id: int) -> tuple:
    if cls_id not in _CLASS_COLORS:
        rng = np.random.default_rng(cls_id * 17 + 3)
        _CLASS_COLORS[cls_id] = tuple(int(x) for x in rng.integers(60, 240, 3).tolist())
    return _CLASS_COLORS[cls_id]


class Renderer:
    def render(
        self,
        frame: np.ndarray,
        yolo_results: List[Dict],
        hand_results: List[Dict],
        action: str,
        caption: str,
        timestamp: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        for det in yolo_results:
            x1, y1, x2, y2 = det["bbox"]
            color = _class_color(det["class_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for hand in hand_results:
            pts = hand["landmarks"]
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(frame, pts[p1], pts[p2], (0, 255, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 165, 255), -1)

        cv2.rectangle(frame, (0, 0), (w, 38), (25, 25, 25), -1)
        banner = f"ACTION: {action.upper().replace('_', ' ')}   |   t = {timestamp:.2f}s"
        cv2.putText(frame, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if caption:
            cv2.rectangle(frame, (0, h - 38), (w, h), (15, 15, 15), -1)
            cv2.putText(
                frame, caption[:110], (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
            )

        return frame
