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
_BASE_WIDTH = 1280.0


def _class_color(cls_id: int) -> tuple:
    if cls_id not in _CLASS_COLORS:
        rng = np.random.default_rng(cls_id * 17 + 3)
        _CLASS_COLORS[cls_id] = tuple(int(x) for x in rng.integers(60, 240, 3).tolist())
    return _CLASS_COLORS[cls_id]


def _scale(w: int) -> float:
    return max(1.0, w / _BASE_WIDTH)


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
        s = _scale(w)

        box_thick    = max(2, int(2 * s))
        label_font   = 0.5 * s
        label_thick  = max(1, int(1.5 * s))
        banner_h     = int(52 * s)
        banner_font  = 0.75 * s
        banner_thick = max(2, int(2.5 * s))
        cap_font     = 0.6 * s
        cap_thick    = max(1, int(2 * s))
        joint_r      = max(4, int(6 * s))
        conn_thick   = max(2, int(3 * s))

        for det in yolo_results:
            x1, y1, x2, y2 = det["bbox"]
            color = _class_color(det["class_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
            label = f"{det['class']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font, label_thick)
            pad = max(4, int(6 * s))
            cv2.rectangle(frame, (x1, y1 - th - pad), (x1 + tw + pad, y1), color, -1)
            cv2.putText(frame, label, (x1 + pad // 2, y1 - pad // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font, (255, 255, 255), label_thick)

        for hand in hand_results:
            pts = hand["landmarks"]
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(frame, pts[p1], pts[p2], (0, 255, 255), conn_thick)
            for pt in pts:
                cv2.circle(frame, pt, joint_r, (0, 165, 255), -1)

        cv2.rectangle(frame, (0, 0), (w, banner_h), (25, 25, 25), -1)
        banner = f"ACTION: {action.upper().replace('_', ' ')}   |   t = {timestamp:.2f}s"
        cv2.putText(frame, banner, (int(14 * s), int(banner_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, banner_font, (255, 255, 255), banner_thick)

        if caption:
            cap_h = int(52 * s)
            cv2.rectangle(frame, (0, h - cap_h), (w, h), (15, 15, 15), -1)
            cv2.putText(frame, caption[:120], (int(14 * s), h - int(14 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, cap_font, (210, 210, 210), cap_thick)

        return frame
