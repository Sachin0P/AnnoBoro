import cv2
import numpy as np
from typing import List, Dict


class Segmenter:
    def __init__(self, flow_threshold: float = 1.5, min_segment_frames: int = 5):
        self.flow_threshold = flow_threshold
        self.min_segment_frames = min_segment_frames

    def detect_segments(self, frames: List[np.ndarray], fps: int = 5) -> List[Dict]:
        if len(frames) < 2:
            return [self._make_segment(0, 0, 0, fps)]

        flow_magnitudes = [0.0]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_magnitudes.append(float(np.mean(mag)))
            prev_gray = curr_gray

        active = [m > self.flow_threshold for m in flow_magnitudes]
        raw_segments = []
        in_seg = False
        start = 0
        for i, a in enumerate(active):
            if a and not in_seg:
                start = i
                in_seg = True
            elif not a and in_seg:
                if i - start >= self.min_segment_frames:
                    raw_segments.append((start, i - 1))
                in_seg = False
        if in_seg and len(active) - start >= self.min_segment_frames:
            raw_segments.append((start, len(active) - 1))

        if not raw_segments:
            raw_segments = [(0, len(frames) - 1)]

        result = []
        for s, e in raw_segments:
            seg_flows = flow_magnitudes[s:e + 1]
            kf_offset = int(np.argmax(seg_flows))
            result.append(self._make_segment(s, e, s + kf_offset, fps))
        return result

    @staticmethod
    def _make_segment(start: int, end: int, keyframe: int, fps: int) -> Dict:
        return {
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "keyframe_idx": keyframe,
        }
