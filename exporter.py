import json
from pathlib import Path
from typing import List, Dict


class Exporter:
    def export(
        self,
        json_path: str,
        action: str,
        segments: List[Dict],
        frames_data: List[Dict],
        fps: int = 5,
    ):
        output_segments = []
        for seg in segments:
            s, e = seg["start_frame"], min(seg["end_frame"] + 1, len(frames_data))

            obj_best: dict = {}
            for i in range(s, e):
                for det in frames_data[i]["yolo"]:
                    cls = det["class"]
                    if cls not in obj_best or det["confidence"] > obj_best[cls]:
                        obj_best[cls] = det["confidence"]
            objects_detected = [
                {"class": k, "max_confidence": round(v, 3)} for k, v in obj_best.items()
            ]

            contact_set: set = set()
            for i in range(s, e):
                fd = frames_data[i]
                for hand in fd["hands"]:
                    wrist = hand["landmarks"][0]
                    for det in fd["yolo"]:
                        x1, y1, x2, y2 = det["bbox"]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = ((wrist[0] - cx) ** 2 + (wrist[1] - cy) ** 2) ** 0.5
                        if dist < 120:
                            contact_set.add((hand["label"], det["class"]))
            hand_contact = [{"hand": h, "object": o} for h, o in sorted(contact_set)]

            output_segments.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "keyframe_idx": seg["keyframe_idx"],
                "caption": seg.get("caption", ""),
                "objects_detected": objects_detected,
                "hand_contact": hand_contact,
            })

        sidecar = {"action": action, "fps": fps, "segments": output_segments}
        Path(json_path).write_text(json.dumps(sidecar, indent=2))
