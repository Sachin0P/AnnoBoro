import argparse
import subprocess
import tempfile
import os
import cv2
from pathlib import Path
from tqdm import tqdm

from annotator import Annotator
from segmenter import Segmenter
from captioner import Captioner
from renderer import Renderer
from exporter import Exporter

VALID_ACTIONS = ["dishwashing", "cooking", "chopping", "clothes_folding", "mopping", "sweeping"]


def extract_frames(video_path: str, output_dir: str, fps: int = 5) -> list:
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", pattern, "-y"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(str(f) for f in Path(output_dir).glob("frame_*.jpg"))


def main():
    parser = argparse.ArgumentParser(description="RoboAnnotate: Domestic Action Video Annotator")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--action", required=True, choices=VALID_ACTIONS, help="Action class label")
    parser.add_argument("--output", default=None, help="Output .mp4 path (default: <input>_annotated.mp4)")
    args = parser.parse_args()

    input_path = args.input
    action = args.action
    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_annotated.mp4")
    json_path = output_path.replace(".mp4", ".json")

    print(f"\n[RoboAnnotate] Input  : {input_path}")
    print(f"[RoboAnnotate] Action : {action}")
    print(f"[RoboAnnotate] Output : {output_path}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("[1/6] Extracting frames at 5 fps via FFmpeg ...")
        frame_paths = extract_frames(input_path, tmpdir)
        print(f"      → {len(frame_paths)} frames extracted")

        if not frame_paths:
            print("[ERROR] No frames extracted. Check the input file and FFmpeg installation.")
            return

        print("[2/6] Loading models ...")
        annotator = Annotator()
        segmenter = Segmenter()
        captioner = Captioner()
        renderer = Renderer()
        exporter = Exporter()

        print("[3/6] Running YOLO + MediaPipe on each frame ...")
        frames_data = []
        raw_frames = []
        for fp in tqdm(frame_paths, desc="  Annotating", unit="frame"):
            img = cv2.imread(fp)
            yolo_results, hand_results = annotator.process(img)
            frames_data.append({"yolo": yolo_results, "hands": hand_results})
            raw_frames.append(img)
        annotator.close()

        print("[4/6] Detecting action segments via optical flow ...")
        segments = segmenter.detect_segments(raw_frames, fps=5)
        print(f"      → {len(segments)} segment(s) detected")

        print("[5/6] Generating BLIP captions for segment keyframes ...")
        for seg in tqdm(segments, desc="  Captioning", unit="seg"):
            kf = seg["keyframe_idx"]
            seg["caption"] = captioner.caption(raw_frames[kf])
            print(f"      keyframe {kf} → \"{seg['caption']}\"")

        print("[6/6] Rendering annotated video ...")
        h, w = raw_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 5, (w, h))

        frame_to_caption: dict = {}
        for seg in segments:
            cap = seg.get("caption", "")
            for i in range(seg["start_frame"], seg["end_frame"] + 1):
                frame_to_caption[i] = cap

        for idx, (img, fd) in enumerate(
            tqdm(zip(raw_frames, frames_data), total=len(raw_frames), desc="  Rendering", unit="frame")
        ):
            annotated = renderer.render(
                img.copy(),
                yolo_results=fd["yolo"],
                hand_results=fd["hands"],
                action=action,
                caption=frame_to_caption.get(idx, ""),
                timestamp=idx / 5.0,
            )
            writer.write(annotated)
        writer.release()
        print(f"      → {output_path}")

        print("[7/7] Exporting JSON sidecar ...")
        exporter.export(json_path=json_path, action=action, segments=segments, frames_data=frames_data, fps=5)
        print(f"      → {json_path}")

    print("\n[RoboAnnotate] Done.\n")


if __name__ == "__main__":
    main()
