# annoboro

> Egocentric domestic action video annotation pipeline for robotics datasets.

annoboro takes a first-person (GoPro / phone) video of a domestic task and produces a fully annotated `.mp4` — with object bounding boxes, hand skeletons, action labels, natural-language captions, and timestamps — plus a structured JSON sidecar ready for downstream robotics training.

---

## Features

- **Object detection** — YOLOv8n draws colored bounding boxes with class label and confidence score on every frame
- **Hand skeleton** — MediaPipe Hands overlays a 21-point skeleton (orange joints, yellow connections) for up to 2 hands
- **Action segmentation** — Farneback optical flow automatically finds start/end boundaries of active segments
- **NL captioning** — BLIP generates a natural-language description for each segment keyframe
- **Rendered overlay** — action banner at top, caption at bottom, timestamp on every frame
- **JSON sidecar** — per-segment objects detected, hand-object contact pairs, captions, and timestamps
- **Apple Silicon native** — runs entirely on MPS (Metal Performance Shaders); no CUDA required

---

## Supported Actions

| Label | Description |
|---|---|
| `dishwashing` | Washing dishes at a sink |
| `cooking` | General stove / prep cooking |
| `chopping` | Knife work on a cutting board |
| `clothes_folding` | Folding laundry |
| `mopping` | Mopping a floor |
| `sweeping` | Sweeping a floor |

---

## Output

For each input video the pipeline writes two files:

**`<name>_annotated.mp4`**
- Colored YOLO bounding boxes with class + confidence
- 21-point hand skeleton per detected hand
- Action label banner across the top
- BLIP caption banner across the bottom
- Timestamp (seconds) in the action banner

**`<name>_annotated.json`**
```json
{
  "action": "cooking",
  "fps": 5,
  "segments": [
    {
      "start_time": 1.2,
      "end_time": 8.6,
      "keyframe_idx": 6,
      "caption": "a person is opening the refrigerator door",
      "objects_detected": [
        { "class": "refrigerator", "max_confidence": 0.91 },
        { "class": "person", "max_confidence": 0.88 }
      ],
      "hand_contact": [
        { "hand": "Right", "object": "refrigerator" }
      ]
    }
  ]
}
```

---

## Setup

### Requirements
- macOS with Apple Silicon (M1 / M2 / M3 / M4 / M5)
- Python 3.10+
- Homebrew

### 1. Install FFmpeg
```bash
brew install ffmpeg
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install torch torchvision
pip install "transformers>=4.35.0,<5.0.0"
pip install ultralytics mediapipe opencv-python Pillow tqdm
```

Or via requirements.txt:
```bash
pip install -r requirements.txt
```

> **Note:** `transformers` must be pinned to `<5.0.0`. Transformers 5.x changes the generation API in a way that breaks BLIP loading on the first run.

---

## Usage

```bash
# Activate venv (required every new terminal session)
source venv/bin/activate

# Run the pipeline
python pipeline.py --input path/to/video.mp4 --action chopping

# Custom output path
python pipeline.py --input video.mp4 --action cooking --output results/annotated.mp4
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Path to input `.mp4` video |
| `--action` | Yes | — | One of the 6 supported action labels |
| `--output` | No | `<input>_annotated.mp4` | Output video path |

---

## Pipeline Steps

```
Input video
    │
    ▼
[1] FFmpeg — extract frames at 5 fps
    │
    ▼
[2] YOLOv8n — detect objects per frame
    + MediaPipe Hands — 21-point hand skeleton per frame
    │
    ▼
[3] Optical Flow (Farneback) — segment active regions by motion
    │
    ▼
[4] BLIP — generate NL caption for each segment keyframe
    │
    ▼
[5] OpenCV renderer — compose all overlays onto frames
    │
    ▼
[6] cv2.VideoWriter — write annotated .mp4 (mp4v codec, 5 fps)
    + JSON sidecar export
```

---

## Project Structure

```
roboannotate/
├── pipeline.py      # CLI entrypoint — orchestrates all steps
├── annotator.py     # YOLOv8n + MediaPipe Hands Tasks API
├── segmenter.py     # Farneback optical flow segmentation
├── captioner.py     # BLIP caption generation (MPS-aware)
├── renderer.py      # OpenCV overlay renderer
├── exporter.py      # JSON sidecar writer
└── requirements.txt
```

---

## Tech Stack

| Component | Library |
|---|---|
| Object detection | [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) |
| Hand skeleton | [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Tasks API) |
| Optical flow | OpenCV `calcOpticalFlowFarneback` |
| Image captioning | [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) via HuggingFace Transformers |
| Frame I/O | FFmpeg + OpenCV |
| Accelerator | PyTorch MPS (Apple Silicon) |

---

## Notes

- On first run, YOLOv8n (~6 MB) and `hand_landmarker.task` (~1 MB) are downloaded automatically; BLIP weights (~990 MB) are downloaded from HuggingFace and cached in `~/.cache/huggingface/`.
- All subsequent runs use the local cache — no re-download.
- The `NORM_RECT without IMAGE_DIMENSIONS` warning from MediaPipe is cosmetic and does not affect results.
- CUDA is never referenced; the device selection is: `"mps" if torch.backends.mps.is_available() else "cpu"`.
