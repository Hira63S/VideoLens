# VideoLens — Semantic Video Search Pipeline

A multi-model video understanding pipeline that lets you search through hours of
camera footage using plain English — built for the kind of problem that happens in
real businesses: *"Show me everything from Wednesday night near the entrance"* or
*"Find any frames with a person near a vehicle after dark."*

The immediate motivation: my family owns gas stations and convenience stores. When
an incident happens — a robbery, a disputed transaction, a slip-and-fall — reviewing
footage means scrubbing through hours of video manually, camera by camera. VideoLens
turns that into a search problem. You describe what you're looking for, and the
system retrieves the most relevant frames ranked by visual similarity, then extracts
a video clip centered on each match.

Built on **nuScenes** (autonomous vehicle sensor data) as the development dataset
because it provides real multi-camera footage with rich metadata — the same pipeline
applies directly to any multi-camera environment.

---

## What It Does

Given a set of camera frames, the pipeline:

1. Filters redundant frames using optical flow (skips near-duplicate frames before they hit the GPU)
2. Detects and classifies objects with YOLOv9 (people, vehicles, bags, etc.)
3. Tracks objects persistently across frames with ByteTrack
4. Encodes each frame's visual content with CLIP
5. Indexes frames in ChromaDB for natural language search

Then, when something happens:

```
Query: "person near entrance at night"
  → returns ranked frames with similarity scores, deduplicated by scene
  → saves matched frames to query_results/
  → optionally extracts video clips centered on each match (--video flag)
```

Also exposed as a **FastAPI service** — query via HTTP with optional video output.

---

## Architecture

```
Raw Video Frames (nuScenes CAM_FRONT, 10 scenes, ~380 frames)
        │
        ▼
┌──────────────────────────────┐
│   Frame Sampler               │  Optical flow (Farneback) — skips frames
│   (Optical Flow)              │  below motion threshold; reduces redundant
└────────────┬──────────────────┘  GPU work on near-duplicate frames
             │
             ▼
┌──────────────────────────────────────────┐
│         GPU Inference Stage               │
│  YOLOv9-c  (detection + classification)   │
│  CLIP ViT-L/14  (semantic embedding)      │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────┐
│   ByteTrack                   │  Assigns persistent IDs to objects
│   (Object Tracking)           │  across frames
└────────────┬──────────────────┘
             │
             ▼
┌──────────────────┐    ┌────────────────────────────────┐
│   output.json     │    │   ChromaDB                      │
│   detections +    │    │   CLIP embeddings + frame paths  │
│   track IDs       │    └────────────┬───────────────────┘
└──────────────────┘                  │
                                      ▼
                       ┌──────────────────────────────────┐
                       │  Natural Language Query            │
                       │  "person near vehicle"             │
                       │  → ranked frames, deduped by scene │
                       └────────────┬─────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────────────┐
                       │  Output                            │
                       │  • Saved frames (query_results/)   │
                       │  • Video clips per matched scene   │
                       │  • results.json with scores        │
                       └──────────────────────────────────┘
```

Two complementary outputs from the same frames: **structured detections**
(YOLO + ByteTrack → `output.json`) and **semantic search** (CLIP → ChromaDB).
They run independently and cross-validate — when CLIP's top result matches what
YOLO detected in that frame, confidence is high.

---

## Real-World Use Case

A convenience store robbery happens Tuesday night. Instead of pulling up each
camera and scrubbing manually:

```bash
python query.py --config configs/example.yaml --query "person near entrance at night" --n_results 5 --video
```

Output:
```
query_results/
  n015-...__CAM_FRONT__....jpg    ← matched frame, scene-0061
  n015-...__CAM_FRONT__....jpg    ← matched frame, scene-0553
  video0.mp4                       ← 20-frame clip centered on match
  video1.mp4
  results.json                     ← ranked matches with scores + timestamps
```

Each clip is scene-isolated — frames are pulled only from the matched scene's
recording session, not stitched across unrelated footage.

---

## Dataset

**[nuScenes](https://www.nuscenes.org/)** (v1.0-mini), CAM_FRONT.

10 scenes, ~380 indexed frames across diverse driving conditions. Used as a
standard benchmark in AV/robotics, and a natural proxy for a multi-camera
business environment.

A notable engineering challenge: nuScenes stores **3D bounding box annotations**
in global coordinates. Getting usable 2D boxes required implementing the full
projection pipeline:

```
3D box (global frame)
  → ego vehicle frame      (via ego_pose)
  → camera frame           (via calibrated_sensor)
  → image plane (pixels)   (via camera intrinsics)
```

---

## Models — chosen for 8GB VRAM (RTX 2070/2080)

| Task | Model | VRAM | Why |
|---|---|---|---|
| Detection | YOLOv9-c | ~2GB | Strong accuracy/speed tradeoff at this VRAM budget |
| Tracking | ByteTrack | negligible | Industry-standard persistent object IDs across frames |
| Semantic search | CLIP ViT-L/14 | ~1.5GB | More discriminative than ViT-B/32; fits alongside YOLO |
| Vector storage | ChromaDB | — | Lightweight, persistent, no external service required |

All models run sequentially on a single GPU within budget — no model swapping or
offloading required at this scale.

---

## Benchmark Results

*(Measured on 380 frames across 10 nuScenes scenes)*

| Metric | Value |
|---|---|
| Inference throughput | ~8 fps |
| Frame sampler skip ratio (optical flow) | 22.5% |
| Average inter-frame optical flow magnitude | 7.19 px/frame |
| CLIP embedding dimension | 768 |
| Indexed frames | 380 across 10 scenes |

**Note on frame sampling:** nuScenes keyframes are pre-selected at 2Hz, so the
22.5% filter rate is conservative. On raw 30fps security footage, the same sampler
eliminates far more near-duplicate frames before they reach the GPU.

---

## Example Search

**Query:** `"person near a vehicle"`

Returns top 5 results deduplicated by scene — each result comes from a different
recording session. With `--video`, a clip is extracted for each match using only
frames from that scene's recording session (no cross-scene bleeding).

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# copy and edit the example config
cp configs/example.yaml configs/myconfig.yaml
# set dataroot, chroma_path, etc.

# run the full pipeline (detection + tracking + embedding + indexing)
python pipeline.py --config configs/myconfig.yaml

# query the indexed frames
python query.py --config configs/myconfig.yaml --query "cars on a road" --n_results 5

# query with video clip extraction
python query.py --config configs/myconfig.yaml --query "person near entrance" --n_results 5 --video

# run as an API server
CONFIG_PATH=configs/myconfig.yaml uvicorn src.api.api:app --reload
# then POST to http://localhost:8000/search
# { "query": "cars on a road", "n_results": 5, "video": true }
```

---

## API

```
POST /search
  { "query": "string", "n_results": 5, "video": false }
  → { "query": "...", "results": [{ "id", "score", "scene", "frame_idx", "image_path", "timestamp" }] }

GET /health
  → { "status": "healthy" }
```

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Config

All pipeline parameters are controlled via YAML config files in `configs/`.
See `configs/example.yaml` for a full reference. CLI flags override YAML values.

```bash
# override a single value at runtime
python query.py --config configs/myconfig.yaml --n_results 10
```

---

## What I'd Improve With More Time

- Run YOLO and CLIP concurrently using CUDA streams instead of sequentially
- Automate precision@5 evaluation against a hand-labeled query set
- Fine-tune YOLOv9 on sparse classes (e.g. bags, cyclists) and measure mAP delta
- Replace optical flow gating with a lightweight learned frame-redundancy classifier
- Add timestamp-aware search: filter ChromaDB results by time range before ranking
- Containerize with Docker for reproducible deployment

---

## Stack

`PyTorch` · `Ultralytics YOLOv9` · `OpenCLIP` · `ByteTrack` · `ChromaDB` ·
`nuscenes-devkit` · `OpenCV` · `FastAPI` · `FFmpeg`