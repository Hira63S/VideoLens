# VideoLens — Semantic Video Search Pipeline

This is a prototype of a video retrieval software built for small businesses. 
The issue of having to go through hours of security footage to find 
a certain event is tedious and time consuming. 
The solution:
A multi-model video understanding pipeline that lets you search through hours of
camera footage using plain English — built for the kind of problem that happens in
real businesses: *"Show me everything from Wednesday night near the entrance"* or
*"Find any frames with a person near a vehicle after dark."*
You describe what you're looking for, and the
system retrieves the most relevant frames ranked by visual similarity, then extracts
a video clip centered on each match.

This specific repo replaces the real-world application built on security footage with
the **nuScenes** (autonomous vehicle sensor data) as the development dataset
because it provides real multi-camera footage with rich metadata — the same pipeline
applies directly to any multi-camera environment.

---

## What It Does

**Search mode** — query footage in plain English:
```
"person near entrance at night"
  → ranked frames, deduplicated by scene
  → matched frame images saved
  → video clips extracted centered on each match
```

**Agent mode** — autonomous monitoring:
```
Agent scans indexed frames every N seconds
  → CLIP matches frames to suspicious activity prompts
  → YOLO confirms detections, ByteTrack checks persistence
  → anomaly scored across 3 rules
  → if triggered: email alert sent with frame + video clip attached
  → all anomalies logged to agent_log.json
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
│   (Optical Flow)              │  below motion threshold; 
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
│   YOLO detections │    │   CLIP embeddings + metadata    │
│   + track IDs     │    └────────────┬───────────────────┘
└──────────────────┘                  │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Query / Agent                        │
                    │                                       │
                    │  Search: text → CLIP → ranked frames  │
                    │  Agent:  prompts → score → alert      │
                    └─────────────────────────────────────┘
```

Two complementary outputs from the same frames: **structured detections**
(YOLO + ByteTrack → `output.json`) and **semantic search** (CLIP → ChromaDB).
They run independently and cross-validate — when CLIP's top result matches what
YOLO detected in that frame, confidence is high.

---

**Two complementary outputs cross-validate each other:**
CLIP retrieves semantically relevant frames. YOLO confirms what's actually in them.
Unit tests verify this agreement — 17 passing tests including cross-validation.

---

**Incident investigation:**
```bash
python query.py --config configs/example.yaml \
  --query "person near entrance at night" --n_results 5 --video
```

**Autonomous monitoring:**
```bash
python agent.py --config configs/example.yaml --continuous --interval 30
```

Output:
```
agent_results/
  alert_scene-1094_112.mp4    ← video clip of flagged moment
  agent_log.json              ← full audit trail
```
Owner receives email with matched frame + video clip attached.

---
## Anomaly Detection

The agent scores each frame across three rules:

| Rule | Signal | Score |
|---|---|---|
| CLIP semantic match | Distance below threshold | +0.4 |
| Person detected | YOLO detects person/pedestrian | +0.3 |
| Persistent tracking | ByteTrack confirms N+ tracks | +0.3 |

Anomaly prompts are plain English — no retraining needed:
```python
SUSPICIOUS_PROMPTS = [
    "person loitering near entrance",
    "person concealing object",
    "person acting suspiciously near shelves",
    "person running inside store",
]
```

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

## Email Alerts

```bash
export ALERT_SENDER_EMAIL=your@gmail.com
export ALERT_SENDER_PASSWORD=your_app_password  # Gmail App Password
export ALERT_RECIPIENT_EMAIL=owner@gmail.com
python agent.py --config configs/myconfig.yaml --continuous
```

---

## API

```
POST /search
  { "query": "string", "n_results": 5, "video": false }
  → ranked results with YOLO detections included

GET /health
  → { "status": "healthy" }
```

Swagger UI at `http://localhost:8000/docs`

---

## Project Structure

```
VideoLens/
  pipeline.py          ← indexing: CLIP + YOLO + ByteTrack
  query.py             ← semantic search + video extraction
  agent.py             ← autonomous monitoring loop
  postprocess.py       ← clip extraction + incident reports
  src/
    agents/
      anomaly.py       ← anomaly scoring logic
      alerts.py        ← email alerts + logging
    api/api.py         ← FastAPI service
    encoder/           ← CLIP
    models/            ← YOLO
    vector_db/         ← ChromaDB
    dataloader/        ← nuScenes
    pipeline/          ← frame sampler
  tests/
    test_query.py
    test_anomaly.py
    test_cross_validation.py
  configs/
    example.yaml
  Dockerfile
  docker-compose.yml
```

---

## What I'd Improve With More Time

- Fine-tune an action recognition model on shoplifting datasets for higher precision
- Add timestamp-aware search (filter by time range e.g. "only after 10pm")
- Multi-camera support (nuScenes has 6 cameras, currently using CAM_FRONT only)
- Precision@5 automated eval against a hand-labeled query set
- Deploy to AWS EC2 with the Pi as a camera relay (edge capture + cloud inference)
- Build a simple web UI for non-technical operators

---

## Stack

`PyTorch` · `Ultralytics YOLOv9` · `OpenCLIP` · `ByteTrack` · `ChromaDB` ·
`nuscenes-devkit` · `OpenCV` · `FastAPI` · `Docker` · `pytest`
