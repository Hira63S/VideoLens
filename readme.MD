# VideoLens — Video Understanding & Semantic Search Pipeline

A multi-model video understanding pipeline that detects, tracks, and semantically
indexes objects in autonomous driving footage. Built to explore the kind of
high-throughput, multi-model video processing infrastructure used in production
video AI systems (object detection, tracking, and semantic search working together
on the same data).

Given a sequence of driving footage frames, the pipeline:
1. Filters redundant frames using optical flow
2. Detects and classifies objects with YOLOv9
3. Tracks objects persistently across frames with ByteTrack
4. Encodes each frame's visual content with CLIP
5. Indexes frames in a vector database for natural language search

You can then query the dataset in plain English — e.g. *"a person on a sidewalk
near a road with cars"* — and get back the most semantically relevant frames,
along with the actual image and YOLO's structured detections for that frame.

---

## Why this project

I built this to get hands-on with the kind of engineering problems involved in
processing video at scale — running multiple models (detection, tracking,
embedding) over the same data efficiently on consumer GPU hardware (RTX
2070/2080, 8GB VRAM), and combining structured detection output with semantic
search into one coherent system. The goal was to go beyond running a single
model in a notebook and instead build something closer to a real pipeline:
modular, benchmarked, and testable.

---

## Architecture

```
Raw Video Frames (nuScenes CAM_FRONT)
        │
        ▼
┌──────────────────────────┐
│   Frame Sampler            │  Optical flow (Farneback) — skips frames
│   (Optical Flow)            │  below a motion threshold to reduce
└────────────┬───────────────┘  redundant GPU work
             │
             ▼
┌────────────────────────────────────────┐
│         GPU Inference Stage             │
│  YOLOv9-c (detection)                   │
│  CLIP ViT-L/14 (semantic embedding)     │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│   ByteTrack                 │  Assigns persistent IDs to objects
│   (Object Tracking)          │  across frames
└────────────┬───────────────┘
             │
             ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│   output.json                │      │   ChromaDB                    │
│   (detections + tracks)      │      │   (CLIP embeddings + paths)    │
└──────────────────────────┘      └────────────┬───────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────┐
                                   │  Natural Language Query    │
                                   │  "pedestrian near truck"   │
                                   │  → ranked relevant frames  │
                                   └──────────────────────────┘
```

The pipeline produces two complementary outputs from the same set of frames:
**structured detections** (YOLO + ByteTrack → `output.json`) and **semantic
search** (CLIP → ChromaDB). They run independently on the same data rather
than depending on each other.

---

## Dataset

**[nuScenes](https://www.nuscenes.org/)** (v1.0-mini split), CAM_FRONT only.

Real autonomous-vehicle sensor data, used as a standard benchmark across the
AV/robotics industry, small enough to run end-to-end locally.

A key engineering challenge: nuScenes stores **3D bounding box annotations**
in global coordinates, not 2D image-plane boxes. Getting usable 2D boxes for
YOLO-style evaluation required implementing the full projection pipeline:

```
3D box (global frame)
  → ego vehicle frame      (via ego_pose)
  → camera frame           (via calibrated_sensor)
  → image plane (pixels)   (via camera intrinsics)
```

---

## Models — chosen for an 8GB VRAM GPU (RTX 2070/2080)

| Task | Model | VRAM | Why |
|---|---|---|---|
| Detection | YOLOv9-c | ~2GB | Strong accuracy/speed tradeoff for this VRAM budget — lighter than `m`/`e` variants while retaining good mAP |
| Tracking | ByteTrack | negligible | Industry-standard tracker for persistent object IDs across frames |
| Semantic search | CLIP ViT-L/14 | ~1.5GB | Larger and more discriminative than ViT-B/32, while still comfortably fitting alongside YOLO in memory |
| Vector storage | ChromaDB | — | Lightweight, persistent, no external service required |

All models run sequentially on a single GPU within budget — no model
swapping or offloading required at this scale.

---

## Benchmark Results

*(Measured on 20 frames from nuScenes scene-0061 — see "Status" below for
scaling to the full mini split)*

| Metric | Value |
|---|---|
| Inference throughput (modular pipeline) | ~8 fps |
| Frame sampler skip ratio (optical flow, threshold=6.0) | 22.5% |
| Average inter-frame optical flow magnitude | 7.19 px/frame |
| CLIP embedding dimension | 768 |
| Collection size after indexing | 20 frames, 20 unique embeddings |

**Note on frame sampling:** nuScenes keyframes are already sparse (2Hz,
pre-selected). The optical flow sampler shows modest filtering (22.5%) on
this data because consecutive keyframes already represent meaningful scene
changes. On raw continuous video (e.g. 30fps dashcam footage), the same
sampler would be expected to filter a much higher percentage of near-duplicate
frames before they reach the GPU.

---

## Example Search

**Query:** `"Find a person on a sidewalk near a road with cars"`

**Top result:** `scene-0061_7` — independently confirmed via YOLO's detections
for that frame to contain car, truck, and pedestrian objects. CLIP correctly
identified this as the most semantically relevant frame without any explicit
labeling — the two halves of the pipeline (semantic search and structured
detection) agreed.

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run the full pipeline (detection + tracking + embedding + indexing)
python pipeline.py --dataroot /path/to/nuScenes-v1.0-mini --max_samples 20

# query the indexed frames
python query.py

# run tests
python test_pipeline.py
```

---

## Status / Known Limitations

- **Scaling beyond 20 samples surfaced a metadata bug** (image paths not
  updating correctly across scene boundaries when processing the full mini
  split) — currently being fixed. This is a good example of why testing at
  small scale isn't sufficient; the bug only appeared once data spanned
  multiple scenes.
- **Search validation is currently manual** — I verify CLIP's top results
  against YOLO's structured detections by inspection. The natural next step
  (in progress) is a small hand-labeled evaluation set (~20-30 queries with
  expected object classes) to compute precision@5 automatically, matching
  how retrieval systems are evaluated in production — live search has no
  ground truth, so evaluation is a separate, deliberately curated offline
  process.

---

## What I'd improve with more time

- Run YOLO and CLIP concurrently using CUDA streams instead of sequentially
- Automate precision@5 evaluation against a labeled query set
- Fine-tune YOLOv9 on a rare/sparse class (e.g. cyclists) and measure mAP delta
- Replace optical flow gating with a lightweight learned frame-redundancy classifier
- Containerize with Docker for reproducible deployment

---

## Stack

`PyTorch` · `Ultralytics YOLOv9` · `OpenCLIP` · `ByteTrack (trackers/supervision)` ·
`ChromaDB` · `nuscenes-devkit` · `OpenCV`