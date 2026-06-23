"""
src/models/detector.py

YOLOv9 model wrapper with clean inference interface.
Handles model loading, warmup, and batched inference.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

from ultralytics import YOLO

console = Console()

# Maps YOLO's COCO indices to BDD100K-relevant classes
# YOLOv9 is pretrained on COCO — these are the overlapping classes
COCO_TO_BDD = {
    0: "pedestrian",    # person
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}


@dataclass
class Detection:
    """Single object detection result."""
    frame_idx: int
    timestamp_sec: float
    class_name: str
    class_idx: int
    confidence: float
    bbox_xyxy: list[float]      # [x1, y1, x2, y2] absolute pixels
    bbox_xywhn: list[float]     # [cx, cy, w, h] normalized


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    clip_id: str
    frame_idx: int
    timestamp_sec: float
    detections: list[Detection]
    image_width: int
    image_height: int

    @property
    def class_counts(self) -> dict[str, int]:
        counts = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class YOLODetector:
    """
    YOLOv9 wrapper for video frame inference.

    Usage:
        detector = YOLODetector(model_size="c", device="cuda")
        results = detector.infer_batch(frames, frame_indices, timestamps, clip_id)
    """

    # Available YOLOv9 sizes: n (nano), s (small), m (medium), c (compact), e (extended)
    # For RTX 2070/2080 8GB: 'c' is the sweet spot — good accuracy, fits in memory
    MODEL_VARIANTS = {
        "n": "yolov9n.pt",
        "s": "yolov9s.pt",
        "m": "yolov9m.pt",
        "c": "yolov9c.pt",
        "e": "yolov9e.pt",
    }

    def __init__(
        self,
        model_size: str = "c",
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        model_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = confidence_threshold
        self.iou = iou_threshold

        model_name = model_path or self.MODEL_VARIANTS.get(model_size, "yolov9c.pt")
        console.print(f"Loading YOLOv9 ({model_name}) on {self.device}...")

        self.model = YOLO(model_name)
        self.model.to(self.device)

        self._warmup()
        console.print(f"[green]YOLOv9 ready.[/green] Conf={self.conf}, IoU={self.iou}")

    def _warmup(self):
        """Run a dummy forward pass to initialize CUDA kernels."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, conf=self.conf)
        if self.device == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_batch(
        self,
        frames: np.ndarray,            # (B, H, W, 3) RGB uint8
        frame_indices: list[int],
        timestamps: list[float],
        clip_id: str,
    ) -> list[FrameDetections]:
        """
        Run YOLOv9 on a batch of frames.
        Returns one FrameDetections per frame.
        """
        if frames is None or len(frames) == 0:
            return []

        B, H, W, _ = frames.shape

        # ultralytics expects a list of numpy arrays or a single batched array
        frame_list = [frames[i] for i in range(B)]

        results = self.model.predict(
            frame_list,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

        frame_detections = []
        for i, result in enumerate(results):
            dets = self._parse_result(
                result,
                frame_idx=frame_indices[i],
                timestamp_sec=timestamps[i],
                img_w=W,
                img_h=H,
            )
            frame_detections.append(
                FrameDetections(
                    clip_id=clip_id,
                    frame_idx=frame_indices[i],
                    timestamp_sec=timestamps[i],
                    detections=dets,
                    image_width=W,
                    image_height=H,
                )
            )

        return frame_detections

    def _parse_result(
        self,
        result,
        frame_idx: int,
        timestamp_sec: float,
        img_w: int,
        img_h: int,
    ) -> list[Detection]:
        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes
        for j in range(len(boxes)):
            coco_cls = int(boxes.cls[j].item())
            conf = float(boxes.conf[j].item())

            # Map COCO class → BDD100K name, skip irrelevant classes
            class_name = COCO_TO_BDD.get(coco_cls)
            if class_name is None:
                continue

            # Absolute pixel bbox
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()

            # Normalized center-format bbox
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            detections.append(Detection(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                class_name=class_name,
                class_idx=coco_cls,
                confidence=conf,
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_xywhn=[cx, cy, bw, bh],
            ))

        return detections

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def vram_usage_mb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e6
        return 0.0