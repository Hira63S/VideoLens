"""
src/pipeline/frame_sampler.py

Optical flow based frame sampler.
Computes inter-frame motion magnitude and skips
frames that are too similar to the previous one.

Why this matters:
  nuScenes keyframes are at 2Hz — already sparse.
  But in a real video pipeline (or when using sweeps),
  consecutive frames are often near-identical.
  Running YOLO on every frame wastes GPU cycles.
  This sampler only passes frames with meaningful scene change.

Usage:
    sampler = FrameSampler(min_flow_threshold=2.0)
    samples = loader.samples  # from NuScenesLoader
    selected, skipped = sampler.select(samples)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

from src.dataloader.nuscenes_dataset import NuScenesSample

console = Console()


@dataclass
class SamplerStats:
    total_frames: int
    selected_frames: int
    skipped_frames: int
    flow_magnitudes: list[float]

    @property
    def skip_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.skipped_frames / self.total_frames

    @property
    def avg_flow(self) -> float:
        if not self.flow_magnitudes:
            return 0.0
        return float(np.mean(self.flow_magnitudes))

    def print(self):
        console.print("\n[bold cyan]Frame Sampler Stats[/bold cyan]")
        console.print(f"  Total frames:    {self.total_frames}")
        console.print(f"  Selected:        {self.selected_frames}")
        console.print(f"  Skipped:         {self.skipped_frames}")
        console.print(f"  Skip ratio:      {self.skip_ratio:.1%}")
        console.print(f"  Avg flow mag:    {self.avg_flow:.2f} px/frame")


class FrameSampler:
    """
    Selects keyframes from a sequence based on optical flow magnitude.

    Args:
        min_flow_threshold: minimum mean optical flow magnitude (pixels)
                            to consider a frame as a new keyframe.
                            Lower = keep more frames.
                            Higher = skip more frames.
                            Good starting range: 1.5 - 4.0
        resize_for_flow:    resize frames to this width before computing
                            flow — much faster, negligible accuracy loss.
        always_keep_first:  always keep the first frame in a sequence.
    """

    def __init__(
        self,
        min_flow_threshold: float = 2.0,
        resize_for_flow: int = 320,
        always_keep_first: bool = True,
    ):
        self.threshold = min_flow_threshold
        self.resize_w = resize_for_flow
        self.always_keep_first = always_keep_first

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select(
        self,
        samples: list[NuScenesSample],
    ) -> tuple[list[NuScenesSample], SamplerStats]:
        """
        Given a list of NuScenesSamples (ordered by timestamp),
        return the subset that represent genuine scene changes.

        Returns:
            selected: list of NuScenesSample to run inference on
            stats:    SamplerStats with skip ratio and flow info
        """
        if not samples:
            return [], SamplerStats(0, 0, 0, [])

        selected = []
        skipped = 0
        flow_magnitudes = []

        prev_gray = None

        for i, sample in enumerate(samples):
            # Load and resize frame for flow computation
            try:
                frame = sample.load_image()           # (H, W, 3) RGB
            except FileNotFoundError:
                console.print(f"[yellow]Skipping missing image: {sample.image_path.name}[/yellow]")
                skipped += 1
                continue

            gray = self._to_gray_resized(frame)

            # Always keep first frame
            if prev_gray is None:
                selected.append(sample)
                prev_gray = gray
                continue

            # Compute optical flow between prev and current
            flow_mag = self._compute_flow_magnitude(prev_gray, gray)
            flow_magnitudes.append(flow_mag)

            if flow_mag >= self.threshold:
                selected.append(sample)
              # update reference only on kept frames
            else:
                skipped += 1
            prev_gray = gray

        stats = SamplerStats(
            total_frames=len(samples),
            selected_frames=len(selected),
            skipped_frames=skipped,
            flow_magnitudes=flow_magnitudes,
        )

        return selected, stats

    def select_by_scene(
        self,
        samples: list[NuScenesSample],
    ) -> tuple[list[NuScenesSample], SamplerStats]:
        """
        Same as select() but resets optical flow state at each scene boundary.
        Use this when samples span multiple scenes.
        """
        from itertools import groupby

        all_selected = []
        all_magnitudes = []
        total_skipped = 0

        # Group by scene, preserving order within each scene
        keyed = sorted(samples, key=lambda s: (s.scene_name, s.timestamp))
        for scene_name, group in groupby(keyed, key=lambda s: s.scene_name):
            scene_samples = list(group)
            selected, stats = self.select(scene_samples)
            all_selected.extend(selected)
            all_magnitudes.extend(stats.flow_magnitudes)
            total_skipped += stats.skipped_frames
            console.print(
                f"  Scene {scene_name}: "
                f"{stats.selected_frames}/{stats.total_frames} frames kept "
                f"(avg flow: {stats.avg_flow:.2f})"
            )

        combined_stats = SamplerStats(
            total_frames=len(samples),
            selected_frames=len(all_selected),
            skipped_frames=total_skipped,
            flow_magnitudes=all_magnitudes,
        )
        return all_selected, combined_stats

    # ------------------------------------------------------------------
    # Optical flow helpers
    # ------------------------------------------------------------------

    def _to_gray_resized(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB frame to grayscale and resize for fast flow computation."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        new_w = self.resize_w
        new_h = int(h * new_w / w)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _compute_flow_magnitude(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> float:
        """
        Compute dense optical flow (Farneback) between two grayscale frames.
        Returns the mean magnitude of the flow vectors — a scalar representing
        how much the scene has changed in pixels/frame.

        Farneback params tuned for speed over accuracy (this is just a gate,
        not the final result — YOLO handles the real detection):
          pyr_scale=0.5  — image pyramid scale
          levels=2       — pyramid levels
          winsize=10     — averaging window size
          iterations=2   — algorithm iterations
          poly_n=5       — pixel neighborhood size
          poly_sigma=1.1 — Gaussian std for polynomial expansion
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=10,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        # flow shape: (H, W, 2) — x and y displacement per pixel
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(magnitude.mean())

    def tune_threshold(
        self,
        samples: list[NuScenesSample],
        thresholds: Optional[list[float]] = None,
    ):
        """
        Helper to find a good threshold for your data.
        Runs the sampler at multiple thresholds and prints the skip ratio.
        Use this once to calibrate, then hardcode the best value.

        Usage:
            sampler.tune_threshold(loader.samples)
        """
        if thresholds is None:
            thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

        console.print("\n[bold]Threshold Tuning[/bold]")
        console.print(f"{'Threshold':>12} {'Selected':>10} {'Skipped':>10} {'Skip %':>8} {'Avg Flow':>10}")
        console.print("-" * 55)

        for t in thresholds:
            temp_sampler = FrameSampler(min_flow_threshold=t)
            # self.threshold = t
            _, stats = temp_sampler.select(samples)
            console.print(
                f"{t:>12.1f} "
                f"{stats.selected_frames:>10} "
                f"{stats.skipped_frames:>10} "
                f"{stats.skip_ratio:>7.1%} "
                f"{stats.avg_flow:>10.2f}"
            )

        # Reset to default
        self.threshold = 2.0