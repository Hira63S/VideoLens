# %matplotlib inline
from nuscenes.nuscenes import NuScenes
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator
from rich.console import Console
 
console = Console()

NUSCENES_TO_YOLO = {
    "vehicle.car":                    "car",
    "vehicle.truck":                  "truck",
    "vehicle.bus.bendy":              "bus",
    "vehicle.bus.rigid":              "bus",
    "vehicle.motorcycle":             "motorcycle",
    "vehicle.bicycle":                "bicycle",
    "human.pedestrian.adult":         "pedestrian",
    "human.pedestrian.child":         "pedestrian",
    "human.pedestrian.wheelchair":    "pedestrian",
    "human.pedestrian.stroller":      "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer":"pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.trafficcone":     "traffic_cone",
    "movable_object.barrier":         "barrier",
    "vehicle.emergency.ambulance":    "car",
    "vehicle.emergency.police":       "car",
    "vehicle.construction":           "truck",
    "vehicle.trailer":                "truck",
}
 
# Final flat class list — these become your YOLO class indices
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CAM_WIDTH  = 1600.0
CAM_HEIGHT = 900.0

@dataclass
class BBox2D:
    """2D Bounding boxes in absolute and normalized formats."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_idx: int
    visibility: int = 4

    @property
    def yolo_format(self) -> list[float]:
        """[class_idx, cx, cy, w, h] normalized 0-1."""
        cx = (self.x1 + self.x2) / 2 / CAM_WIDTH
        cy = (self.y1 + self.y2) / 2 / CAM_HEIGHT
        w  = (self.x2 - self.x1) / CAM_WIDTH
        h  = (self.y2 - self.y1) / CAM_HEIGHT
        return [self.class_idx, cx, cy, w, h]
 
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


dataclass
class NuScenesSample:
    """
    One nuScenes keyframe from CAM_FRONT with its annotations.
    A 'sample' in nuScenes = one timestep across all sensors.
    We only use the front camera image here.
    """
    sample_token: str
    image_path: Path
    timestamp: int                        # microseconds
    scene_name: str
    bboxes: list[BBox2D] = field(default_factory=list)
 
    def load_image(self) -> np.ndarray:
        """Load image as (H, W, 3) RGB numpy array."""
        import cv2
        bgr = cv2.imread(str(self.image_path))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
    @property
    def yolo_annotations(self) -> list[list[float]]:
        """All boxes in YOLO format: [[class_idx, cx, cy, w, h], ...]"""
        return [b.yolo_format for b in self.bboxes]


    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.bboxes:
            counts[b.class_name] = counts.get(b.class_name, 0) + 1
        return counts

@dataclass
class NuScenesSample:

    sample_token: str
    image_path: Path
    timestamp: int
    scene_name: str
    bboxes: list[BBox2D] = field(default_factory=list)

    def load_image(self) -> np.ndarray:

        import cv2

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    @property
    def yolo_annotations(self) -> list[list[float]]:
        return [b.yolo_format for b in self.bboxes]

    @property
    def class_counts(self) -> dict[str,int]:
        counts: dict[str, int] = {}
        for b in self.bboxes:
            counts[b.class_name] = counts.get(b.class_name, 0) + 1
        return counts

    
class NuScenesLoader:
    
    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        max_samples: Optional[int] = None,
        min_visibility: int = 2,
    ):
        self.dataroot = Path(dataroot)
        self.version = version
        self.max_samples = max_samples
        self.min_visibility = min_visibility
 
        self.nusc = self._init_devkit()
        self.samples = self._load_samples()
 
        console.print(
            f"[green]NuScenesLoader ready.[/green] "
            f"{len(self.samples)} samples | "
            f"version={version} | "
            f"camera=CAM_FRONT"
        )

    def _init_devkit(self):
        """Initialize the nuScenes devkit."""
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError(
                "nuscenes-devkit not installed.\n"
                "Run: pip install nuscenes-devkit"
            )
 
        console.print(f"Loading nuScenes {self.version} from {self.dataroot} ...")
        nusc = NuScenes(
            version=self.version,
            dataroot=str(self.dataroot),
            verbose=False,
        )
        console.print(f"  Scenes: {len(nusc.scene)}")
        console.print(f"  Samples: {len(nusc.sample)}")
        return nusc

    def _load_samples(self) -> list[NuScenesSample]:
        """
        Build a flat list of nuscenessample from all scenes.
        """
        samples = []

        for scene in self.nusc.scene:
            scene_name = scene['name']
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                timestamp = sample['timestamp']

                # we only want the front cam for 2D boxes
                cam_token = sample['data'].get('CAM_FRONT')
                if not cam_token:
                    sample_token = sample['next']
                    continue

                # get the data 
                cam_data = self.nusc.get('sample_data', cam_token)
                image_path = self.dataroot / cam_data["filename"]


                bboxes = self._get_2d_boxes(cam_token)

                if bboxes:
                    samples.append(NuScenesSample(
                        sample_token=sample_token,
                        image_path=Path(image_path),
                        timestamp=timestamp,
                        scene_name=scene_name,
                        bboxes=bboxes,
                    ))

                if self.max_samples and len(samples) >= self.max_samples:
                    return samples

                sample_token = sample['next']
        return samples
    

    def _get_2d_boxes(self, cam_sample_data_token: str) -> list[BBox2D]:
        """
        Project 3D boxes to 2D and filter by visibility.
        """
        from pyquaternion import Quaternion
        from nuscenes.utils.geometry_utils import view_points, BoxVisibility
        from nuscenes.utils.data_classes import Box

        # get camera data
        cam_data = self.nusc.get('sample_data', cam_sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        sample = self.nusc.get('sample', cam_data['sample_token'])

        # camera intrinsics matrix
        intrinsic = np.array(cs_record['camera_intrinsic'])


        bboxes = []
        for ann_token in sample['anns']:
            ann = self.nusc.get("sample_annotation", ann_token)
            category = ann['category_name']
            class_name = NUSCENES_TO_YOLO.get(category)
            if class_name is None:
                continue
            
            vis = int(ann.get("visibility_token", "4"))
            if vis < self.min_visibility:
                continue
            
            box = Box(
                center=ann['translation'],
                size=ann['size'],
                orientation=Quaternion(ann['rotation']),
            )

            # global -> ego frame
            box.translate(-np.array(ego_record['translation']))
            box.rotate(Quaternion(ego_record['rotation']).inverse)

            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            if box.center[2] < 0.1:
                continue

            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, intrinsic, normalize=True)

            xs = corners_2d[0]
            ys = corners_2d[1]

            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())

    
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(CAM_WIDTH, x2); y2 = min(CAM_HEIGHT, y2)

            if x2 <= x1 or y2 <= y1:
                continue
            
            bboxes.append(BBox2D(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_name=class_name,
                class_idx=CLASS_TO_IDX[class_name],
                visibility=vis,
            ))
        return bboxes

    def __len__(self) -> int:
        return len(self.samples)

    
    def iter_samples(self) -> Iterator[NuScenesSample]:
        for sample in self.samples:
            yield sample
        
    def iter_scenes(self) -> Iterator[tuple[str, list[NuScenesSample]]]:
        """Iterate scene by scene, yielding (scene_name, [samples])."""
        from itertools import groupby
        keyed = sorted(self.samples, key=lambda s: s.scene_name)
        for scene_name, group in groupby(keyed, key=lambda s: s.scene_name):
            yield scene_name, list(group)
 
    def get_sample_by_token(self, token: str) -> Optional[NuScenesSample]:
        for s in self.samples:
            if s.sample_token == token:
                return s
        return None
 
    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
 
    def class_distribution(self) -> dict[str, int]:
        """Count total annotations per class across all samples."""
        dist: dict[str, int] = {c: 0 for c in CLASSES}
        for sample in self.samples:
            for cls, count in sample.class_counts.items():
                dist[cls] = dist.get(cls, 0) + count
        return dict(sorted(dist.items(), key=lambda x: -x[1]))
 
    def print_stats(self):
        """Print a summary table of the loaded dataset."""
        from rich.table import Table
 
        table = Table(title="nuScenes Dataset Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
 
        table.add_row("Total samples", str(len(self.samples)))
        table.add_row("Unique scenes", str(len({s.scene_name for s in self.samples})))
        table.add_row("Camera", "CAM_FRONT only")
        table.add_row("Image size", f"{int(CAM_WIDTH)}x{int(CAM_HEIGHT)}")
 
        console.print(table)
 
        dist = self.class_distribution()
        table2 = Table(title="Annotation Distribution")
        table2.add_column("Class", style="cyan")
        table2.add_column("Count", style="green")
        for cls, count in dist.items():
            table2.add_row(cls, str(count))
        console.print(table2)


if __name__ == "__main__":
    nu_scenes = NuScenesLoader(dataroot="/mnt/i/nuScenes-panoptic-v1.0-all", version="v1.0-mini", min_visibility=2)

# nu_scenes = NuScenesLoader(dataroot="./", version="v1.0-mini", min_visibility=2)
# nu_scenes.print_stats()
# print("length:",nu_scenes.__len__())