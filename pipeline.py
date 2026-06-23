import numpy as np
import time
from PIL import Image
import supervision as sv
import json
import torch


from src.dataloader.nuscenes_dataset import NuScenesLoader
from src.pipeline.frame_sampler import FrameSampler
from src.models.detector import YOLODetector
from src.encoder.clip_encoder import get_clip_model, encode_images
from src.vector_db.chroma import initialize_chroma_db, add_embeddings
from src.args import parse_args
from trackers import ByteTrackTracker

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()

def pipeline_test(args):
    print(f"dataroot: {args.dataroot}")
    print(f"version: {args.version}")
def pipeline_test(args):

    nuscenes = NuScenesLoader(dataroot=args.dataroot, version=args.version, max_samples=args.max_samples)
    samples = nuscenes.samples
    sampler = FrameSampler(min_flow_threshold=args.flow_threshold)
    selected, stats = sampler.select(samples)
    print(f"Selected {len(selected)} frames out of {len(samples)} total frames.")

    # start DB
    collection = initialize_chroma_db(args.chroma_path, args.db_name)
    # get clip
    model, tokenizer, preprocess = get_clip_model(args)
    # get yolo
    detector = YOLODetector(model_size=args.model_size)
    #initialize tracker
    tracker = ByteTrackTracker()

    # metrics
    count = 0
    batch = []
    batch_indices = []
    batch_timestamps = []
    all_frames = []
    batch_image_paths = []
    start_time = time.time()
    
    for i, sample in enumerate(selected):
        image = sample.load_image()
        batch.append(image)
        batch_indices.append(i)
        batch_timestamps.append(sample.timestamp)
        batch_image_paths.append(str(sample.image_path))
        count += 1
        if len(batch) == args.batch_size or i == len(selected) - 1:
            # get the YOLO results
            results = detector.infer_batch(np.stack(batch),
                        batch_indices,
                        batch_timestamps,
                        sample.scene_name)
            # embed the batch with CLIP and add to chromadb
            ids = [f"{sample.scene_name}_{idx}" for idx in batch_indices]
            metadatas = [
            {
                "scene": sample.scene_name,
                "timestamp": ts,
                "image_path": path,
                "frame_idx": idx
            } for ts, idx, path in zip(batch_timestamps, batch_indices, batch_image_paths)]
            print("metadatas: ", metadatas[:10])
            with torch.no_grad():
                clip_embeddings = encode_images(model, preprocess, batch)
                add_embeddings(collection, ids, clip_embeddings, metadatas)

                # Here you would typically store or process the detections as needed
            # also call the tracker model:

            # parse through the YOLO findings
            for frame_det in results:
                print(f"Frame {frame_det.frame_idx} | "
                f"detections: {len(frame_det.detections)} | "
                f"classes: {frame_det.class_counts}")

                sv_dets = sv.Detections(
                xyxy=np.array([d.bbox_xyxy for d in frame_det.detections]),
                confidence=np.array([d.confidence for d in frame_det.detections]),
                class_id=np.array([d.class_idx for d in frame_det.detections]),
                )
                tracked = tracker.update(sv_dets)
                total = len(tracked.tracker_id)
                confirmed = (tracked.tracker_id != -1).sum()


                all_frames.append({
                    "frame_idx": frame_det.frame_idx,
                    "timestamp": frame_det.timestamp_sec,
                    "detections": [
                        {"class": d.class_name, "confidence": d.confidence,
                        "track_id": int(tid), "bbox": d.bbox_xyxy}
                        for d, tid in zip(frame_det.detections, tracked.tracker_id)
                    ]
                })
            count = 0
            batch = []
            batch_indices = []
            batch_timestamps = []
            batch_image_paths = []
    end_time = time.time()
    print(f"Pipeline test completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    pipeline_test(args)
