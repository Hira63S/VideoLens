from src.encoder.clip_encoder import get_clip_model, encode_text
from src.vector_db.chroma import initialize_chroma_db, query_embeddings, load_image_from_result
from src.dataloader.nuscenes_dataset import NuScenesLoader
from src.args import parse_args
import os
import cv2
from pathlib import Path 
from PIL import Image
import yaml


_model = None
_tokenizer = None
_collection = None

def _load():
    global _model, _tokenizer, _collection
    if _model is None:
        args = parse_args()
        config_path = os.environ.get("CONFIG_PATH")
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                
        _model, _tokenizer, _ = get_clip_model(args)
        _collection = initialize_chroma_db(args.chroma_path, args.db_name)

def search_videos(query, n_results, video=False):
    _load()
    print(f"Searching for: {query}")
    print(f"n_results: {n_results}")
    query_embedding = encode_text(_model, _tokenizer, query)

    print("Raw Chroma results:")
    raw_results = query_embeddings(_collection, query_embedding, n_results=n_results * 3)
    print("results: ", raw_results)
    
    scenes_detected = set()
    dup_ids, dupe_metas, dupe_distance = [], [], []

    for ids, meta, distance in zip(
        raw_results["ids"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0],
    ):
        if meta["scene"] not in scenes_detected:
            dup_ids.append(ids)
            dupe_metas.append(meta)
            dupe_distance.append(distance)
            scenes_detected.add(meta["scene"])
        if len(dup_ids) == n_results:
            break

    # save the found videos:
    save_dir = "query_results"
    os.makedirs(save_dir, exist_ok=True)
    # image_path:
    for k, meta in enumerate(dupe_metas):
        image = Image.open(meta['image_path'])
        image.save(os.path.join(save_dir, Path(meta["image_path"]).name))

        if video:
            # in search_videos, before calling build_video_clip, get all frame paths for this scene

            # also get ALL frames for this scene from the full DB
            all_scene_results = _collection.get(
                where={"scene": meta["scene"]},
                include=["metadatas"]
            )
            scene_frame_paths = sorted([m["image_path"] for m in all_scene_results["metadatas"]])
            print(f"  image_path: {meta['image_path']}")
            print(f"  exists: {os.path.exists(meta['image_path'])}")
            print(f"  building clip: video{k}.mp4")
            result = build_video_clip(meta["image_path"], meta["scene"], os.path.join(save_dir, f"video{k}.mp4"),
                                      scene_frame_paths=scene_frame_paths,)
            print(f"  clip result: {result}")
    return [
        {
            "id": id_,
            "score": round(1 - distance, 4),  # chroma returns distances, convert to similarity
            "scene": meta["scene"],
            "frame_idx": meta["frame_idx"],
            "image_path": meta["image_path"],
            "timestamp": meta.get("timestamp"),
        }
        for id_, meta, distance in zip(
            dup_ids, dupe_metas, dupe_distance
        )
    ]

def build_video_clip(
    matched_image_path: str,
    scene_name: str,
    output_path: str,
    scene_frame_paths: list=None,
    context_frames: int=10,
    fps: int=4
):

    frame_path = Path(matched_image_path)

    if scene_frame_paths:
        all_frames = [Path(p) for p in scene_frame_paths]
    else:
        scene_dir = frame_path.parent
        session_prefix = frame_path.name.split("__CAM_FRONT__")[0]
        all_frames = sorted(scene_dir.glob("*.jpg"))
        all_frames = [f for f in all_frames if f.name.startswith(session_prefix)]

    if not all_frames:
        print(f"Video frames not found")
        return None

    try:
        idx = all_frames.index(frame_path)
    except ValueError:
        return None
    
    start = max(0, idx - context_frames)
    end = min(len(all_frames), idx + context_frames + 1)
    clip_frames = all_frames[start:end]

    sample = cv2.imread(str(clip_frames[0]))
    if sample is None:
        return None 
    
    h, w = sample.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fp in clip_frames:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)

    writer.release()
    print(f"  [video] Saved clip ({len(clip_frames)} frames): {output_path}")
    return output_path


if __name__ == "__main__":
    import json
    args = parse_args()

    results = search_videos(args.query, args.n_results, video=args.video)
    print(f"video={args.video}, query={args.query}")
    with open("query_results/results.json", "w") as f:
        json.dump({"query": args.query, "results": results}, f, indent=2)
    print(f"\nSaved {len(results)} results to query_results/")
