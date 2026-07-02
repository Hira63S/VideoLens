import argparse
import yaml
import os



def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for video frame selection and object detection.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataroot", type=str, default="/mnt/i/nuScenes-panoptic-v1.0-all", help="Path to the dataset root directory.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Version of the dataset to use.")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to process.")
    parser.add_argument("--model_size", type=str, default="c", help="Size of the YOLO model to use (e.g., 'n', 's', 'm', 'l', 'x').")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output results.")
    parser.add_argument("--flow_threshold", type=float, default=0.5, help="Threshold for optical flow magnitude.")
    parser.add_argument("--chroma_path", type=str, default="./chroma_db")
    parser.add_argument("--db_name", type=str, default="video_frames", help="Name of the ChromaDB collection.")
    parser.add_argument("--clip_model", type=str, default="ViT-L-14", help="CLIP model to use for embeddings.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing frames.")
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--video", action="store_true", help="Export video clips for each result")
    parser.add_argument("--query", type=str, default="cars on a road", help="Search query")

    args, _ = parser.parse_known_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        cli_passed = {
            a.lstrip("-").replace("-", "_")
            for a in __import__("sys").argv
            if a.startswith("--")
        }
        for key, value in config.items():
            if key not in cli_passed and hasattr(args, key):
                setattr(args, key, value)
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Arguments parsed successfully:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")