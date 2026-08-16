# agent.py

import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from query import search_videos, build_video_clip
from src.agents.anomaly import score_results
from src.agents.alerts import send_alert, log_alert
from src.args import parse_args


# Suspicious activity prompts — CLIP searches for these
SUSPICIOUS_PROMPTS = [
    "person loitering near entrance",
    "person concealing object",
    "person acting suspiciously near shelves",
    "unauthorized person in restricted area",
    "person running inside store",
]


def run_agent(
    prompts: list = None,
    n_results: int = 10,
    scan_interval: int = 30,
    anomaly_threshold: float = 0.5,
    save_dir: str = "agent_results",
    video: bool = True,
    continuous: bool = False,
):
    """
    VideoLens anomaly detection agent.
    
    Scans indexed frames using suspicious activity prompts,
    scores results for anomalies, and sends alerts when triggered.
    
    Args:
        prompts: list of suspicious activity queries for CLIP
        n_results: number of results to fetch per prompt
        scan_interval: seconds between scans (continuous mode)
        anomaly_threshold: minimum score to trigger alert
        save_dir: directory to save flagged frames and clips
        video: extract video clips for flagged frames
        continuous: run continuously (True) or single scan (False)
    """
    if prompts is None:
        prompts = SUSPICIOUS_PROMPTS

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[VideoLens Agent] Starting...")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Anomaly threshold: {anomaly_threshold}")
    print(f"  Mode: {'continuous' if continuous else 'single scan'}\n")

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning...")

        all_flagged = []
        clip_path = None
        for prompt in prompts:
            print(f"  Query: '{prompt}'")
            try:
                results = search_videos(prompt, n_results=n_results, video=False)
                scored = score_results(results, clip_threshold=1.4)
                scored = scored[:3]
                for anomaly in scored:
                    if anomaly.triggered and anomaly.anomaly_score >= anomaly_threshold:
                        print(f"  [!] ANOMALY DETECTED — score={anomaly.anomaly_score}")
                        print(f"      Scene: {anomaly.scene}, Frame: {anomaly.frame_idx}")
                        print(f"      Reasons: {anomaly.reasons}")

                        # extract video clip
                        if video:
                            clip_path = str(Path(save_dir) / f"alert_{anomaly.scene}_{anomaly.frame_idx}.mp4")
                            build_video_clip(
                                anomaly.image_path,
                                anomaly.scene,
                                clip_path,
                                context_frames=10,
                                fps=4,
                            )

                        # send alert
                        all_flagged.append(anomaly)

            except Exception as e:
                print(f"  [error] {prompt}: {e}")

        if all_flagged:
            best = max(all_flagged, key=lambda a:a.anomaly_score)
            send_alert(best, clip_path=clip_path)
            log_alert(best, log_path=str(Path(save_dir) / "agent_log.json"))
        print(f"\n  Scan complete — {len(all_flagged)} anomalies flagged")

        if not continuous:
            break

        print(f"  Next scan in {scan_interval}s...")
        time.sleep(scan_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval in seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    parser.add_argument("--no_video", action="store_true", help="Skip video clip extraction")
    parser.add_argument("--save_dir", type=str, default="agent_results")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_agent(
        scan_interval=args.interval,
        anomaly_threshold=args.threshold,
        save_dir=args.save_dir,
        video=not args.no_video,
        continuous=args.continuous,
    )