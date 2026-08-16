# src/agent/anomaly.py

from dataclasses import dataclass

@dataclass
class AnomalyScore:
    frame_idx: int
    image_path: str
    scene: str
    timestamp: int
    clip_score: float
    detected_classes: list
    track_ids: list
    anomaly_score: float
    triggered: bool
    reasons: list

def score_frame(result: dict, 
                clip_threshold: float = 1.4,
                min_track_frames: int = 3,
                suspicious_classes: list = None) -> AnomalyScore:
    """
    Score a single query result for anomaly likelihood.
    
    Rules:
    - CLIP distance below threshold (strong semantic match to suspicious prompt)
    - Person detected by YOLO
    - Confirmed tracks (track_id != -1) above minimum
    """
    if suspicious_classes is None:
        suspicious_classes = ["red car", "truck"]

    detections = result.get("detections", [])
    distance = abs(result.get("score", 1.0))  # lower distance = stronger match
    
    detected_classes = [d["class"] for d in detections]
    track_ids = [d["track_id"] for d in detections]
    confirmed_tracks = [t for t in track_ids if t != -1]

    anomaly_score = 0.0
    reasons = []

    # Rule 1 — strong CLIP semantic match
    if distance < clip_threshold:
        anomaly_score += 0.4
        reasons.append(f"Strong CLIP match (distance={distance:.3f})")

    # Rule 2 — person detected
    person_detected = any(c in suspicious_classes for c in detected_classes)
    if person_detected:
        anomaly_score += 0.3
        reasons.append(f"Person detected by YOLO")

    # Rule 3 — confirmed persistent tracks
    if len(confirmed_tracks) >= min_track_frames:
        anomaly_score += 0.3
        reasons.append(f"{len(confirmed_tracks)} confirmed tracks (persistent objects)")

    triggered = anomaly_score >= 0.5

    return AnomalyScore(
        frame_idx=result.get("frame_idx"),
        image_path=result.get("image_path"),
        scene=result.get("scene"),
        timestamp=result.get("timestamp"),
        clip_score=distance,
        detected_classes=detected_classes,
        track_ids=track_ids,
        anomaly_score=round(anomaly_score, 2),
        triggered=triggered,
        reasons=reasons,
    )


def score_results(results: list, **kwargs) -> list[AnomalyScore]:
    """Score a list of query results and return sorted by anomaly score."""
    scores = [score_frame(r, **kwargs) for r in results]
    return sorted(scores, key=lambda x: x.anomaly_score, reverse=True)