# tests/test_anomaly.py
"""
Unit tests for anomaly scoring.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.anomaly import score_frame, score_results


# Mock results for testing
MOCK_RESULT_HIGH_ANOMALY = {
    "id": "scene-0061_7",
    "score": -0.59,
    "scene": "scene-0061",
    "frame_idx": 7,
    "image_path": "/fake/path.jpg",
    "timestamp": 1532402931198511,
    "detections": [
        {"class": "pedestrian", "confidence": 0.94, "track_id": 1, "bbox": []},
        {"class": "car", "confidence": 0.87, "track_id": 2, "bbox": []},
        {"class": "car", "confidence": 0.76, "track_id": 3, "bbox": []},
        {"class": "car", "confidence": 0.65, "track_id": 4, "bbox": []},
    ]
}

MOCK_RESULT_LOW_ANOMALY = {
    "id": "scene-0061_1",
    "score": -0.95,
    "scene": "scene-0061",
    "frame_idx": 1,
    "image_path": "/fake/path.jpg",
    "timestamp": 1532402927647951,
    "detections": []
}


class TestAnomalyScoring:

    def test_high_anomaly_triggers(self):
        """Frame with person + confirmed tracks + strong CLIP match should trigger."""
        result = score_frame(MOCK_RESULT_HIGH_ANOMALY, clip_threshold=1.4)
        assert result.triggered is True

    def test_low_anomaly_no_trigger(self):
        """Frame with no detections and weak CLIP match should not trigger."""
        result = score_frame(MOCK_RESULT_LOW_ANOMALY, clip_threshold=1.4)
        assert result.triggered is False

    def test_score_between_0_and_1(self):
        """Anomaly score should always be between 0 and 1."""
        for mock in [MOCK_RESULT_HIGH_ANOMALY, MOCK_RESULT_LOW_ANOMALY]:
            result = score_frame(mock)
            assert 0.0 <= result.anomaly_score <= 1.0

    def test_person_detection_adds_score(self):
        """Detecting a person should increase anomaly score."""
        result_with_person = score_frame(MOCK_RESULT_HIGH_ANOMALY)
        result_without_person = score_frame(MOCK_RESULT_LOW_ANOMALY)
        assert result_with_person.anomaly_score > result_without_person.anomaly_score

    def test_score_results_sorted(self):
        """score_results should return results sorted by anomaly score descending."""
        results = [MOCK_RESULT_LOW_ANOMALY, MOCK_RESULT_HIGH_ANOMALY]
        scored = score_results(results)
        assert scored[0].anomaly_score >= scored[1].anomaly_score

    def test_reasons_populated_when_triggered(self):
        """Triggered anomalies should have reasons explaining why."""
        result = score_frame(MOCK_RESULT_HIGH_ANOMALY)
        assert len(result.reasons) > 0