# tests/test_cross_validation.py
"""
Cross-validation tests — verify CLIP retrieval results are confirmed by YOLO detections.
This tests the core assumption of the pipeline: that semantic search and structured
detection agree on what's in a frame.
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import search_videos


# Query → expected YOLO classes that should appear in results
CROSS_VALIDATION_CASES = [
    {
        "query": "cars on a road",
        "expected_classes": ["car", "truck"],
        "min_matches": 1,  # at least 1 result should have these classes
    },
    {
        "query": "person walking near road",
        "expected_classes": ["pedestrian", "person"],
        "min_matches": 1,
    },
    {
        "query": "bus on a street",
        "expected_classes": ["bus", "car"],
        "min_matches": 1,
    },
]


def get_all_classes(result):
    """Extract all detected class names from a result."""
    return [d["class"] for d in result.get("detections", [])]


def result_contains_class(result, expected_classes):
    """Check if a result contains at least one of the expected classes."""
    detected = get_all_classes(result)
    return any(cls in detected for cls in expected_classes)


class TestCrossValidation:

    def test_cars_query_returns_car_detections(self):
        """CLIP query for cars should return frames where YOLO detects cars."""
        results = search_videos("cars on a road", n_results=5)
        
        matches = [r for r in results if result_contains_class(r, ["car", "truck"])]
        
        print(f"\nQuery: 'cars on a road'")
        print(f"Results: {len(results)}")
        print(f"Confirmed by YOLO: {len(matches)}")
        for r in results:
            print(f"  scene={r['scene']} frame={r['frame_idx']} classes={get_all_classes(r)}")
        
        assert len(matches) >= 1, (
            f"Expected at least 1 result with car/truck detections, "
            f"got {len(matches)} out of {len(results)}"
        )

    def test_results_have_detections(self):
        """All query results should have YOLO detections populated."""
        results = search_videos("vehicles on street", n_results=3)
        
        for r in results:
            assert "detections" in r, f"Result missing detections key: {r['id']}"

    def test_cross_validation_rate(self):
        """
        At least 60% of results for a vehicle query should have vehicle detections.
        This measures how well CLIP and YOLO agree.
        """
        results = search_videos("cars on a road", n_results=5)
        
        confirmed = [r for r in results if result_contains_class(r, ["car", "truck", "bus"])]
        rate = len(confirmed) / len(results) if results else 0
        
        print(f"\nCross-validation rate: {rate:.0%} ({len(confirmed)}/{len(results)})")
        
        assert rate >= 0.6, (
            f"Cross-validation rate too low: {rate:.0%}. "
            f"CLIP and YOLO disagree on too many results."
        )

    @pytest.mark.parametrize("case", CROSS_VALIDATION_CASES)
    def test_parametrized_cross_validation(self, case):
        """Run cross-validation for multiple query types."""
        results = search_videos(case["query"], n_results=5)
        matches = [r for r in results if result_contains_class(r, case["expected_classes"])]
        
        assert len(matches) >= case["min_matches"], (
            f"Query '{case['query']}' — expected {case['min_matches']} confirmed results, "
            f"got {len(matches)}"
        )