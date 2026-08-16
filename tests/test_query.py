# tests/test_query.py
"""
Unit tests for the query pipeline.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import search_videos


class TestSearchVideos:

    def test_returns_list(self):
        """search_videos should return a list."""
        results = search_videos("cars on a road", n_results=3)
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        """search_videos should return at most n_results results."""
        results = search_videos("cars on a road", n_results=3)
        assert len(results) <= 3

    def test_result_has_required_fields(self):
        """Each result should have required fields."""
        results = search_videos("cars on a road", n_results=2)
        required_fields = ["id", "score", "scene", "frame_idx", "image_path", "timestamp", "detections"]
        
        for r in results:
            for field in required_fields:
                assert field in r, f"Missing field '{field}' in result {r.get('id')}"

    def test_results_deduplicated_by_scene(self):
        """Results should not have duplicate scenes."""
        results = search_videos("cars on a road", n_results=5)
        scenes = [r["scene"] for r in results]
        assert len(scenes) == len(set(scenes)), "Duplicate scenes in results"

    def test_image_paths_exist(self):
        """Image paths in results should point to existing files."""
        results = search_videos("cars on a road", n_results=3)
        for r in results:
            assert os.path.exists(r["image_path"]), (
                f"Image not found: {r['image_path']}"
            )