#!/usr/bin/env python3
"""Laptop-safe unit tests: no torch, CUDA, or LiveEdit import required."""

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "wan" / "modules" / "history_layout.py"
SPEC = importlib.util.spec_from_file_location("history_layout_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def chunks(count):
    return [
        MODULE.ChunkSpan(i, i * 3, 3, i * 30, 30)
        for i in range(count)
    ]


class HistoryLayoutPlannerTest(unittest.TestCase):
    def test_sink_recent(self):
        planner = MODULE.SinkRecentPlanner(sink_count=2, recent_count=3)
        self.assertEqual(planner.select(chunks(7)), [0, 1, 4, 5, 6])

    def test_non_contiguous_stride(self):
        planner = MODULE.StridedHistoryPlanner(stride=2, recent_count=1)
        self.assertEqual(planner.select(chunks(7)), [0, 2, 4, 6])

    def test_explicit_order_and_current_invariant(self):
        planner = MODULE.ExplicitLayoutPlanner(
            {"6": [6, 4, 0, 2]}, order="selection"
        )
        self.assertEqual(planner.select(chunks(7)), [4, 0, 2, 6])

    def test_variable_anchor_schedule(self):
        planner = MODULE.AnchorRecentPlanner(
            anchors=[1], recent_count=2, schedule={"6": [3]}
        )
        self.assertEqual(planner.select(chunks(7)), [3, 5, 6])

    def test_scene_contrast(self):
        planner = MODULE.SceneContrastPlanner(
            {str(i): "a" if i < 3 else "b" for i in range(7)},
            contrast_count=2,
            recent_count=1,
        )
        self.assertEqual(planner.select(chunks(7)), [1, 2, 6])

    def test_scene_contrast_prefers_annotated_large_difference(self):
        planner = MODULE.SceneContrastPlanner(
            {"0": "indoor", "1": "forest", "2": "beach"},
            contrast_count=1,
            recent_count=1,
            scene_distance={"indoor|beach": 0.4, "forest|beach": 0.9},
        )
        self.assertEqual(planner.select(chunks(3)), [1, 2])


if __name__ == "__main__":
    unittest.main()
