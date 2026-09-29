#!/usr/bin/env python3
"""Laptop-safe manifest expansion tests."""

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "history_layout" / "manifest.py"
SPEC = importlib.util.spec_from_file_location("history_manifest_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ManifestTest(unittest.TestCase):
    def test_grid_and_case_scene_materialization(self):
        manifest = {
            "version": 1,
            "output_root": "out",
            "dataset": [{
                "id": "a", "source_path": "a.mp4", "instruction": "edit",
                "scene_by_chunk": {"0": "x"},
            }],
            "inference": {},
            "seeds": [7, 8],
            "variants": [{
                "id": "contrast",
                "policy": {
                    "policy": "chunk_layout",
                    "planner": {"type": "scene_contrast"},
                },
                "sweep": {"policy.planner.contrast_count": [1, 2]},
            }],
        }
        MODULE.validate_manifest(manifest)
        runs = MODULE.expand_runs(manifest)
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0].policy["planner"]["scene_by_chunk"], {"0": "x"})
        self.assertNotEqual(runs[0].run_id, runs[1].run_id)


if __name__ == "__main__":
    unittest.main()
