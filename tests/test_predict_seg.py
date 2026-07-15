from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from predict_seg import (
    DEFAULT_MODEL,
    prediction_from_result,
    save_prediction_json,
    summarize_mask_predictions,
)
from segmentation_utils import CLASSIFIER_BEST_MODEL
from segmentation_utils import write_json


class PredictSegTests(unittest.TestCase):
    def test_without_masks_is_normal(self) -> None:
        summary, masks = summarize_mask_predictions(
            image_name="normal.jpg", image_shape=(20, 30), threshold=0.25
        )
        self.assertEqual(summary["status"], "NORMAL")
        self.assertEqual(summary["crack_count"], 0)
        self.assertIsNone(summary["max_confidence"])
        self.assertEqual(masks, [])

    def test_mask_above_threshold_is_cracked(self) -> None:
        mask = np.zeros((1, 10, 10), dtype=np.float32)
        mask[0, 2:5, 3:7] = 1.0
        summary, masks = summarize_mask_predictions(
            image_name="rachado.jpg",
            image_shape=(10, 10),
            threshold=0.25,
            confidences=np.asarray([0.91]),
            class_ids=np.asarray([0]),
            masks=mask,
        )
        self.assertEqual(summary["status"], "RACHADO")
        self.assertEqual(summary["crack_count"], 1)
        self.assertAlmostEqual(summary["max_confidence"], 0.91)
        self.assertEqual(summary["detections"][0]["mask_area_pixels"], 12)
        self.assertEqual(len(masks), 1)

    def test_mask_below_threshold_is_normal(self) -> None:
        summary, _ = summarize_mask_predictions(
            image_name="duvidoso.jpg",
            image_shape=(10, 10),
            threshold=0.25,
            confidences=np.asarray([0.20]),
            class_ids=np.asarray([0]),
            masks=np.ones((1, 10, 10), dtype=np.float32),
        )
        self.assertEqual(summary["status"], "NORMAL")

    def test_mock_yolo_result_is_interpreted(self) -> None:
        result = SimpleNamespace(
            boxes=SimpleNamespace(conf=np.asarray([0.8]), cls=np.asarray([0])),
            masks=SimpleNamespace(data=np.ones((1, 4, 4), dtype=np.float32)),
        )
        summary, masks = prediction_from_result(
            result,
            image_name="mock.jpg",
            image_shape=(8, 8),
            threshold=0.25,
        )
        self.assertEqual(summary["status"], "RACHADO")
        self.assertEqual(summary["detections"][0]["mask_area_pixels"], 64)
        self.assertEqual(masks[0].shape, (8, 8))

    def test_json_generation(self) -> None:
        summary, _ = summarize_mask_predictions(
            image_name="normal.jpg", image_shape=(10, 10), threshold=0.25
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.json"
            save_prediction_json(path, summary)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "NORMAL")
        self.assertEqual(payload["detections"], [])

    def test_json_accepts_numpy_scalars_from_ultralytics_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_json(path, {"images": np.int64(2), "curve": np.asarray([0.1, 0.2])})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["images"], 2)
        self.assertEqual(payload["curve"], [0.1, 0.2])

    def test_segmentation_defaults_do_not_target_classifier(self) -> None:
        segmentation_path = (ROOT / DEFAULT_MODEL).resolve()
        self.assertNotEqual(segmentation_path, CLASSIFIER_BEST_MODEL.resolve())
        self.assertIn("runs/segment", segmentation_path.as_posix())

    def test_all_segmentation_scripts_import(self) -> None:
        modules = (
            "segmentation_utils",
            "prepare_seg_dataset",
            "validate_seg_dataset",
            "annotate_cracks",
            "train_seg",
            "evaluate_seg",
            "predict_seg",
            "analyze_seg_training",
        )
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_classifier_hash_is_stable_during_unit_test(self) -> None:
        if not CLASSIFIER_BEST_MODEL.is_file():
            self.skipTest("Classificador existente nao encontrado neste ambiente")
        before = hashlib.sha256(CLASSIFIER_BEST_MODEL.read_bytes()).hexdigest()
        summarize_mask_predictions(
            image_name="normal.jpg", image_shape=(4, 4), threshold=0.25
        )
        after = hashlib.sha256(CLASSIFIER_BEST_MODEL.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
