from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from segmentation_utils import validate_segmentation_text, write_dataset_yaml
from validate_seg_dataset import validate_prepared_dataset
from tests.helpers import VALID_POLYGON, write_image, write_label


class LabelValidationTests(unittest.TestCase):
    def test_coordinate_out_of_range_is_error(self) -> None:
        _, errors = validate_segmentation_text(
            "0 0.1 0.1 1.2 0.2 0.3 0.4\n", require_non_empty=True
        )
        self.assertTrue(any("fora do intervalo" in error for error in errors))

    def test_polygon_with_fewer_than_three_points_is_error(self) -> None:
        _, errors = validate_segmentation_text(
            "0 0.1 0.1 0.2 0.2\n", require_non_empty=True
        )
        self.assertTrue(any("menos de tres pontos" in error for error in errors))

    def test_class_other_than_zero_is_error(self) -> None:
        _, errors = validate_segmentation_text(
            "1 0.1 0.1 0.2 0.2 0.3 0.3\n", require_non_empty=True
        )
        self.assertTrue(any("classe 1 invalida" in error for error in errors))


class PreparedDatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "seg"
        self.yaml_path = self.root / "data.yaml"
        for split_index, split in enumerate(("train", "val", "test")):
            normal_name = f"normal__ovo{split_index + 1:02d}-foto1.jpg"
            crack_name = f"rachado__ovo{split_index + 1:02d}-foto1.jpg"
            write_image(self.root / "images" / split / normal_name, 20 + split_index * 20)
            write_image(self.root / "images" / split / crack_name, 130 + split_index * 20)
            write_label(self.root / "labels" / split / Path(normal_name).with_suffix(".txt"), "")
            write_label(
                self.root / "labels" / split / Path(crack_name).with_suffix(".txt"),
                VALID_POLYGON,
            )
        write_dataset_yaml(self.root, self.yaml_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_dataset_passes(self) -> None:
        report = validate_prepared_dataset(self.yaml_path, report_path=None)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["distribution"]["test"]["rachado_images"], 1)

    def test_image_without_label_is_error(self) -> None:
        (self.root / "labels" / "test" / "rachado__ovo03-foto1.txt").unlink()
        report = validate_prepared_dataset(self.yaml_path, report_path=None)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Imagem sem label" in error for error in report["errors"]))

    def test_non_empty_normal_label_is_error(self) -> None:
        write_label(
            self.root / "labels" / "val" / "normal__ovo02-foto1.txt",
            VALID_POLYGON,
        )
        report = validate_prepared_dataset(self.yaml_path, report_path=None)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Label nao vazio" in error for error in report["errors"]))

    def test_same_egg_in_multiple_splits_is_error(self) -> None:
        source = self.root / "images" / "test" / "normal__ovo03-foto1.jpg"
        target = self.root / "images" / "test" / "normal__ovo01-foto2.jpg"
        source.rename(target)
        old_label = self.root / "labels" / "test" / "normal__ovo03-foto1.txt"
        old_label.rename(self.root / "labels" / "test" / "normal__ovo01-foto2.txt")
        report = validate_prepared_dataset(self.yaml_path, report_path=None)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Mesmo ovo" in error for error in report["errors"]))

    def test_yaml_generation_has_expected_paths_and_class(self) -> None:
        text = self.yaml_path.read_text(encoding="utf-8")
        self.assertIn("train: images/train", text)
        self.assertIn("test: images/test", text)
        self.assertIn("0: rachadura", text)


if __name__ == "__main__":
    unittest.main()
