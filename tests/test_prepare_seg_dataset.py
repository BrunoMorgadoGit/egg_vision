from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prepare_seg_dataset import prepare_seg_dataset
from train_seg import validate_run_destination
from segmentation_utils import (
    allocate_group_counts,
    assign_group_splits,
    discover_source_images,
    egg_group_key,
    extract_egg_id,
    find_exact_duplicates,
    load_dataset_yaml,
    resolve_dataset_root,
)
from tests.helpers import VALID_POLYGON, write_image, write_label


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareSegDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw_dataset"
        self.annotations = self.root / "annotations"
        self.output = self.root / "dataset"
        self.report = self.root / "reports" / "prepare.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create_complete_source(self) -> list[Path]:
        paths = []
        normal_specs = [
            ("ovo01-foto1.jpg", 20),
            ("ovo01-foto2.jpg", 40),
            ("ovo02-foto1.jpg", 60),
            ("ovo03-foto1.jpg", 80),
        ]
        crack_specs = [
            ("ovo01-foto1.jpg", 120),
            ("ovo02-foto1.jpg", 150),
            ("ovo03-foto1.jpg", 180),
        ]
        for name, value in normal_specs:
            paths.append(write_image(self.raw / "normal" / "Normal" / name, value))
        for name, value in crack_specs:
            image = write_image(self.raw / "rachado" / "Rachado" / name, value)
            paths.append(image)
            write_label(
                self.annotations / "rachado" / "Rachado" / Path(name).with_suffix(".txt"),
                VALID_POLYGON,
            )
        return paths

    def test_extracts_egg_identifier(self) -> None:
        self.assertEqual(extract_egg_id("ovo01-foto2.jpg"), "ovo01")
        self.assertEqual(extract_egg_id("LOTE_OVO123_A.png"), "ovo123")

    def test_same_number_in_different_classes_is_not_same_egg(self) -> None:
        self.assertNotEqual(
            egg_group_key("normal", "ovo01-foto1.jpg"),
            egg_group_key("rachado", "ovo01-foto1.jpg"),
        )

    def test_expected_group_allocations_for_current_dataset(self) -> None:
        self.assertEqual(
            allocate_group_counts(16, (0.70, 0.15, 0.15)),
            {"train": 11, "val": 2, "test": 3},
        )
        self.assertEqual(
            allocate_group_counts(27, (0.70, 0.15, 0.15)),
            {"train": 19, "val": 4, "test": 4},
        )

    def test_same_egg_photos_receive_same_split(self) -> None:
        self.create_complete_source()
        records, invalid, missing = discover_source_images(self.raw)
        self.assertFalse(invalid)
        self.assertFalse(missing)
        assignments = assign_group_splits(records, seed=42)
        normal_ovo01 = [
            assignments[record.group_key]
            for record in records
            if record.group_key == "normal:ovo01"
        ]
        self.assertEqual(len(normal_ovo01), 2)
        self.assertEqual(len(set(normal_ovo01)), 1)

    def test_prepares_copies_empty_normal_labels_and_yaml(self) -> None:
        source_paths = self.create_complete_source()
        hashes_before = {path: file_hash(path) for path in source_paths}
        code = prepare_seg_dataset(
            raw_root=self.raw,
            annotations_root=self.annotations,
            output_root=self.output,
            report_path=self.report,
            seed=42,
        )
        self.assertEqual(code, 0)
        self.assertTrue(self.report.is_file())
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "prepared")

        normal_labels = sorted((self.output / "labels").rglob("normal__*.txt"))
        crack_labels = sorted((self.output / "labels").rglob("rachado__*.txt"))
        self.assertEqual(len(normal_labels), 4)
        self.assertEqual(len(crack_labels), 3)
        self.assertTrue(all(path.read_text(encoding="utf-8") == "" for path in normal_labels))
        self.assertTrue(all(path.read_text(encoding="utf-8").strip() for path in crack_labels))
        self.assertFalse(any(path.is_symlink() for path in self.output.rglob("*")))
        self.assertEqual(hashes_before, {path: file_hash(path) for path in source_paths})

        yaml_path = self.output / "data.yaml"
        payload, errors = load_dataset_yaml(yaml_path)
        self.assertFalse(errors)
        self.assertEqual(payload["names"], {0: "rachadura"})
        self.assertEqual(resolve_dataset_root(payload, yaml_path), self.output.resolve())

    def test_missing_crack_annotation_blocks_copy(self) -> None:
        write_image(self.raw / "normal" / "Normal" / "ovo01-foto1.jpg", 30)
        write_image(self.raw / "rachado" / "Rachado" / "ovo01-foto1.jpg", 160)
        code = prepare_seg_dataset(
            raw_root=self.raw,
            annotations_root=self.annotations,
            output_root=self.output,
            report_path=self.report,
        )
        self.assertEqual(code, 2)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(len(report["missing_crack_annotations"]), 1)
        self.assertFalse(any((self.output / "images").rglob("*.jpg")))

    def test_detects_exact_duplicates(self) -> None:
        first = write_image(self.raw / "normal" / "Normal" / "ovo01-foto1.jpg", 70)
        second = self.raw / "normal" / "Normal" / "ovo02-foto1.jpg"
        second.parent.mkdir(parents=True, exist_ok=True)
        second.write_bytes(first.read_bytes())
        (self.raw / "rachado").mkdir(parents=True)
        records, _, _ = discover_source_images(self.raw)
        duplicates = find_exact_duplicates(records)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual({item.group_key for item in duplicates[0]}, {"normal:ovo01", "normal:ovo02"})

    def test_existing_segmentation_run_is_not_silently_overwritten(self) -> None:
        project = self.root / "runs" / "segment"
        run_dir = project / "eggvision_crack_seg"
        run_dir.mkdir(parents=True)
        (run_dir / "args.yaml").write_text("task: segment\n", encoding="utf-8")
        errors = validate_run_destination(project, "eggvision_crack_seg", resume=None)
        self.assertEqual(len(errors), 1)
        self.assertIn("ja possui arquivos", errors[0])


if __name__ == "__main__":
    unittest.main()
