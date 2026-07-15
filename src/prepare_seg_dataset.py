"""Prepara o dataset YOLO Segmentation sem modificar o dataset bruto."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_utils import (  # noqa: E402
    ANNOTATIONS_DIR,
    RAW_DATASET_DIR,
    SEG_DATASET_DIR,
    SEGMENT_RUNS_DIR,
    SOURCE_CLASSES,
    SPLITS,
    SourceImage,
    assign_group_splits,
    dataset_has_payload,
    discover_source_images,
    ensure_dataset_directories,
    find_exact_duplicates,
    prepared_basename,
    source_annotation_path,
    validate_segmentation_label,
    write_dataset_yaml,
    write_json,
)


DEFAULT_REPORT = SEGMENT_RUNS_DIR / "dataset_preparation_report.json"


def _empty_distribution() -> dict:
    return {
        split: {
            "images": 0,
            "eggs": 0,
            "normal_images": 0,
            "normal_eggs": 0,
            "rachado_images": 0,
            "rachado_eggs": 0,
        }
        for split in SPLITS
    }


def distribution_for(
    records: list[SourceImage], assignments: dict[str, str]
) -> dict:
    distribution = _empty_distribution()
    eggs: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        split = assignments[record.group_key]
        distribution[split]["images"] += 1
        distribution[split][f"{record.source_class}_images"] += 1
        eggs[(split, record.source_class)].add(record.group_key)

    for split in SPLITS:
        all_eggs = set()
        for source_class in SOURCE_CLASSES:
            class_eggs = eggs[(split, source_class)]
            distribution[split][f"{source_class}_eggs"] = len(class_eggs)
            all_eggs.update(class_eggs)
        distribution[split]["eggs"] = len(all_eggs)

    return distribution


def inspect_annotations(
    records: list[SourceImage], annotations_root: Path
) -> tuple[list[str], list[dict], list[str]]:
    missing: list[str] = []
    invalid: list[dict] = []
    normal_annotation_warnings: list[str] = []

    for record in records:
        label_path = source_annotation_path(record, annotations_root)
        if record.source_class == "normal":
            if label_path.is_file() and label_path.read_text(encoding="utf-8").strip():
                normal_annotation_warnings.append(str(label_path))
            continue

        if not label_path.is_file():
            missing.append(str(label_path))
            continue
        _, errors = validate_segmentation_label(label_path, require_non_empty=True)
        for error in errors:
            invalid.append({"path": str(label_path), "error": error})

    return missing, invalid, normal_annotation_warnings


def inspect_name_collisions(records: list[SourceImage]) -> list[dict]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_name[prepared_basename(record).lower()].append(str(record.path))
    return [
        {"output_name": name, "sources": paths}
        for name, paths in sorted(by_name.items())
        if len(paths) > 1
    ]


def build_report(
    raw_root: Path,
    annotations_root: Path,
    output_root: Path,
    records: list[SourceImage],
    invalid_images: list[dict],
    missing_dirs: list[str],
    assignments: dict[str, str],
) -> dict:
    missing_labels, invalid_labels, normal_label_warnings = inspect_annotations(
        records, annotations_root
    )
    duplicates = find_exact_duplicates(records)
    duplicate_items = [
        {
            "sha256": group[0].sha256,
            "group_keys": sorted({item.group_key for item in group}),
            "paths": [str(item.path) for item in group],
        }
        for group in duplicates
    ]
    duplicate_conflicts = [
        item for item in duplicate_items if len(item["group_keys"]) > 1
    ]
    collisions = inspect_name_collisions(records)

    counts = Counter(record.source_class for record in records)
    egg_counts = {
        source_class: len(
            {record.group_key for record in records if record.source_class == source_class}
        )
        for source_class in SOURCE_CLASSES
    }

    errors: list[str] = []
    if missing_dirs:
        errors.append(f"{len(missing_dirs)} pasta(s) fonte ausente(s)")
    if invalid_images:
        errors.append(f"{len(invalid_images)} imagem(ns) invalida(s)")
    if missing_labels:
        errors.append(f"{len(missing_labels)} imagem(ns) rachada(s) sem anotacao")
    if invalid_labels:
        errors.append(f"{len(invalid_labels)} erro(s) em labels de rachadura")
    if normal_label_warnings:
        errors.append(
            f"{len(normal_label_warnings)} label(s) nao vazio(s) associado(s) a imagens normais"
        )
    if duplicate_conflicts:
        errors.append(
            f"{len(duplicate_conflicts)} duplicata(s) exata(s) com IDs de ovo diferentes"
        )
    if collisions:
        errors.append(f"{len(collisions)} colisao(oes) de nome no dataset de saida")
    if counts["normal"] == 0:
        errors.append("nenhuma imagem normal valida")
    if counts["rachado"] == 0:
        errors.append("nenhuma imagem rachada valida")

    return {
        "status": "blocked" if errors else "ready",
        "raw_root": str(raw_root),
        "annotations_root": str(annotations_root),
        "output_root": str(output_root),
        "seed": None,
        "source": {
            "images": dict(counts),
            "eggs": egg_counts,
            "total_images": len(records),
            "total_eggs": sum(egg_counts.values()),
        },
        "distribution": distribution_for(records, assignments) if records else _empty_distribution(),
        "missing_source_directories": missing_dirs,
        "invalid_images": invalid_images,
        "missing_crack_annotations": missing_labels,
        "invalid_crack_annotations": invalid_labels,
        "non_empty_normal_annotations": normal_label_warnings,
        "exact_duplicates": duplicate_items,
        "duplicate_conflicts": duplicate_conflicts,
        "output_name_collisions": collisions,
        "possible_split_leaks": [],
        "errors": errors,
    }


def _print_report(report: dict, report_path: Path) -> None:
    source = report["source"]
    print("\nFonte:")
    print(
        f"  normal:  {source['images'].get('normal', 0)} imagens / "
        f"{source['eggs'].get('normal', 0)} ovos"
    )
    print(
        f"  rachado: {source['images'].get('rachado', 0)} imagens / "
        f"{source['eggs'].get('rachado', 0)} ovos"
    )
    print("\nSplit planejado por ovo:")
    for split, stats in report["distribution"].items():
        print(
            f"  {split:5s}: {stats['images']:2d} imagens / {stats['eggs']:2d} ovos "
            f"(normal {stats['normal_images']}/{stats['normal_eggs']}, "
            f"rachado {stats['rachado_images']}/{stats['rachado_eggs']})"
        )

    missing = report["missing_crack_annotations"]
    invalid = report["invalid_crack_annotations"]
    print(f"\nAnotacoes rachadas ausentes: {len(missing)}")
    for path in missing[:10]:
        print(f"  - {path}")
    if len(missing) > 10:
        print(f"  ... e mais {len(missing) - 10}")
    print(f"Anotacoes rachadas invalidas: {len(invalid)}")
    print(f"Duplicatas exatas: {len(report['exact_duplicates'])}")
    print(f"Relatorio: {report_path}")


def prepare_seg_dataset(
    *,
    raw_root: Path = RAW_DATASET_DIR,
    annotations_root: Path = ANNOTATIONS_DIR,
    output_root: Path = SEG_DATASET_DIR,
    report_path: Path = DEFAULT_REPORT,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    rebuild: bool = False,
) -> int:
    """Prepara o dataset; retorna zero apenas quando todas as labels sao validas."""
    raw_root = raw_root.resolve()
    annotations_root = annotations_root.resolve()
    output_root = output_root.resolve()
    report_path = report_path.resolve()

    annotations_root.joinpath("rachado").mkdir(parents=True, exist_ok=True)

    if rebuild and output_root.exists():
        shutil.rmtree(output_root)
    elif dataset_has_payload(output_root):
        print(f"ERRO: dataset de saida ja contem arquivos: {output_root}")
        print("Use --rebuild explicitamente para recria-lo.")
        return 2

    ensure_dataset_directories(output_root)
    yaml_path = write_dataset_yaml(output_root)

    records, invalid_images, missing_dirs = discover_source_images(raw_root)
    assignments = assign_group_splits(records, ratios=ratios, seed=seed)
    report = build_report(
        raw_root,
        annotations_root,
        output_root,
        records,
        invalid_images,
        missing_dirs,
        assignments,
    )
    report["seed"] = seed
    report["ratios"] = {split: ratio for split, ratio in zip(SPLITS, ratios)}
    report["data_yaml"] = str(yaml_path)

    if report["errors"]:
        write_json(report_path, report)
        print("Dataset de segmentacao bloqueado:")
        for error in report["errors"]:
            print(f"  - {error}")
        _print_report(report, report_path)
        print("\nNenhuma imagem foi copiada para o dataset de segmentacao.")
        return 2

    copied = Counter()
    for record in records:
        split = assignments[record.group_key]
        image_name = prepared_basename(record)
        image_dest = output_root / "images" / split / image_name
        label_dest = output_root / "labels" / split / f"{Path(image_name).stem}.txt"
        shutil.copy2(record.path, image_dest)
        if record.source_class == "normal":
            label_dest.write_text("", encoding="utf-8")
        else:
            shutil.copy2(source_annotation_path(record, annotations_root), label_dest)
        copied[(split, record.source_class)] += 1

    report["status"] = "prepared"
    report["copied"] = {
        split: {
            source_class: copied[(split, source_class)]
            for source_class in SOURCE_CLASSES
        }
        for split in SPLITS
    }
    write_json(report_path, report)
    _print_report(report, report_path)
    print("\nDataset de segmentacao preparado com copias reais (shutil.copy2).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara imagens e labels para YOLO Segmentation por ID de ovo."
    )
    parser.add_argument("--raw", type=Path, default=RAW_DATASET_DIR)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=SEG_DATASET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove explicitamente apenas o diretorio --output antes de recriar.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6 or any(value < 0 for value in ratios):
        print("ERRO: --train, --val e --test devem ser nao negativos e somar 1.0.")
        return 2
    return prepare_seg_dataset(
        raw_root=args.raw,
        annotations_root=args.annotations,
        output_root=args.output,
        report_path=args.report,
        ratios=ratios,
        seed=args.seed,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    raise SystemExit(main())
