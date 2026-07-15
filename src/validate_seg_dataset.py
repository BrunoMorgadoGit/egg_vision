"""Valida anotações-fonte e datasets YOLO de segmentacao preparados."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_seg_dataset import inspect_annotations  # noqa: E402
from segmentation_utils import (  # noqa: E402
    ANNOTATIONS_DIR,
    CRACK_CLASS_ID,
    CRACK_CLASS_NAME,
    IGNORED_FILENAMES,
    RAW_DATASET_DIR,
    SEG_DATA_YAML,
    SEGMENT_RUNS_DIR,
    SOURCE_CLASSES,
    SPLITS,
    SUPPORTED_IMAGE_EXTENSIONS,
    discover_source_images,
    egg_group_key,
    extract_egg_id,
    inspect_image,
    load_dataset_yaml,
    prepared_source_class,
    resolve_dataset_root,
    sha256_file,
    validate_segmentation_label,
    write_json,
)


DEFAULT_REPORT = SEGMENT_RUNS_DIR / "dataset_validation_report.json"


def _normalize_names(names) -> dict[int, str] | None:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError):
            return None
    return None


def validate_source_annotations(
    *,
    raw_root: Path = RAW_DATASET_DIR,
    annotations_root: Path = ANNOTATIONS_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict:
    records, invalid_images, missing_dirs = discover_source_images(raw_root.resolve())
    missing, invalid_labels, normal_labels = inspect_annotations(
        records, annotations_root.resolve()
    )
    counts = Counter(record.source_class for record in records)
    eggs = {
        source_class: len(
            {record.group_key for record in records if record.source_class == source_class}
        )
        for source_class in SOURCE_CLASSES
    }
    errors = []
    errors.extend(f"Pasta fonte ausente: {path}" for path in missing_dirs)
    errors.extend(
        f"Imagem invalida: {item['path']} ({item['reason']})" for item in invalid_images
    )
    errors.extend(f"Imagem rachada sem label: {path}" for path in missing)
    errors.extend(item["error"] for item in invalid_labels)
    errors.extend(f"Imagem normal com label nao vazio: {path}" for path in normal_labels)

    report = {
        "mode": "source_annotations",
        "valid": not errors,
        "raw_root": str(raw_root.resolve()),
        "annotations_root": str(annotations_root.resolve()),
        "images": dict(counts),
        "eggs": eggs,
        "missing_crack_annotations": missing,
        "invalid_crack_annotations": invalid_labels,
        "non_empty_normal_annotations": normal_labels,
        "invalid_images": invalid_images,
        "errors": errors,
        "warnings": [],
    }
    write_json(report_path.resolve(), report)
    return report


def validate_prepared_dataset(
    data_path: Path = SEG_DATA_YAML,
    *,
    report_path: Path | None = DEFAULT_REPORT,
) -> dict:
    data_path = data_path.resolve()
    payload, yaml_errors = load_dataset_yaml(data_path)
    errors = list(yaml_errors)
    warnings: list[str] = []
    root = resolve_dataset_root(payload, data_path) if payload else data_path.parent

    required_keys = ("train", "val", "test", "names")
    for key in required_keys:
        if key not in payload:
            errors.append(f"YAML sem chave obrigatoria: {key}")

    names = _normalize_names(payload.get("names"))
    if names != {CRACK_CLASS_ID: CRACK_CLASS_NAME}:
        errors.append(
            f"YAML deve definir somente 0: {CRACK_CLASS_NAME}; encontrado {names}"
        )

    image_counts = {split: Counter() for split in SPLITS}
    egg_sets = {split: {source_class: set() for source_class in SOURCE_CLASSES} for split in SPLITS}
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_records: dict[str, list[dict]] = defaultdict(list)
    all_image_stems: dict[tuple[str, str], Path] = {}
    all_label_stems: dict[tuple[str, str], Path] = {}

    for split in SPLITS:
        image_config = payload.get(split, f"images/{split}")
        image_dir = (root / str(image_config)).resolve()
        label_dir = (root / "labels" / split).resolve()

        if not image_dir.is_dir():
            errors.append(f"Diretorio de imagens inexistente: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"Diretorio de labels inexistente: {label_dir}")
            continue

        image_files = sorted(
            (path for path in image_dir.rglob("*") if path.is_file()),
            key=lambda item: item.as_posix().lower(),
        )
        label_files = sorted(
            (path for path in label_dir.rglob("*") if path.is_file()),
            key=lambda item: item.as_posix().lower(),
        )

        for path in image_files:
            if path.name.lower() in IGNORED_FILENAMES:
                continue
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                errors.append(f"Extensao de imagem nao suportada: {path}")
                continue
            valid, reason = inspect_image(path)
            if not valid:
                errors.append(f"Imagem corrompida/invalida: {path} ({reason})")
                continue

            source_class = prepared_source_class(path.name)
            if source_class is None:
                errors.append(
                    f"Imagem sem prefixo de origem normal__ ou rachado__: {path}"
                )
                continue

            stem_key = (split, path.stem)
            all_image_stems[stem_key] = path
            image_counts[split][source_class] += 1
            group_key = egg_group_key(source_class, path.name)
            group_splits[group_key].add(split)
            egg_sets[split][source_class].add(group_key)
            hash_records[sha256_file(path)].append(
                {
                    "path": str(path),
                    "split": split,
                    "group_key": group_key,
                    "source_class": source_class,
                }
            )

            label_path = label_dir / f"{path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"Imagem sem label: {path}")
                continue
            _, label_errors = validate_segmentation_label(
                label_path, require_non_empty=(source_class == "rachado")
            )
            errors.extend(label_errors)
            if source_class == "normal" and label_path.read_text(encoding="utf-8").strip():
                errors.append(f"Label nao vazio em imagem normal: {label_path}")

        for path in label_files:
            if path.name.lower() in IGNORED_FILENAMES:
                continue
            if path.suffix.lower() != ".txt":
                errors.append(f"Arquivo de label com extensao invalida: {path}")
                continue
            stem_key = (split, path.stem)
            all_label_stems[stem_key] = path
            if stem_key not in all_image_stems:
                errors.append(f"Label sem imagem: {path}")

    leaks = {
        group_key: sorted(splits)
        for group_key, splits in group_splits.items()
        if len(splits) > 1
    }
    for group_key, splits in leaks.items():
        errors.append(f"Mesmo ovo em splits diferentes: {group_key} -> {splits}")

    exact_duplicates = [items for items in hash_records.values() if len(items) > 1]
    for duplicate_group in exact_duplicates:
        duplicate_splits = {item["split"] for item in duplicate_group}
        if len(duplicate_splits) > 1:
            errors.append(
                "Duplicata exata em splits diferentes: "
                + ", ".join(item["path"] for item in duplicate_group)
            )
        else:
            warnings.append(
                "Duplicata exata no mesmo split: "
                + ", ".join(item["path"] for item in duplicate_group)
            )

    total_by_class = {
        source_class: sum(image_counts[split][source_class] for split in SPLITS)
        for source_class in SOURCE_CLASSES
    }
    for source_class in SOURCE_CLASSES:
        if total_by_class[source_class] >= 3:
            for split in SPLITS:
                if image_counts[split][source_class] == 0:
                    errors.append(
                        f"Split {split} sem exemplo {source_class}, embora existam "
                        f"{total_by_class[source_class]} imagens no total"
                    )

    total_images = sum(sum(counter.values()) for counter in image_counts.values())
    if total_images == 0:
        errors.append("Dataset preparado nao possui imagens")

    distribution = {
        split: {
            "images": sum(image_counts[split].values()),
            "eggs": sum(len(egg_sets[split][name]) for name in SOURCE_CLASSES),
            "normal_images": image_counts[split]["normal"],
            "normal_eggs": len(egg_sets[split]["normal"]),
            "rachado_images": image_counts[split]["rachado"],
            "rachado_eggs": len(egg_sets[split]["rachado"]),
        }
        for split in SPLITS
    }

    report = {
        "mode": "prepared_dataset",
        "valid": not errors,
        "data_yaml": str(data_path),
        "dataset_root": str(root),
        "distribution": distribution,
        "exact_duplicates": exact_duplicates,
        "split_leaks": leaks,
        "errors": errors,
        "warnings": warnings,
    }
    if report_path is not None:
        write_json(report_path.resolve(), report)
    return report


def print_report(report: dict, report_path: Path) -> None:
    print(f"Modo: {report['mode']}")
    if report["mode"] == "source_annotations":
        print(
            f"Imagens: normal={report['images'].get('normal', 0)}, "
            f"rachado={report['images'].get('rachado', 0)}"
        )
        print(
            f"Ovos: normal={report['eggs'].get('normal', 0)}, "
            f"rachado={report['eggs'].get('rachado', 0)}"
        )
        print(f"Labels rachadas ausentes: {len(report['missing_crack_annotations'])}")
        print(f"Labels rachadas invalidas: {len(report['invalid_crack_annotations'])}")
    else:
        for split, stats in report["distribution"].items():
            print(
                f"{split:5s}: {stats['images']} imagens / {stats['eggs']} ovos "
                f"(normal={stats['normal_images']}, rachado={stats['rachado_images']})"
            )

    for warning in report["warnings"]:
        print(f"AVISO: {warning}")
    for error in report["errors"]:
        print(f"ERRO: {error}")
    print(f"Relatorio JSON: {report_path.resolve()}")
    print("VALIDO" if report["valid"] else "INVALIDO (erro impeditivo)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida o dataset ou as anotacoes-fonte de segmentacao."
    )
    parser.add_argument("--data", type=Path, default=SEG_DATA_YAML)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-annotations",
        action="store_true",
        help="Valida raw_dataset e annotations antes da preparacao dos splits.",
    )
    parser.add_argument("--raw", type=Path, default=RAW_DATASET_DIR)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.source_annotations:
        report = validate_source_annotations(
            raw_root=args.raw,
            annotations_root=args.annotations,
            report_path=args.report,
        )
    else:
        report = validate_prepared_dataset(args.data, report_path=args.report)
    print_report(report, args.report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
