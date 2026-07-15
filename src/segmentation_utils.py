"""Utilitarios compartilhados do pipeline de segmentacao de rachaduras."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_DIR = PROJECT_ROOT / "raw_dataset"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations" / "egg_crack_seg"
SEG_DATASET_DIR = PROJECT_ROOT / "datasets" / "egg_crack_seg"
SEG_DATA_YAML = SEG_DATASET_DIR / "data.yaml"
SEGMENT_RUNS_DIR = PROJECT_ROOT / "runs" / "segment"
CLASSIFIER_BEST_MODEL = (
    PROJECT_ROOT / "runs" / "classify" / "eggvision_mvp" / "weights" / "best.pt"
)

SOURCE_CLASSES = ("normal", "rachado")
SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IGNORED_FILENAMES = {".gitkeep", ".ds_store", "thumbs.db"}
CRACK_CLASS_ID = 0
CRACK_CLASS_NAME = "rachadura"
EGG_ID_PATTERN = re.compile(r"(ovo\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SourceImage:
    """Imagem-fonte com a proveniencia necessaria para um split seguro."""

    path: Path
    source_class: str
    class_root: Path
    relative_path: Path
    egg_id: str
    group_key: str
    sha256: str

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "source_class": self.source_class,
            "relative_path": self.relative_path.as_posix(),
            "egg_id": self.egg_id,
            "group_key": self.group_key,
            "sha256": self.sha256,
        }


def extract_egg_id(filename: str | Path) -> str:
    """Extrai ``ovoXX``; usa o stem completo quando o padrao nao existe."""
    stem = Path(filename).stem
    match = EGG_ID_PATTERN.search(stem)
    return match.group(1).lower() if match else stem.lower()


def egg_group_key(source_class: str, filename: str | Path) -> str:
    """Combina classe e ID para que normal/ovo01 e rachado/ovo01 nao colidam."""
    if source_class not in SOURCE_CLASSES:
        raise ValueError(f"Classe-fonte invalida: {source_class}")
    return f"{source_class}:{extract_egg_id(filename)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "nao e um arquivo"
    if path.name.lower() in IGNORED_FILENAMES:
        return False, "arquivo auxiliar"
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"extensao nao suportada ({path.suffix or 'sem extensao'})"
    if path.stat().st_size == 0:
        return False, "arquivo vazio"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return False, "OpenCV nao conseguiu abrir"
    return True, "ok"


def discover_source_images(raw_root: Path) -> tuple[list[SourceImage], list[dict], list[str]]:
    """Descobre imagens normais/rachadas recursivamente e valida cada arquivo."""
    records: list[SourceImage] = []
    invalid: list[dict] = []
    missing_dirs: list[str] = []

    for source_class in SOURCE_CLASSES:
        class_root = raw_root / source_class
        if not class_root.is_dir():
            missing_dirs.append(str(class_root))
            continue

        paths = sorted(
            (path for path in class_root.rglob("*") if path.is_file()),
            key=lambda item: item.as_posix().lower(),
        )
        for path in paths:
            if path.name.lower() in IGNORED_FILENAMES:
                continue
            valid, reason = inspect_image(path)
            if not valid:
                invalid.append({"path": str(path), "reason": reason})
                continue
            relative = path.relative_to(class_root)
            records.append(
                SourceImage(
                    path=path,
                    source_class=source_class,
                    class_root=class_root,
                    relative_path=relative,
                    egg_id=extract_egg_id(path.name),
                    group_key=egg_group_key(source_class, path.name),
                    sha256=sha256_file(path),
                )
            )

    return records, invalid, missing_dirs


def source_annotation_path(record: SourceImage, annotations_root: Path) -> Path:
    """Espelha a arvore da classe bruta dentro de annotations/."""
    return (annotations_root / record.source_class / record.relative_path).with_suffix(".txt")


def prepared_basename(record: SourceImage) -> str:
    """Evita colisao entre classes que compartilham nomes como ovo01-foto1.jpg."""
    return f"{record.source_class}__{record.path.stem}{record.path.suffix.lower()}"


def prepared_source_class(filename: str | Path) -> str | None:
    prefix = Path(filename).name.split("__", 1)[0].lower()
    return prefix if prefix in SOURCE_CLASSES else None


def validate_segmentation_text(
    text: str,
    *,
    require_non_empty: bool,
    label_name: str = "label",
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    """Valida texto YOLO Segmentation e devolve os poligonos normalizados."""
    polygons: list[list[tuple[float, float]]] = []
    errors: list[str] = []
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not non_empty_lines:
        if require_non_empty:
            errors.append(f"{label_name}: label vazio para imagem rachada")
        return polygons, errors

    for line_number, line in enumerate(non_empty_lines, 1):
        tokens = line.split()
        location = f"{label_name}: linha {line_number}"

        try:
            class_id = int(tokens[0])
        except (IndexError, ValueError):
            errors.append(f"{location}: classe nao numerica")
            continue

        if class_id != CRACK_CLASS_ID:
            errors.append(
                f"{location}: classe {class_id} invalida; esperado {CRACK_CLASS_ID}"
            )

        coordinate_tokens = tokens[1:]
        if len(coordinate_tokens) < 6:
            errors.append(f"{location}: poligono com menos de tres pontos")
            continue
        if len(coordinate_tokens) % 2 != 0:
            errors.append(f"{location}: quantidade impar de coordenadas")
            continue

        try:
            coordinates = [float(value) for value in coordinate_tokens]
        except ValueError:
            errors.append(f"{location}: coordenada nao numerica")
            continue

        if any(not math.isfinite(value) for value in coordinates):
            errors.append(f"{location}: coordenada nao finita")
            continue
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            errors.append(f"{location}: coordenada fora do intervalo 0 a 1")
            continue

        points = list(zip(coordinates[0::2], coordinates[1::2]))
        polygons.append(points)

    return polygons, errors


def validate_segmentation_label(
    path: Path,
    *,
    require_non_empty: bool,
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    if not path.is_file():
        return [], [f"{path}: label ausente"]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [], [f"{path}: nao foi possivel ler ({error})"]
    return validate_segmentation_text(
        text,
        require_non_empty=require_non_empty,
        label_name=str(path),
    )


def allocate_group_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    """Aloca grupos com pelo menos um por split quando total >= 3."""
    if total < 0:
        raise ValueError("Total de grupos nao pode ser negativo")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("As proporcoes de split devem somar 1.0")
    if total == 0:
        return dict.fromkeys(SPLITS, 0)
    if total == 1:
        return {"train": 1, "val": 0, "test": 0}
    if total == 2:
        return {"train": 1, "val": 1, "test": 0}

    train_count = round(total * ratios[0])
    val_count = round(total * ratios[1])
    counts = {
        "train": train_count,
        "val": val_count,
        "test": total - train_count - val_count,
    }

    for split in SPLITS:
        if counts[split] > 0:
            continue
        donor = max(SPLITS, key=lambda name: counts[name])
        if counts[donor] <= 1:
            raise ValueError("Nao foi possivel garantir um grupo por split")
        counts[donor] -= 1
        counts[split] += 1

    return counts


def assign_group_splits(
    records: Iterable[SourceImage],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Atribui cada chave classe+ovo a exatamente um split."""
    groups_by_class: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups_by_class[record.source_class].add(record.group_key)

    assignments: dict[str, str] = {}
    for class_index, source_class in enumerate(SOURCE_CLASSES):
        keys = sorted(groups_by_class.get(source_class, set()))
        rng = random.Random(seed + (class_index * 1009))
        rng.shuffle(keys)
        counts = allocate_group_counts(len(keys), ratios)

        offset = 0
        for split in SPLITS:
            split_keys = keys[offset:offset + counts[split]]
            for group_key in split_keys:
                assignments[group_key] = split
            offset += counts[split]

    return assignments


def find_exact_duplicates(records: Iterable[SourceImage]) -> list[list[SourceImage]]:
    by_hash: dict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    return [items for items in by_hash.values() if len(items) > 1]


def ensure_dataset_directories(dataset_root: Path) -> None:
    for split in SPLITS:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def dataset_has_payload(dataset_root: Path) -> bool:
    for kind in ("images", "labels"):
        root = dataset_root / kind
        if root.exists() and any(path.is_file() for path in root.rglob("*")):
            return True
    return False


def yaml_path_value(dataset_root: Path) -> str:
    try:
        return dataset_root.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(dataset_root.resolve())


def write_dataset_yaml(dataset_root: Path, yaml_path: Path | None = None) -> Path:
    yaml_path = yaml_path or dataset_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": yaml_path_value(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {CRACK_CLASS_ID: CRACK_CLASS_NAME},
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path


def load_dataset_yaml(yaml_path: Path) -> tuple[dict, list[str]]:
    if not yaml_path.is_file():
        return {}, [f"YAML nao encontrado: {yaml_path}"]
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return {}, [f"YAML invalido: {error}"]
    if not isinstance(payload, dict):
        return {}, ["YAML deve conter um mapeamento"]
    return payload, []


def resolve_dataset_root(payload: dict, yaml_path: Path) -> Path:
    configured = payload.get("path")
    if not configured:
        return yaml_path.parent.resolve()
    path = Path(str(configured)).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    yaml_candidate = (yaml_path.parent / path).resolve()
    if project_candidate.exists() or not yaml_candidate.exists():
        return project_candidate
    return yaml_candidate


def _json_default(value):
    """Converte escalares/arrays NumPy e Paths retornados por bibliotecas."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Objeto {type(value).__name__} nao e serializavel em JSON")


def write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
