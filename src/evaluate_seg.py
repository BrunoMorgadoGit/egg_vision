"""Avalia um modelo de segmentacao no split val ou test."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_utils import PROJECT_ROOT, SEG_DATA_YAML, write_json  # noqa: E402
from train_seg import resolve_device  # noqa: E402
from validate_seg_dataset import validate_prepared_dataset  # noqa: E402


DEFAULT_MODEL = Path("runs/segment/eggvision_crack_seg/weights/best.pt")


def model_run_directory(model_path: Path) -> Path:
    if model_path.parent.name == "weights":
        return model_path.parent.parent
    return model_path.parent


def serializable_metrics(metrics) -> dict:
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    payload = {
        "box": {
            "precision": values.get("metrics/precision(B)"),
            "recall": values.get("metrics/recall(B)"),
            "map50": values.get("metrics/mAP50(B)"),
            "map50_95": values.get("metrics/mAP50-95(B)"),
        },
        "mask": {
            "precision": values.get("metrics/precision(M)"),
            "recall": values.get("metrics/recall(M)"),
            "map50": values.get("metrics/mAP50(M)"),
            "map50_95": values.get("metrics/mAP50-95(M)"),
        },
        "fitness": values.get("fitness"),
        "raw": values,
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }
    try:
        payload["per_class"] = metrics.summary()
    except (AttributeError, TypeError, ValueError):
        payload["per_class"] = []
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia best.pt de segmentacao, preferencialmente no teste."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=SEG_DATA_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"ERRO: modelo nao encontrado: {model_path}")
        return 2

    dataset_report = validate_prepared_dataset(args.data, report_path=None)
    if not dataset_report["valid"]:
        print("ERRO: dataset invalido; avaliacao nao executada.")
        for error in dataset_report["errors"][:20]:
            print(f"  - {error}")
        return 2

    split_stats = dataset_report["distribution"][args.split]
    if split_stats["images"] < 20 or split_stats["rachado_images"] < 10:
        print(
            "AVISO: conjunto pequeno; uma unica falha pode alterar fortemente as metricas. "
            "Os resultados nao sao conclusivos."
        )

    run_dir = model_run_directory(model_path)
    output_path = args.output.resolve() if args.output else (
        run_dir / "evaluation" / f"{args.split}_metrics.json"
    )
    device = resolve_device(args.device)
    previous_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        model = YOLO(str(model_path))
        if model.task != "segment":
            print("ERRO: checkpoint informado nao e de segmentacao.")
            return 2
        metrics = model.val(
            data=str(args.data.resolve()),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=str(run_dir / "evaluation"),
            name=f"{args.split}_ultralytics",
            plots=True,
            save_json=False,
            exist_ok=True,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"ERRO durante a avaliacao: {error}")
        return 1
    finally:
        os.chdir(previous_cwd)

    payload = serializable_metrics(metrics)
    payload.update(
        {
            "model": str(model_path),
            "data": str(args.data.resolve()),
            "split": args.split,
            "split_distribution": split_stats,
            "ultralytics_output": str(metrics.save_dir),
            "small_test_warning": args.split == "test"
            and (split_stats["images"] < 20 or split_stats["rachado_images"] < 10),
        }
    )
    write_json(output_path, payload)
    print("Metricas reais de bounding box:", payload["box"])
    print("Metricas reais de mascara:", payload["mask"])
    print(f"JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
