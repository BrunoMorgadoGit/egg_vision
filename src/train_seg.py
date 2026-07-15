"""Treinamento separado do modelo YOLOv8 de segmentacao de rachaduras."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_utils import (  # noqa: E402
    PROJECT_ROOT,
    SEG_DATA_YAML,
    SEGMENT_RUNS_DIR,
)
from validate_seg_dataset import validate_prepared_dataset  # noqa: E402


DEFAULT_MODEL = "yolov8n-seg.pt"
DEFAULT_NAME = "eggvision_crack_seg"


def resolve_device(requested: str) -> str | int:
    if requested.lower() != "auto":
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_project(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_resume_path(resume: str, project: Path, name: str) -> Path:
    if resume == "auto":
        return project / name / "weights" / "last.pt"
    candidate = Path(resume).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def validate_run_destination(project: Path, name: str, resume: str | None) -> list[str]:
    if resume:
        return []
    run_dir = project / name
    if run_dir.exists() and any(run_dir.iterdir()):
        return [
            f"A execucao {run_dir} ja possui arquivos. Escolha outro --name "
            "ou use --resume para continuar last.pt."
        ]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina YOLOv8 Segmentation para a classe unica rachadura."
    )
    parser.add_argument("--data", type=Path, default=SEG_DATA_YAML)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="LAST_PT",
        help="Retoma LAST_PT; sem valor usa <project>/<name>/weights/last.pt.",
    )
    return parser


def print_configuration(args, project: Path, device: str | int) -> None:
    print("Configuracao final de segmentacao:")
    print(f"  data:       {args.data.resolve()}")
    print(f"  model:      {args.model}")
    print(f"  epochs:     {args.epochs}")
    print(f"  patience:   {args.patience}")
    print(f"  imgsz:      {args.imgsz}")
    print(f"  batch:      {args.batch}")
    print(f"  device:     {device}")
    print(f"  workers:    {args.workers}")
    print(f"  seed:       {args.seed}")
    print(f"  output:     {project / args.name}")
    print("  augment:    degrees=5, translate=0.05, scale=0.10, hsv leve")
    print("  mosaic/mixup: 0.0")


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.patience < 0 or args.imgsz < 32 or args.batch < 1:
        print("ERRO: epochs/imgsz/batch devem ser positivos e patience nao negativo.")
        return 2

    validation_report = validate_prepared_dataset(args.data, report_path=None)
    if not validation_report["valid"]:
        print("ERRO: treinamento bloqueado pela validacao do dataset:")
        for error in validation_report["errors"][:20]:
            print(f"  - {error}")
        remaining = len(validation_report["errors"]) - 20
        if remaining > 0:
            print(f"  ... e mais {remaining} erro(s)")
        preparation_report = SEGMENT_RUNS_DIR / "dataset_preparation_report.json"
        if preparation_report.is_file():
            try:
                preparation = json.loads(preparation_report.read_text(encoding="utf-8"))
                missing = preparation.get("missing_crack_annotations", [])
                if missing:
                    print(f"  - {len(missing)} imagem(ns) rachada(s) ainda sem poligono real")
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        print("Anote as rachaduras, prepare o dataset e valide novamente.")
        return 2

    project = resolve_project(args.project)
    destination_errors = validate_run_destination(project, args.name, args.resume)
    if destination_errors:
        for error in destination_errors:
            print(f"ERRO: {error}")
        return 2

    device = resolve_device(args.device)
    print(f"CUDA disponivel: {torch.cuda.is_available()}")
    print_configuration(args, project, device)
    print("Se ocorrer falta de memoria, tente --batch 2 e depois --batch 1.")
    print("Se ainda for necessario, reduza --imgsz explicitamente e registre a alteracao.")

    previous_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        if args.resume:
            resume_path = resolve_resume_path(args.resume, project, args.name)
            if not resume_path.is_file():
                print(f"ERRO: checkpoint de retomada nao encontrado: {resume_path}")
                return 2
            model = YOLO(str(resume_path))
            if model.task != "segment":
                print(f"ERRO: {resume_path} nao e um modelo de segmentacao.")
                return 2
            print(f"Retomando treinamento de: {resume_path}")
            model.train(resume=True, device=device, workers=args.workers)
        else:
            model = YOLO(args.model)
            if model.task != "segment":
                print(f"ERRO: {args.model} nao e um checkpoint YOLO de segmentacao.")
                return 2
            model.train(
                data=str(args.data.resolve()),
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device,
                workers=args.workers,
                project=str(project),
                name=args.name,
                seed=args.seed,
                deterministic=True,
                plots=True,
                save=True,
                val=True,
                exist_ok=False,
                degrees=5.0,
                translate=0.05,
                scale=0.10,
                hsv_h=0.01,
                hsv_s=0.15,
                hsv_v=0.15,
                mosaic=0.0,
                mixup=0.0,
                flipud=0.0,
                fliplr=0.5,
                erasing=0.0,
                copy_paste=0.0,
            )
    except Exception as error:
        print(f"ERRO durante o treinamento: {error}")
        return 1
    finally:
        os.chdir(previous_cwd)

    run_dir = project / args.name
    print(f"Treinamento concluido. Melhor peso esperado: {run_dir / 'weights' / 'best.pt'}")
    print(f"Ultimo peso esperado: {run_dir / 'weights' / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
