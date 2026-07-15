"""Inferencia de segmentacao: desenha somente mascaras/contornos de rachadura."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_utils import (  # noqa: E402
    CRACK_CLASS_ID,
    CRACK_CLASS_NAME,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_EXTENSIONS,
    write_json,
)
from train_seg import resolve_device  # noqa: E402


DEFAULT_MODEL = Path("runs/segment/eggvision_crack_seg/weights/best.pt")
DEFAULT_OUTPUT = Path("outputs/segmentation_predictions")


def as_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def resize_binary_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"Mascara deve ser 2D; shape recebido: {mask.shape}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def summarize_mask_predictions(
    *,
    image_name: str,
    image_shape: tuple[int, int],
    threshold: float,
    confidences=None,
    class_ids=None,
    masks=None,
) -> tuple[dict, list[np.ndarray]]:
    """Interpreta arrays/mocks sem depender de GPU ou de um modelo real."""
    confidence_values = as_numpy(confidences).reshape(-1)
    class_values = as_numpy(class_ids).reshape(-1)
    mask_values = as_numpy(masks)
    if mask_values.size == 0:
        mask_values = np.empty((0, *image_shape), dtype=np.float32)
    elif mask_values.ndim == 2:
        mask_values = mask_values[np.newaxis, ...]

    detections = []
    accepted_masks: list[np.ndarray] = []
    total_pixels = int(image_shape[0] * image_shape[1])
    item_count = min(len(confidence_values), len(class_values), len(mask_values))

    for index in range(item_count):
        confidence = float(confidence_values[index])
        class_id = int(class_values[index])
        if class_id != CRACK_CLASS_ID or confidence < threshold:
            continue
        binary_mask = resize_binary_mask(mask_values[index], image_shape)
        area_pixels = int(np.count_nonzero(binary_mask))
        detections.append(
            {
                "class_id": CRACK_CLASS_ID,
                "class_name": CRACK_CLASS_NAME,
                "confidence": confidence,
                "mask_area_pixels": area_pixels,
                "mask_area_ratio": area_pixels / total_pixels if total_pixels else 0.0,
            }
        )
        accepted_masks.append(binary_mask)

    maximum = max((item["confidence"] for item in detections), default=None)
    summary = {
        "image": image_name,
        "status": "RACHADO" if detections else "NORMAL",
        "threshold": float(threshold),
        "crack_count": len(detections),
        "max_confidence": maximum,
        "detections": detections,
    }
    return summary, accepted_masks


def prediction_from_result(
    result,
    *,
    image_name: str,
    image_shape: tuple[int, int],
    threshold: float,
) -> tuple[dict, list[np.ndarray]]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None:
        return summarize_mask_predictions(
            image_name=image_name,
            image_shape=image_shape,
            threshold=threshold,
        )
    return summarize_mask_predictions(
        image_name=image_name,
        image_shape=image_shape,
        threshold=threshold,
        confidences=getattr(boxes, "conf", None),
        class_ids=getattr(boxes, "cls", None),
        masks=getattr(masks, "data", None),
    )


def render_prediction(
    image: np.ndarray,
    summary: dict,
    masks: list[np.ndarray],
    *,
    show_text: bool = True,
) -> np.ndarray:
    rendered = image.copy()
    overlay = image.copy()
    for mask in masks:
        overlay[mask] = (0, 0, 255)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rendered, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    if masks:
        rendered = cv2.addWeighted(overlay, 0.30, rendered, 0.70, 0)

    if show_text:
        status = summary["status"]
        confidence = summary["max_confidence"]
        label = status if confidence is None else f"{status} {confidence:.1%}"
        color = (0, 0, 255) if status == "RACHADO" else (0, 180, 0)
        cv2.rectangle(rendered, (12, 12), (350, 58), (0, 0, 0), -1)
        cv2.putText(
            rendered,
            label,
            (22, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def save_prediction_json(path: Path, summary: dict) -> Path:
    return write_json(path, summary)


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda item: item.as_posix().lower(),
        )
    return []


def output_stem(image_path: Path, source: Path) -> str:
    if source.is_file():
        return image_path.stem
    relative = image_path.relative_to(source)
    return "__".join(relative.with_suffix("").parts)


def process_image(
    model,
    image_path: Path,
    *,
    source_root: Path,
    output_dir: Path,
    threshold: float,
    imgsz: int,
    device: str | int,
    save_json: bool,
    show: bool,
    show_text: bool,
) -> tuple[dict | None, list[Path]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        print(f"ERRO: imagem invalida: {image_path}")
        return None, []

    results = model.predict(
        source=image,
        conf=threshold,
        imgsz=imgsz,
        device=device,
        classes=[CRACK_CLASS_ID],
        retina_masks=True,
        verbose=False,
    )
    result = results[0] if results else None
    summary, masks = prediction_from_result(
        result,
        image_name=image_path.name,
        image_shape=image.shape[:2],
        threshold=threshold,
    )
    rendered = render_prediction(image, summary, masks, show_text=show_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(image_path, source_root)
    image_output = output_dir / f"{stem}_seg.jpg"
    if not cv2.imwrite(str(image_output), rendered):
        raise RuntimeError(f"Nao foi possivel salvar {image_output}")
    generated = [image_output]

    if save_json:
        json_output = output_dir / f"{stem}_seg.json"
        save_prediction_json(json_output, summary)
        generated.append(json_output)

    if show:
        cv2.imshow("EggVision - Segmentacao", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(
        f"{image_path.name}: {summary['status']} | rachaduras={summary['crack_count']} "
        f"| max_conf={summary['max_confidence']}"
    )
    return summary, generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Localiza rachaduras com YOLO Segmentation em imagem ou pasta."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", required=True, help="Imagem ou pasta de imagens.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.conf <= 1.0:
        print("ERRO: --conf deve estar entre 0 e 1.")
        return 2
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"ERRO: modelo nao encontrado: {model_path}")
        return 2
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"ERRO: fonte nao encontrada: {source}")
        return 2
    images = collect_images(source)
    if not images:
        print(f"ERRO: nenhuma imagem suportada encontrada em {source}")
        return 2

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "segment":
        print("ERRO: checkpoint informado nao e de segmentacao.")
        return 2
    raw_names = model.names
    names = (
        {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: str(value) for index, value in enumerate(raw_names)}
    )
    if names.get(CRACK_CLASS_ID) != CRACK_CLASS_NAME:
        print(
            f"ERRO: classe 0 deve ser '{CRACK_CLASS_NAME}', encontrada "
            f"'{names.get(CRACK_CLASS_ID)}'."
        )
        return 2

    device = resolve_device(args.device)
    output_dir = args.output.resolve()
    summaries = []
    generated_paths: list[Path] = []
    for image_path in images:
        try:
            summary, generated = process_image(
                model,
                image_path,
                source_root=source,
                output_dir=output_dir,
                threshold=args.conf,
                imgsz=args.imgsz,
                device=device,
                save_json=args.save_json,
                show=args.show,
                show_text=not args.no_text,
            )
        except (RuntimeError, ValueError) as error:
            print(f"ERRO em {image_path}: {error}")
            continue
        if summary is not None:
            summaries.append(summary)
            generated_paths.extend(generated)

    if not summaries:
        print("ERRO: nenhuma imagem foi processada com sucesso.")
        return 1
    if args.save_json and source.is_dir():
        aggregate = output_dir / "predictions.json"
        write_json(aggregate, summaries)
        generated_paths.append(aggregate)
    print(f"Arquivos gerados: {len(generated_paths)} em {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
