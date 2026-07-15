"""Ferramenta OpenCV para anotar poligonos reais de rachaduras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_utils import (  # noqa: E402
    ANNOTATIONS_DIR,
    CRACK_CLASS_ID,
    IGNORED_FILENAMES,
    RAW_DATASET_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    inspect_image,
    validate_segmentation_label,
)


WINDOW_NAME = "EggVision - Anotacao de rachaduras"
SHORTCUTS = """
Atalhos:
  clique esquerdo  adicionar ponto
  Enter             finalizar poligono atual (minimo 3 pontos)
  U                 desfazer ultimo ponto
  R                 limpar poligono atual
  D                 excluir ultimo poligono concluido
  S                 salvar anotacao
  N                 proxima imagem
  B                 imagem anterior
  Q ou Esc          sair
""".strip()


def list_cracked_images(raw_cracked_dir: Path) -> list[Path]:
    if not raw_cracked_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in raw_cracked_dir.rglob("*")
            if path.is_file()
            and path.name.lower() not in IGNORED_FILENAMES
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda item: item.as_posix().lower(),
    )


def annotation_path_for_image(
    image_path: Path, raw_cracked_dir: Path, annotation_dir: Path
) -> Path:
    relative = image_path.relative_to(raw_cracked_dir)
    return (annotation_dir / relative).with_suffix(".txt")


def normalized_to_pixels(
    polygon: list[tuple[float, float]], width: int, height: int
) -> np.ndarray:
    points = [
        (
            min(width - 1, max(0, round(x * width))),
            min(height - 1, max(0, round(y * height))),
        )
        for x, y in polygon
    ]
    return np.asarray(points, dtype=np.int32)


def draw_polygons(
    image: np.ndarray,
    polygons: list[list[tuple[float, float]]],
    current_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    rendered = image.copy()
    overlay = image.copy()
    height, width = image.shape[:2]

    for polygon in polygons:
        points = normalized_to_pixels(polygon, width, height)
        if len(points) >= 3:
            cv2.fillPoly(overlay, [points], (0, 0, 255))
            cv2.polylines(rendered, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
    rendered = cv2.addWeighted(overlay, 0.25, rendered, 0.75, 0)

    if current_points:
        points = normalized_to_pixels(current_points, width, height)
        for point in points:
            cv2.circle(rendered, tuple(point), 4, (0, 255, 0), -1, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.polylines(rendered, [points], False, (0, 255, 0), 2, cv2.LINE_AA)
    return rendered


def write_annotation(path: Path, polygons: list[list[tuple[float, float]]]) -> None:
    lines = []
    for polygon in polygons:
        coordinates = " ".join(f"{value:.6f}" for point in polygon for value in point)
        lines.append(f"{CRACK_CLASS_ID} {coordinates}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_previews(
    raw_cracked_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    images = list_cracked_images(raw_cracked_dir)
    generated = 0
    skipped = 0
    for image_path in images:
        label_path = annotation_path_for_image(image_path, raw_cracked_dir, annotation_dir)
        if not label_path.is_file():
            skipped += 1
            continue
        polygons, errors = validate_segmentation_label(label_path, require_non_empty=True)
        valid, _ = inspect_image(image_path)
        if errors or not valid:
            print(f"Ignorada: {image_path}")
            for error in errors:
                print(f"  {error}")
            skipped += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rendered = draw_polygons(image, polygons)
        relative = image_path.relative_to(raw_cracked_dir)
        output_name = "__".join(relative.with_suffix("").parts) + "_mask_preview.jpg"
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            print(f"Falha ao salvar: {output_path}")
            skipped += 1
            continue
        print(f"Preview: {output_path}")
        generated += 1
    return generated, skipped


class AnnotationApp:
    def __init__(
        self,
        raw_cracked_dir: Path,
        annotation_dir: Path,
        *,
        start_image: str | None = None,
        max_width: int = 1400,
        max_height: int = 850,
    ) -> None:
        self.raw_cracked_dir = raw_cracked_dir.resolve()
        self.annotation_dir = annotation_dir.resolve()
        self.images = list_cracked_images(self.raw_cracked_dir)
        self.index = 0
        self.max_width = max_width
        self.max_height = max_height
        self.image: np.ndarray | None = None
        self.polygons: list[list[tuple[float, float]]] = []
        self.current_points: list[tuple[float, float]] = []
        self.display_scale = 1.0
        self.dirty = False

        if start_image:
            for index, path in enumerate(self.images):
                if path.name == start_image or path.as_posix().endswith(start_image):
                    self.index = index
                    break

    @property
    def current_image_path(self) -> Path:
        return self.images[self.index]

    @property
    def current_label_path(self) -> Path:
        return annotation_path_for_image(
            self.current_image_path, self.raw_cracked_dir, self.annotation_dir
        )

    def load_current(self) -> None:
        image = cv2.imread(str(self.current_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV nao conseguiu abrir {self.current_image_path}")
        self.image = image
        self.polygons = []
        self.current_points = []
        self.dirty = False
        if self.current_label_path.is_file():
            polygons, errors = validate_segmentation_label(
                self.current_label_path, require_non_empty=True
            )
            self.polygons = polygons
            for error in errors:
                print(f"AVISO: {error}")

    def mouse_callback(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.image is None:
            return
        height, width = self.image.shape[:2]
        original_x = x / self.display_scale
        original_y = y / self.display_scale
        normalized_x = min(1.0, max(0.0, original_x / width))
        normalized_y = min(1.0, max(0.0, original_y / height))
        self.current_points.append((normalized_x, normalized_y))
        self.dirty = True

    def render(self) -> np.ndarray:
        if self.image is None:
            raise RuntimeError("Nenhuma imagem carregada")
        rendered = draw_polygons(self.image, self.polygons, self.current_points)
        total = len(self.images)
        state = "ALTERADO" if self.dirty else "salvo/sem alteracao"
        header = (
            f"{self.index + 1}/{total}  {self.current_image_path.name}  "
            f"poligonos={len(self.polygons)}  {state}"
        )
        cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(
            rendered,
            header,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        height, width = rendered.shape[:2]
        self.display_scale = min(1.0, self.max_width / width, self.max_height / height)
        if self.display_scale < 1.0:
            rendered = cv2.resize(
                rendered,
                (round(width * self.display_scale), round(height * self.display_scale)),
                interpolation=cv2.INTER_AREA,
            )
        return rendered

    def finalize_polygon(self) -> None:
        if len(self.current_points) < 3:
            print("O poligono precisa de pelo menos tres pontos.")
            return
        self.polygons.append(list(self.current_points))
        self.current_points.clear()
        self.dirty = True

    def save(self) -> None:
        if self.current_points:
            print("Finalize o poligono atual com Enter antes de salvar.")
            return
        if not self.polygons:
            print("Imagem rachada exige pelo menos um poligono; nada foi salvo.")
            return
        write_annotation(self.current_label_path, self.polygons)
        self.dirty = False
        print(f"Salvo: {self.current_label_path}")

    def move(self, step: int) -> None:
        if self.dirty:
            print("Ha alteracoes nao salvas. Use S antes de trocar de imagem.")
            return
        next_index = self.index + step
        if not 0 <= next_index < len(self.images):
            print("Inicio/fim da lista de imagens.")
            return
        self.index = next_index
        self.load_current()

    def run(self) -> int:
        if not self.images:
            print(f"Nenhuma imagem rachada encontrada em {self.raw_cracked_dir}")
            return 1
        print(SHORTCUTS)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.load_current()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (255,):
                continue
            if key in (ord("q"), 27):
                if self.dirty:
                    print("Saindo com alteracoes atuais nao salvas; labels salvas foram preservadas.")
                break
            if key in (10, 13):
                self.finalize_polygon()
            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()
                    self.dirty = True
            elif key == ord("r"):
                self.current_points.clear()
                self.dirty = True
            elif key == ord("d"):
                if self.polygons:
                    self.polygons.pop()
                    self.dirty = True
            elif key == ord("s"):
                self.save()
            elif key == ord("n"):
                self.move(1)
            elif key == ord("b"):
                self.move(-1)

        cv2.destroyAllWindows()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anota poligonos de rachadura e salva em YOLO Segmentation."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_DATASET_DIR / "rachado",
        help="Pasta recursiva com imagens rachadas.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATIONS_DIR / "rachado",
        help="Pasta de labels que espelha --raw.",
    )
    parser.add_argument("--start", help="Nome/caminho relativo da imagem inicial.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Salva overlays das labels existentes sem abrir a interface.",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=Path("outputs/segmentation_annotation_preview"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.preview:
        generated, skipped = create_previews(
            args.raw.resolve(), args.annotations.resolve(), args.preview_output.resolve()
        )
        print(f"Previews gerados: {generated}; sem label/invalidos: {skipped}")
        return 0 if generated > 0 else 1
    return AnnotationApp(
        args.raw,
        args.annotations,
        start_image=args.start,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
