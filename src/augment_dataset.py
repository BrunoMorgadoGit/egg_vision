"""
EggVision - Pré-visualização e geração controlada de data augmentation
======================================================================
Aplica transformações moderadas com OpenCV apenas sobre dataset/train.

Uso:
  python src/augment_dataset.py --preview
  python src/augment_dataset.py --apply
"""

import argparse
import os
import random
import shutil
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PROJECT_CLASSES, get_project_root, safe_mkdir, scan_class_directory


AUGMENTED_CLASSES = ("normal", "rachado")
DEFAULT_SEED = 42
DEFAULT_VARIANTS = 3
DEFAULT_PREVIEW_SAMPLES = 2


@dataclass(frozen=True)
class TransformParams:
    angle: float
    translate_x: float
    translate_y: float
    scale: float
    brightness: float
    contrast: float
    gamma: float
    noise_sigma: float
    noise_seed: int
    blur: bool
    flip: bool


def stable_seed(path: str, variant: int, seed: int) -> int:
    """Seed determinística por arquivo e variante."""
    value = seed + variant * 1009
    for char in path:
        value = (value * 33 + ord(char)) % (2 ** 32)
    return value


def sample_params(path: str, variant: int, seed: int) -> TransformParams:
    """Sorteia transformações moderadas e realistas."""
    rng = random.Random(stable_seed(path, variant, seed))
    return TransformParams(
        angle=rng.uniform(-12.0, 12.0),
        translate_x=rng.uniform(-0.04, 0.04),
        translate_y=rng.uniform(-0.04, 0.04),
        scale=rng.uniform(0.96, 1.06),
        brightness=rng.uniform(-10.0, 10.0),
        contrast=rng.uniform(0.92, 1.10),
        gamma=rng.uniform(0.92, 1.08),
        noise_sigma=rng.uniform(0.0, 3.0),
        noise_seed=rng.randrange(0, 2 ** 32),
        blur=rng.random() < 0.25,
        flip=rng.random() < 0.35,
    )


def adjust_gamma(image, gamma: float):
    """Aplica variação leve de exposição via correção gamma."""
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((value / 255.0) ** inv_gamma) * 255
        for value in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


def apply_transform(image, params: TransformParams):
    """Aplica affine + brilho/contraste + exposição + ruído/blur leves."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(center, params.angle, params.scale)
    matrix[0, 2] += params.translate_x * width
    matrix[1, 2] += params.translate_y * height
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    transformed = cv2.convertScaleAbs(
        transformed,
        alpha=params.contrast,
        beta=params.brightness,
    )
    transformed = adjust_gamma(transformed, params.gamma)

    if params.noise_sigma > 0:
        noise = np.random.default_rng(params.noise_seed).normal(
            0,
            params.noise_sigma,
            transformed.shape,
        )
        transformed = np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if params.blur:
        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)

    if params.flip:
        transformed = cv2.flip(transformed, 1)

    return transformed


def describe_params(params: TransformParams) -> str:
    """Descrição curta rastreável das transformações aplicadas."""
    flags = []
    if params.blur:
        flags.append("blur leve")
    if params.flip:
        flags.append("flip horizontal")
    suffix = f"; {', '.join(flags)}" if flags else ""
    return (
        f"rot={params.angle:+.1f}deg, "
        f"shift=({params.translate_x:+.2%},{params.translate_y:+.2%}), "
        f"zoom={params.scale:.2f}, brilho={params.brightness:+.1f}, "
        f"contraste={params.contrast:.2f}, gamma={params.gamma:.2f}, "
        f"ruido={params.noise_sigma:.1f}{suffix}"
    )


def output_name(source_path: str, variant: int) -> str:
    """Nome rastreável: origem_augXX.ext."""
    stem, ext = os.path.splitext(os.path.basename(source_path))
    return f"{stem}_aug{variant:02d}{ext.lower()}"


def collect_train_images() -> dict[str, list[str]]:
    """Coleta apenas imagens de dataset/train/normal e dataset/train/rachado."""
    root = get_project_root()
    result = {}
    for class_name in AUGMENTED_CLASSES:
        class_path = os.path.join(root, "dataset", "train", class_name)
        result[class_name] = scan_class_directory(class_path)["images"]
    return result


def choose_preview_images(images_by_class: dict[str, list[str]], samples: int) -> dict[str, list[str]]:
    """Seleciona poucas imagens por classe em ordem determinística."""
    selected = {}
    for class_name, images in images_by_class.items():
        selected[class_name] = sorted(images)[:samples]
    return selected


def save_variants(images_by_class: dict[str, list[str]],
                  output_root: str,
                  variants: int,
                  seed: int) -> list[tuple[str, str, str]]:
    """Gera variantes e retorna relatório (origem, destino, transformações)."""
    report = []

    for class_name, images in images_by_class.items():
        output_dir = os.path.join(output_root, "train", class_name)
        safe_mkdir(output_dir)

        for image_path in sorted(images):
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Ignorando imagem ilegível: {image_path}")
                continue

            for variant in range(1, variants + 1):
                params = sample_params(image_path, variant, seed)
                augmented = apply_transform(image, params)
                destination = os.path.join(output_dir, output_name(image_path, variant))

                if not cv2.imwrite(destination, augmented):
                    raise RuntimeError(f"não foi possível salvar {destination}")

                report.append((image_path, destination, describe_params(params)))

    return report


def run_preview(variants: int, samples: int, seed: int) -> list[tuple[str, str, str]]:
    """Gera exemplos de augmentation sem alterar dataset original."""
    root = get_project_root()
    output_root = os.path.join(root, "outputs", "augmentation_preview")
    images_by_class = choose_preview_images(collect_train_images(), samples)

    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    safe_mkdir(output_root)

    return save_variants(images_by_class, output_root, variants, seed)


def run_apply(variants: int, seed: int, overwrite: bool) -> list[tuple[str, str, str]]:
    """Gera imagens físicas em dataset_augmented/, separado do dataset original."""
    root = get_project_root()
    output_root = os.path.join(root, "dataset_augmented")
    train_output = os.path.join(output_root, "train")

    if os.path.exists(output_root):
        if not overwrite:
            print(f"\n❌ Diretório já existe: {output_root}")
            print("   Use --overwrite para recriar o dataset_augmented.")
            sys.exit(1)
        shutil.rmtree(output_root)

    for class_name in PROJECT_CLASSES:
        safe_mkdir(os.path.join(train_output, class_name))

    images_by_class = collect_train_images()
    return save_variants(images_by_class, output_root, variants, seed)


def print_report(report: list[tuple[str, str, str]], title: str) -> None:
    """Mostra relatório de origem, destino e transformação aplicada."""
    root = get_project_root()

    print("=" * 72)
    print(f"   EggVision - {title}")
    print("=" * 72)
    print(f"Variações geradas: {len(report)}")

    for source, destination, description in report:
        print(f"- {os.path.relpath(source, root)}")
        print(f"  -> {os.path.relpath(destination, root)}")
        print(f"  {description}")

    print("\nValidação/teste não são alterados por este script.")


def main():
    parser = argparse.ArgumentParser(description="EggVision - Data augmentation com OpenCV")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Gera exemplos em outputs/augmentation_preview")
    mode.add_argument("--apply", action="store_true", help="Gera dataset_augmented/train separado")
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS, help="Variações por imagem")
    parser.add_argument("--samples", type=int, default=DEFAULT_PREVIEW_SAMPLES, help="Amostras por classe no preview")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed determinística")
    parser.add_argument("--overwrite", action="store_true", help="Recria dataset_augmented no modo --apply")
    args = parser.parse_args()

    if args.variants < 1 or args.variants > 5:
        print("\n❌ Use entre 1 e 5 variações por imagem.")
        sys.exit(1)

    if args.preview:
        report = run_preview(args.variants, args.samples, args.seed)
        print_report(report, "Pré-visualização de Augmentation")
    else:
        report = run_apply(args.variants, args.seed, args.overwrite)
        print_report(report, "Geração Física em dataset_augmented")


if __name__ == "__main__":
    main()
