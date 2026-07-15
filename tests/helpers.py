from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VALID_POLYGON = "0 0.200000 0.200000 0.800000 0.200000 0.500000 0.800000\n"


def write_image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    cv2.circle(image, (10 + (value % 20), 15), 5, (255 - value, value, 100), -1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Falha ao criar imagem de teste: {path}")
    return path


def write_label(path: Path, content: str = VALID_POLYGON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
