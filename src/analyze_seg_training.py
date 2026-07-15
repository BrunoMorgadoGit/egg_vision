"""Gera um diagnostico auxiliar, nao conclusivo, a partir de results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from segmentation_utils import PROJECT_ROOT, write_json


DEFAULT_RESULTS = Path("runs/segment/eggvision_crack_seg/results.csv")


def _float(row: dict, key: str) -> float | None:
    value = row.get(key)
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def analyze_results(results_path: Path) -> dict:
    with results_path.open(newline="", encoding="utf-8") as file:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    if not rows:
        raise ValueError("results.csv nao possui epocas")

    metric_key = next(
        (
            key
            for key in ("metrics/mAP50-95(M)", "metrics/mAP50(M)")
            if any(_float(row, key) is not None for row in rows)
        ),
        None,
    )
    if metric_key is None:
        raise ValueError("results.csv nao contem metrica de mascara")

    valid_metric_rows = [
        (index, _float(row, metric_key))
        for index, row in enumerate(rows)
        if _float(row, metric_key) is not None
    ]
    best_index, best_metric = max(valid_metric_rows, key=lambda item: item[1])
    final_index = len(rows) - 1
    train_loss = _float(rows[-1], "train/seg_loss")
    val_loss = _float(rows[-1], "val/seg_loss")

    window = rows[-min(5, len(rows)):]
    train_values = [_float(row, "train/seg_loss") for row in window]
    val_values = [_float(row, "val/seg_loss") for row in window]
    usable_trend = all(value is not None for value in train_values + val_values)
    sustained_divergence = bool(
        usable_trend
        and len(window) >= 4
        and train_values[-1] < train_values[0]
        and val_values[-1] > val_values[0]
    )

    return {
        "results_csv": str(results_path.resolve()),
        "best_epoch": int(_float(rows[best_index], "epoch") or best_index + 1),
        "final_epoch": int(_float(rows[final_index], "epoch") or final_index + 1),
        "best_validation_metric": {"name": metric_key, "value": best_metric},
        "epochs_without_metric_improvement": final_index - best_index,
        "final_train_seg_loss": train_loss,
        "final_val_seg_loss": val_loss,
        "final_val_minus_train_seg_loss": (
            val_loss - train_loss if train_loss is not None and val_loss is not None else None
        ),
        "overfitting_indication": sustained_divergence,
        "note": (
            "Diagnostico auxiliar baseado em tendencia de varias epocas; "
            "nao substitui avaliacao no teste nem coleta de mais ovos."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume results.csv da segmentacao.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_path = args.results.resolve()
    if not results_path.is_file():
        print(f"ERRO: results.csv nao encontrado: {results_path}")
        return 2
    try:
        report = analyze_results(results_path)
    except (OSError, ValueError) as error:
        print(f"ERRO: {error}")
        return 1
    output = args.output.resolve() if args.output else results_path.parent / "training_diagnostic.json"
    write_json(output, report)
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
