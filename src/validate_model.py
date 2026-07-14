"""
EggVision - Validação do Modelo
================================
Calcula métricas por classe e matriz de confusão para um modelo treinado.

Uso:
  python src/validate_model.py --split val
  python src/validate_model.py --split test --conf 0.70
"""

import argparse
import csv
import os
import sys

import cv2
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    PROJECT_CLASSES,
    UNCERTAIN_CLASS,
    format_confidence,
    get_model_class_names,
    get_project_root,
    interpretar_resultado,
    load_trained_model,
    safe_mkdir,
    scan_class_directory,
)


ROOT = get_project_root()
DATASET_PATH = os.path.join(ROOT, "dataset")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT, "runs", "classify", "eggvision_mvp")


def collect_split_images(split: str, class_names: list[str]) -> list[tuple[str, str]]:
    """Coleta imagens do split para as classes que existem no modelo."""
    samples = []
    split_path = os.path.join(DATASET_PATH, split)

    for class_name in class_names:
        class_path = os.path.join(split_path, class_name)
        scan = scan_class_directory(class_path)
        for filepath in scan["images"]:
            samples.append((filepath, class_name))

    return samples


def write_confusion_csv(path: str, labels: list[str], matrix) -> None:
    """Salva matriz de confusão em CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["real/predito", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def validate_model(model_path: str | None,
                   split: str,
                   confidence_threshold: float) -> None:
    """Valida o modelo no split escolhido."""
    print("=" * 60)
    print("   🥚 EggVision — Validação do Modelo")
    print("=" * 60)

    model = load_trained_model(model_path)
    model_classes = [name for name in get_model_class_names(model) if name in PROJECT_CLASSES]

    if not model_classes:
        print("\n❌ O modelo carregado não possui classes do EggVision.")
        print("   Esperado: normal, rachado e futuramente sujo.")
        sys.exit(1)

    if "sujo" not in model_classes:
        print(
            "\n⚠️  Este modelo não possui a classe 'sujo'. "
            "Ele ainda não é capaz de reconhecer ovos sujos."
        )

    split_path = os.path.join(DATASET_PATH, split)
    if not os.path.isdir(split_path):
        print(f"\n❌ Split não encontrado: {split_path}")
        sys.exit(1)

    samples = collect_split_images(split, model_classes)
    if not samples:
        print(f"\n❌ Nenhuma imagem válida encontrada em dataset/{split}/ para {model_classes}.")
        sys.exit(1)

    print(f"\n📁 Split: {split}")
    print(f"🏷️  Classes avaliadas: {', '.join(model_classes)}")
    print(f"📊 Amostras: {len(samples)}")
    print(f"🎚️  Confiança mínima para aceitar predição: {format_confidence(confidence_threshold)}")
    print("\n🔄 Executando inferência...")

    y_true = []
    y_pred = []
    uncertain_count = 0
    per_image_rows = []

    for filepath, true_class in samples:
        image = cv2.imread(filepath)
        if image is None:
            print(f"   ⚠️  Ignorando imagem ilegível: {os.path.relpath(filepath, ROOT)}")
            continue

        result = model(image, verbose=False)[0]
        prediction = interpretar_resultado(result, confidence_threshold)

        predicted_class = prediction["classe_prevista"]
        final_result = prediction["resultado"]
        confidence = prediction["confianca"]

        if final_result == UNCERTAIN_CLASS:
            uncertain_count += 1

        y_true.append(true_class)
        y_pred.append(predicted_class)
        per_image_rows.append({
            "arquivo": os.path.relpath(filepath, ROOT),
            "real": true_class,
            "predito": predicted_class,
            "resultado_final": final_result,
            "confianca": f"{confidence:.6f}",
        })

    if not y_true:
        print("\n❌ Nenhuma imagem pôde ser avaliada.")
        sys.exit(1)

    labels = model_classes
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"validation_{split}")
    safe_mkdir(output_dir)
    report_path = os.path.join(output_dir, "metrics.txt")
    matrix_path = os.path.join(output_dir, "confusion_matrix.csv")
    per_image_path = os.path.join(output_dir, "predictions.csv")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write(f"\n\nincertos_por_confianca: {uncertain_count}\n")
        file.write(f"limite_confianca: {confidence_threshold:.4f}\n")

    write_confusion_csv(matrix_path, labels, matrix)

    with open(per_image_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["arquivo", "real", "predito", "resultado_final", "confianca"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print("\n📋 Métricas por classe:")
    print(report)

    print("🧩 Matriz de confusão (linhas=real, colunas=predito):")
    header = "real\\pred".ljust(12) + "".join(label[:10].rjust(12) for label in labels)
    print(header)
    for label, row in zip(labels, matrix):
        print(label[:10].ljust(12) + "".join(str(value).rjust(12) for value in row))

    print(f"\n🎚️  Predições marcadas como '{UNCERTAIN_CLASS}': {uncertain_count}")

    if "rachado" in labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["rachado"],
            zero_division=0,
        )
        false_rachado_as_normal = sum(
            1 for true_class, pred_class in zip(y_true, y_pred)
            if true_class == "rachado" and pred_class == "normal"
        )
        total_rachado = sum(1 for true_class in y_true if true_class == "rachado")

        print("\n🚨 Foco crítico: classe 'rachado'")
        print(f"   Precision: {precision[0]:.4f}")
        print(f"   Recall:    {recall[0]:.4f}")
        print(f"   F1-score:  {f1[0]:.4f}")
        print(
            f"   Rachados classificados como normal: "
            f"{false_rachado_as_normal}/{total_rachado}"
        )

        if false_rachado_as_normal > 0:
            print("   ⚠️  Prioridade: reduzir falso normal para ovos rachados.")

    print("\n💾 Arquivos salvos:")
    print(f"   - {report_path}")
    print(f"   - {matrix_path}")
    print(f"   - {per_image_path}")


def main():
    parser = argparse.ArgumentParser(description="EggVision - Validação do Modelo")
    parser.add_argument("--model", default=None, help="Caminho para best.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Split avaliado")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confiança mínima para marcar resultado como aceito",
    )
    args = parser.parse_args()
    validate_model(args.model, args.split, args.conf)


if __name__ == "__main__":
    main()
