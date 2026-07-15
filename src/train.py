"""
EggVision - Script de Treinamento
==================================
Treina um modelo YOLOv8 de classificação para identificar ovos
normal, rachado e futuramente sujo.

Uso:
    python src/train.py
"""

import os
import shutil
import sys
from datetime import datetime

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    IMBALANCE_WARNING_RATIO,
    MIN_TRAIN_IMAGES_PER_CLASS,
    MIN_VAL_IMAGES_PER_CLASS,
    PROJECT_CLASSES,
    REQUIRED_SPLITS,
    clear_directory,
    collect_dataset_image_records,
    collect_invalid_files,
    find_cross_split_similar_images,
    get_active_training_classes,
    get_project_root,
    get_split_counts,
)


# =============================================
# CONFIGURAÇÕES DO TREINAMENTO
# =============================================

# Modelo pré-treinado de classificação. Não use modelos de detecção aqui.
MODEL_NAME = "yolov8n-cls.pt"

# Tamanho das imagens de entrada (pixels)
IMGSZ = 320

# Épocas máximas. O early stopping interrompe antes se não houver melhora.
EPOCHS = 80
PATIENCE = 15

# Tamanho do batch (reduza para 8 ou 4 se faltar memória)
BATCH = 8

# Seed para reprodutibilidade
SEED = 42

# Caminhos do projeto
ROOT = get_project_root()
DATASET_PATH = os.path.join(ROOT, "dataset")
PROJECT_PATH = os.path.join(ROOT, "runs", "classify")
RUN_NAME = "eggvision_mvp"
ACTIVE_DATASET_PATH = os.path.join(PROJECT_PATH, "_active_dataset")


def ensure_classification_model() -> None:
    """Impede uso acidental de pesos YOLO de detecção no treino."""
    model_file = os.path.basename(MODEL_NAME)
    if model_file.startswith("yolo") and "-cls" not in model_file:
        print("\n❌ Modelo base incorreto para classificação.")
        print(f"   Configurado: {MODEL_NAME}")
        print("   Use um modelo *-cls.pt, por exemplo: yolov8n-cls.pt")
        sys.exit(1)


def print_counts(counts: dict) -> None:
    """Exibe contagens por split e classe."""
    for split in REQUIRED_SPLITS:
        print(f"\n   📂 {split}/")
        for class_name in PROJECT_CLASSES:
            count = counts.get(split, {}).get(class_name, 0)
            status = "✅" if count > 0 else "⚠️ "
            print(f"      {status} {class_name:10s}: {count:4d}")


def validate_required_folders() -> None:
    """Verifica existência de dataset/train|val|test/normal|rachado|sujo."""
    if not os.path.isdir(DATASET_PATH):
        print(f"\n❌ dataset/ não encontrado: {DATASET_PATH}")
        print("💡 Execute: python src/prepare_dataset.py")
        sys.exit(1)

    missing = []
    for split in REQUIRED_SPLITS:
        split_path = os.path.join(DATASET_PATH, split)
        if not os.path.isdir(split_path):
            missing.append(os.path.join("dataset", split))
            continue

        for class_name in PROJECT_CLASSES:
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                missing.append(os.path.join("dataset", split, class_name))

    if missing:
        print("\n❌ Estrutura do dataset incompleta:")
        for path in missing:
            print(f"   - {path}/")
        print("\n💡 Execute: python src/prepare_dataset.py")
        sys.exit(1)


def validate_dataset_for_training() -> tuple[dict, list[str]]:
    """Executa validações obrigatórias antes do treinamento."""
    print("\n🔍 Verificando dataset antes do treino...")
    validate_required_folders()

    counts = get_split_counts(DATASET_PATH)
    print_counts(counts)

    invalid_files = collect_invalid_files(DATASET_PATH)
    if invalid_files:
        print("\n❌ Arquivos inválidos encontrados no dataset:")
        for filepath, reason in invalid_files[:20]:
            print(f"   - {os.path.relpath(filepath, ROOT)} ({reason})")
        if len(invalid_files) > 20:
            print(f"   ... +{len(invalid_files) - 20} arquivo(s)")
        print("\nCorrija ou remova esses arquivos antes de treinar.")
        sys.exit(1)

    active_classes = get_active_training_classes(counts)

    sujo_total = sum(counts[split].get("sujo", 0) for split in REQUIRED_SPLITS)
    if "sujo" not in active_classes:
        print(
            "\n⚠️  Classe 'sujo' sem dados suficientes para treino. "
            "Ela será ignorada temporariamente."
        )
        if sujo_total == 0:
            print("   Nenhuma imagem real de 'sujo' foi encontrada no dataset preparado.")
        else:
            print("   Há imagens de 'sujo', mas faltam amostras em train, val e/ou test.")
        print("   O modelo resultante ainda NÃO reconhecerá ovos sujos.")

    missing_core = [class_name for class_name in ("normal", "rachado")
                    if class_name not in active_classes]
    if missing_core:
        print("\n❌ Classes obrigatórias sem imagens suficientes em train e val:")
        for class_name in missing_core:
            print(f"   - {class_name}")
        print("   Adicione imagens reais e execute prepare_dataset.py novamente.")
        sys.exit(1)

    if len(active_classes) < 2:
        print("\n❌ Treinamento requer pelo menos duas classes ativas.")
        sys.exit(1)

    for class_name in active_classes:
        train_count = counts["train"][class_name]
        val_count = counts["val"][class_name]
        if train_count < MIN_TRAIN_IMAGES_PER_CLASS:
            print(
                f"\n⚠️  Poucas imagens de treino em '{class_name}': {train_count}. "
                f"Recomendado: >= {MIN_TRAIN_IMAGES_PER_CLASS}."
            )
        if val_count < MIN_VAL_IMAGES_PER_CLASS:
            print(
                f"\n⚠️  Poucas imagens de validação em '{class_name}': {val_count}. "
                f"Recomendado: >= {MIN_VAL_IMAGES_PER_CLASS}."
            )

    train_values = [counts["train"][class_name] for class_name in active_classes]
    if len(train_values) >= 2 and min(train_values) > 0:
        ratio = max(train_values) / min(train_values)
        if ratio > IMBALANCE_WARNING_RATIO:
            print(
                f"\n⚠️  Desequilíbrio alto no treino: {max(train_values)} vs "
                f"{min(train_values)} ({ratio:.1f}x)."
            )
        elif ratio > 2:
            print(f"\n⚠️  Desequilíbrio leve no treino: razão {ratio:.1f}x.")

    records = collect_dataset_image_records(DATASET_PATH)
    exact_pairs, similar_pairs = find_cross_split_similar_images(records)

    if exact_pairs:
        print("\n❌ Possíveis duplicatas exatas entre treino/validação/teste:")
        for left, right in exact_pairs[:10]:
            print(f"   - {os.path.relpath(left['path'], ROOT)}")
            print(f"     {os.path.relpath(right['path'], ROOT)}")
        print("\nRemova o vazamento entre splits antes de treinar.")
        sys.exit(1)

    if similar_pairs:
        print("\n⚠️  Possíveis imagens muito semelhantes entre splits:")
        for left, right, distance in similar_pairs[:10]:
            print(
                f"   - {os.path.relpath(left['path'], ROOT)} ↔ "
                f"{os.path.relpath(right['path'], ROOT)} (distância={distance})"
            )
        print("   Revise manualmente para evitar vazamento de dados.")

    print(f"\n✅ Classes que serão treinadas: {', '.join(active_classes)}")
    return counts, active_classes


def build_active_dataset_view(active_classes: list[str]) -> str:
    """
    Cria uma visão temporária do dataset contendo só classes treináveis.

    Isso permite manter dataset/*/sujo/ vazio na estrutura principal sem fazer
    o Ultralytics interpretar a pasta vazia como classe treinada.
    """
    clear_directory(ACTIVE_DATASET_PATH)

    for split in REQUIRED_SPLITS:
        split_dest = os.path.join(ACTIVE_DATASET_PATH, split)
        os.makedirs(split_dest, exist_ok=True)

        for class_name in active_classes:
            source = os.path.join(DATASET_PATH, split, class_name)
            dest = os.path.join(split_dest, class_name)
            shutil.copytree(source, dest)

    return ACTIVE_DATASET_PATH


def has_training_artifacts(run_path: str) -> bool:
    """Retorna True se a execução já tem artefatos de treino relevantes."""
    artifact_paths = [
        os.path.join(run_path, "weights", "best.pt"),
        os.path.join(run_path, "weights", "last.pt"),
        os.path.join(run_path, "results.csv"),
    ]
    return any(os.path.exists(path) for path in artifact_paths)


def resolve_run_name() -> tuple[str, bool]:
    """Evita sobrescrever silenciosamente uma execução anterior."""
    default_run_path = os.path.join(PROJECT_PATH, RUN_NAME)
    if not os.path.exists(default_run_path):
        return RUN_NAME, False

    if not has_training_artifacts(default_run_path):
        print(
            f"\n⚠️  Execução incompleta encontrada em: {default_run_path}\n"
            "   O diretório será reutilizado porque não há best.pt, last.pt ou results.csv."
        )
        return RUN_NAME, True

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RUN_NAME}_{suffix}"
    print(
        f"\n⚠️  Execução anterior encontrada em: {default_run_path}\n"
        f"   Nova execução será salva como: {run_name}"
    )
    return run_name, False


def treinar():
    """Executa o treinamento do modelo YOLO de classificação."""
    print("=" * 55)
    print("   🥚 EggVision — Treinamento YOLOv8 Classification")
    print("=" * 55)

    ensure_classification_model()
    _, active_classes = validate_dataset_for_training()
    training_data_path = build_active_dataset_view(active_classes)
    run_name, exist_ok = resolve_run_name()

    print(f"\n📦 Modelo base:    {MODEL_NAME}")
    print(f"📐 Tamanho imagem: {IMGSZ}px")
    print(f"🔁 Épocas máximas: {EPOCHS}")
    print(f"⏱️  Early stopping: {PATIENCE} épocas sem melhora")
    print(f"📊 Batch size:     {BATCH}")
    print(f"🎲 Seed:           {SEED}")
    print(f"🏷️  Classes:        {', '.join(active_classes)}")
    print(f"📁 Dataset ativo:  {training_data_path}")
    print(f"💾 Resultados em:  {PROJECT_PATH}/{run_name}/")

    print(f"\n🔄 Carregando modelo de classificação: {MODEL_NAME}")
    modelo = YOLO(MODEL_NAME)

    augmentation_params = {
        "degrees": 15,
        "fliplr": 0.5,
        "translate": 0.08,
        "scale": 0.15,
        "hsv_s": 0.15,
        "hsv_v": 0.15,
        "erasing": 0.05,
    }

    print("\n🚀 Iniciando treinamento...\n")
    resultados = modelo.train(
        data=training_data_path,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        project=PROJECT_PATH,
        name=run_name,
        exist_ok=exist_ok,
        verbose=True,
        seed=SEED,
        deterministic=True,
        workers=0,
        save=True,
        plots=True,
        val=True,
        **augmentation_params,
    )

    best_path = os.path.join(PROJECT_PATH, run_name, "weights", "best.pt")
    last_path = os.path.join(PROJECT_PATH, run_name, "weights", "last.pt")
    confusion_path = os.path.join(PROJECT_PATH, run_name, "confusion_matrix.png")
    results_path = os.path.join(PROJECT_PATH, run_name, "results.csv")

    print("\n" + "=" * 55)
    print("   ✅ Treinamento concluído!")
    print("=" * 55)
    print(f"\n🏆 Melhor modelo: {best_path}")
    print(f"📊 Último modelo: {last_path}")
    print(f"📉 Métricas por época: {results_path}")
    print(f"🧩 Matriz de confusão: {confusion_path}")

    if "sujo" not in active_classes:
        print(
            "\n⚠️  Este modelo foi treinado sem a classe 'sujo'. "
            "Ele ainda não é capaz de reconhecer ovos sujos."
        )

    print("\n💡 Valide com foco em falso normal para rachado:")
    print("   python src/validate_model.py --split val")
    print("\n💡 Para testar uma imagem:")
    print("   python src/predict_image.py caminho/da/imagem.jpg")

    return resultados


if __name__ == "__main__":
    treinar()
