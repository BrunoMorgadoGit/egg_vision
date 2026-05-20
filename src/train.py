"""
EggVision - Script de Treinamento
==================================
Treina um modelo YOLO de classificação para identificar
ovos normais ou defeituosos.

Uso:
    python src/train.py

Requisitos:
    - Dataset organizado em dataset/train/ e dataset/val/
    - Cada pasta deve conter subpastas: normal/ e defeituoso/
    - Execute prepare_dataset.py antes deste script
"""

import os
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_project_root, count_images_by_class


# =============================================
# CONFIGURAÇÕES DO TREINAMENTO
# =============================================

# Nome do modelo pré-treinado de classificação YOLO
# Opções YOLO11: yolo11n-cls.pt, yolo11s-cls.pt, yolo11m-cls.pt
# Opções YOLOv8: yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt
# Se o YOLO11 não estiver disponível, troque para yolov8n-cls.pt
MODEL_NAME = "yolo11n-cls.pt"

# Tamanho das imagens de entrada (pixels)
IMGSZ = 224

# Número de épocas de treinamento
EPOCHS = 50

# Tamanho do batch (ajuste conforme sua memória GPU/RAM)
BATCH = 16

# Seed para reprodutibilidade
SEED = 42

# Caminhos do projeto
ROOT = get_project_root()
DATASET_PATH = os.path.join(ROOT, "dataset")
PROJECT_PATH = os.path.join(ROOT, "runs", "classify")
RUN_NAME = "eggvision_mvp"

# Classes esperadas
EXPECTED_CLASSES = ["normal", "defeituoso"]


def verificar_dataset():
    """Verifica se o dataset está correto antes de treinar."""
    print("\n🔍 Verificando dataset...")

    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "val")

    if not os.path.isdir(train_path):
        print(f"\n❌ Pasta train/ não encontrada: {train_path}")
        print("💡 Execute: python src/prepare_dataset.py")
        sys.exit(1)

    # Verificar classes em cada split
    for split_name, split_path in [("train", train_path), ("val", val_path)]:
        if not os.path.isdir(split_path):
            print(f"   ⚠️  {split_name}/ não encontrado (continuando...)")
            continue

        counts = count_images_by_class(split_path)

        for cls in EXPECTED_CLASSES:
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                print(f"   ❌ {split_name}/{cls}/ não encontrado!")
                sys.exit(1)

            cnt = counts.get(cls, 0)
            print(f"   ✅ {split_name}/{cls}: {cnt} imagens")

            if cnt == 0:
                print(f"      ⚠️  Sem imagens! Adicione imagens antes de treinar.")
                sys.exit(1)

    # Verificar test também (informativo)
    test_path = os.path.join(DATASET_PATH, "test")
    if os.path.isdir(test_path):
        counts = count_images_by_class(test_path)
        for cls in EXPECTED_CLASSES:
            cnt = counts.get(cls, 0)
            print(f"   📋 test/{cls}: {cnt} imagens")

    print("   ✅ Dataset verificado!\n")


def treinar():
    """Executa o treinamento do modelo YOLO de classificação."""
    print("=" * 55)
    print("   🥚 EggVision — Treinamento de Classificação")
    print("=" * 55)

    verificar_dataset()

    # Informar configurações
    print(f"📦 Modelo base:    {MODEL_NAME}")
    print(f"📐 Tamanho imagem: {IMGSZ}px")
    print(f"🔁 Épocas:         {EPOCHS}")
    print(f"📊 Batch size:     {BATCH}")
    print(f"🎲 Seed:           {SEED}")
    print(f"📁 Dataset:        {DATASET_PATH}")
    print(f"💾 Resultados em:  {PROJECT_PATH}/{RUN_NAME}/")
    print()

    # Carregar modelo pré-treinado
    print(f"🔄 Carregando modelo: {MODEL_NAME}")
    modelo = YOLO(MODEL_NAME)

    # =============================================
    # Parâmetros de Data Augmentation
    # =============================================
    # Todos os parâmetros abaixo são compatíveis com Ultralytics YOLO.
    # Se algum causar erro na sua versão, comente a linha correspondente.
    augmentation_params = {
        "degrees": 25,          # Rotação aleatória ±25 graus
        "fliplr": 0.5,          # Espelhamento horizontal com 50% de chance
        "translate": 0.1,       # Translação de até 10%
        "scale": 0.2,           # Zoom/escala de até ±20%
        "hsv_s": 0.2,           # Variação de saturação ±20%
        "hsv_v": 0.2,           # Variação de brilho ±20%
        "erasing": 0.1,         # Random erasing com 10% de chance
        # "auto_augment": "randaugment",  # Descomentar se compatível
    }

    # Iniciar treinamento
    print("\n🚀 Iniciando treinamento...\n")
    resultados = modelo.train(
        data=DATASET_PATH,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT_PATH,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        seed=SEED,
        **augmentation_params,
    )

    # Informar onde o modelo foi salvo
    best_path = os.path.join(PROJECT_PATH, RUN_NAME, "weights", "best.pt")
    last_path = os.path.join(PROJECT_PATH, RUN_NAME, "weights", "last.pt")

    print("\n" + "=" * 55)
    print("   ✅ Treinamento concluído!")
    print("=" * 55)
    print(f"\n🏆 Melhor modelo: {best_path}")
    print(f"📊 Último modelo: {last_path}")
    print()
    print("💡 Para testar:")
    print("   python src/predict_image.py caminho/da/imagem.jpg")
    print("   python src/predict_video.py caminho/do/video.mp4")
    print("   python src/predict_webcam.py")

    return resultados


if __name__ == "__main__":
    treinar()
