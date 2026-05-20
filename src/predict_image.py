"""
EggVision - Predição em Imagem
===============================
Classifica uma única imagem de ovo como "normal" ou "defeituoso".

Uso:
    python src/predict_image.py caminho/da/imagem.jpg

Saída:
    - Classe e confiança no terminal
    - Janela com a imagem anotada (pressione qualquer tecla para fechar)
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_trained_model,
    obter_resultado,
    format_confidence,
    draw_prediction_on_frame,
    exibir_imagem,
)


def predizer_imagem(caminho_imagem: str, caminho_modelo: str = None):
    """Realiza a predição em uma imagem individual."""
    print("=" * 55)
    print("   🥚 EggVision — Predição em Imagem")
    print("=" * 55)

    # Validar caminho
    caminho_imagem = os.path.abspath(caminho_imagem)
    if not os.path.isfile(caminho_imagem):
        print(f"\n❌ Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)

    # Carregar modelo
    modelo = load_trained_model(caminho_modelo)

    # Ler imagem
    print(f"\n📷 Lendo imagem: {caminho_imagem}")
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print(f"\n❌ Não foi possível ler a imagem!")
        sys.exit(1)

    # Predição
    print("🔄 Classificando...")
    resultados = modelo(imagem, verbose=False)
    classe_nome, confianca = obter_resultado(resultados[0])

    # Resultado no terminal
    print(f"\n{'─' * 40}")
    print(f"   📋 Resultado da Classificação")
    print(f"{'─' * 40}")
    print(f"   🏷️  Classe:     {classe_nome}")
    print(f"   📊 Confiança:  {format_confidence(confianca)}")
    print(f"{'─' * 40}")

    if classe_nome.lower() == "normal":
        print(f"\n   ✅ Ovo NORMAL detectado!")
    else:
        print(f"\n   ⚠️  Ovo DEFEITUOSO detectado!")

    # Desenhar e exibir
    draw_prediction_on_frame(imagem, classe_nome, confianca)
    exibir_imagem(imagem, "EggVision - Resultado")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Caminho da imagem não fornecido!")
        print("\n💡 Uso: python src/predict_image.py caminho/da/imagem.jpg")
        print("\n📌 Exemplos:")
        print("   python src/predict_image.py dataset/test/normal/ovo_01.jpg")
        print("   python src/predict_image.py dataset/test/defeituoso/ovo_02.jpg")
        sys.exit(1)

    caminho = sys.argv[1]
    modelo_path = sys.argv[2] if len(sys.argv) > 2 else None
    predizer_imagem(caminho, modelo_path)
