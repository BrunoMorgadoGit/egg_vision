"""
EggVision - Predição em Imagem
===============================
Classifica uma única imagem de ovo como normal, rachado ou incerto.

Uso:
    python src/predict_image.py caminho/da/imagem.jpg
    python src/predict_image.py caminho/da/imagem.jpg --conf 0.70 --no-show
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    PROJECT_CLASSES,
    UNCERTAIN_CLASS,
    draw_prediction_on_frame,
    exibir_imagem,
    format_confidence,
    get_model_class_names,
    get_project_root,
    interpretar_resultado,
    load_trained_model,
    safe_mkdir,
)


def default_output_path(image_path: str) -> str:
    """Define caminho padrão para a imagem processada."""
    root = get_project_root()
    output_dir = os.path.join(root, "outputs", "predictions")
    safe_mkdir(output_dir)

    stem, _ = os.path.splitext(os.path.basename(image_path))
    return os.path.join(output_dir, f"{stem}_pred.jpg")


def validate_model_classes(model) -> list[str]:
    """Garante que o best.pt carregado é compatível com o EggVision."""
    model_classes = get_model_class_names(model)
    eggvision_classes = [name for name in model_classes if name in PROJECT_CLASSES]

    if not eggvision_classes:
        print("\n❌ O modelo carregado não parece ser um classificador EggVision treinado.")
        print("   Esperado: classes normal, rachado e futuramente sujo.")
        print("   Treine primeiro com: python src/train.py")
        sys.exit(1)

    if "sujo" not in eggvision_classes:
        print(
            "\n⚠️  Este modelo não possui a classe 'sujo'. "
            "Ele ainda não é capaz de reconhecer ovos sujos."
        )

    return eggvision_classes


def predizer_imagem(caminho_imagem: str,
                    caminho_modelo: str | None = None,
                    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                    output_path: str | None = None,
                    show_image: bool = True):
    """Realiza a predição em uma imagem individual."""
    print("=" * 55)
    print("   🥚 EggVision — Predição em Imagem")
    print("=" * 55)

    caminho_imagem = os.path.abspath(caminho_imagem)
    if not os.path.isfile(caminho_imagem):
        print(f"\n❌ Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)

    modelo = load_trained_model(caminho_modelo)
    validate_model_classes(modelo)

    print(f"\n📷 Lendo imagem: {caminho_imagem}")
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print("\n❌ Não foi possível ler a imagem!")
        sys.exit(1)

    print("🔄 Classificando...")
    try:
        resultado = modelo(imagem, verbose=False)[0]
        prediction = interpretar_resultado(resultado, confidence_threshold)
    except ValueError as error:
        print(f"\n❌ {error}")
        sys.exit(1)

    classe_prevista = prediction["classe_prevista"]
    confianca = prediction["confianca"]
    resultado_final = prediction["resultado"]

    print(f"\n{'─' * 45}")
    print("   📋 Resultado da Classificação")
    print(f"{'─' * 45}")
    print(f"   🏷️  Classe prevista: {classe_prevista}")
    print(f"   📊 Confiança:        {format_confidence(confianca)}")
    print(f"   🎚️  Conf. mínima:    {format_confidence(confidence_threshold)}")
    print(f"   ✅ Resultado final:  {resultado_final}")
    print(f"{'─' * 45}")

    if resultado_final == UNCERTAIN_CLASS:
        print("\n   ⚠️  Resultado INCERTO: confiança abaixo do limite mínimo.")
    elif resultado_final == "normal":
        print("\n   ✅ Ovo NORMAL detectado.")
    elif resultado_final == "rachado":
        print("\n   🚨 Ovo RACHADO detectado.")
    elif resultado_final == "sujo":
        print("\n   ⚠️  Ovo SUJO detectado.")

    draw_prediction_on_frame(imagem, resultado_final, confianca)

    if output_path is None:
        output_path = default_output_path(caminho_imagem)
    else:
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            safe_mkdir(output_dir)

    if not cv2.imwrite(output_path, imagem):
        print(f"\n❌ Não foi possível salvar a imagem processada: {output_path}")
        sys.exit(1)

    print(f"\n💾 Imagem processada salva em: {output_path}")

    if show_image:
        exibir_imagem(imagem, "EggVision - Resultado")

    return prediction


def main():
    parser = argparse.ArgumentParser(description="EggVision - Predição em Imagem")
    parser.add_argument("imagem", help="Caminho da imagem")
    parser.add_argument("modelo", nargs="?", default=None, help="Caminho opcional para best.pt")
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confiança mínima para aceitar a classe prevista",
    )
    parser.add_argument("--output", default=None, help="Caminho para salvar a imagem processada")
    parser.add_argument("--no-show", action="store_true", help="Não abrir janela OpenCV")
    args = parser.parse_args()

    predizer_imagem(
        args.imagem,
        args.modelo,
        confidence_threshold=args.conf,
        output_path=args.output,
        show_image=not args.no_show,
    )


if __name__ == "__main__":
    main()
