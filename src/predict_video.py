"""
EggVision - Predição em Vídeo
==============================
Classifica ovos frame por frame em um vídeo local.

Uso:
    python src/predict_video.py caminho/do/video.mp4

Formatos aceitos: .mp4, .avi, .mkv
Controles: Pressione 'q' para sair
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_trained_model,
    obter_resultado,
    draw_prediction_on_frame,
)

VALID_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")


def predizer_video(caminho_video: str, caminho_modelo: str = None):
    """Realiza a predição em um vídeo local, frame por frame."""
    print("=" * 55)
    print("   🥚 EggVision — Predição em Vídeo")
    print("=" * 55)

    # Validar caminho
    caminho_video = os.path.abspath(caminho_video)
    if not os.path.isfile(caminho_video):
        print(f"\n❌ Vídeo não encontrado: {caminho_video}")
        sys.exit(1)

    _, ext = os.path.splitext(caminho_video)
    if ext.lower() not in VALID_VIDEO_EXTENSIONS:
        print(f"\n⚠️  Extensão '{ext}' não testada. Aceitos: {VALID_VIDEO_EXTENSIONS}")

    # Carregar modelo
    modelo = load_trained_model(caminho_modelo)

    # Abrir vídeo
    print(f"\n🎬 Abrindo vídeo: {caminho_video}")
    captura = cv2.VideoCapture(caminho_video)

    if not captura.isOpened():
        print("\n❌ Não foi possível abrir o vídeo!")
        sys.exit(1)

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   Resolução: {largura}x{altura}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total de frames: {total_frames}")
    print(f"\n▶️  Reproduzindo... Pressione 'q' para sair.\n")

    frame_count = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            print("\n🏁 Fim do vídeo!")
            break

        frame_count += 1

        # Predição
        resultados = modelo(frame, verbose=False)
        classe_nome, confianca = obter_resultado(resultados[0])
        draw_prediction_on_frame(frame, classe_nome, confianca)

        # Contador de frames
        texto_frame = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, texto_frame, (20, altura - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("EggVision - Video", frame)

        delay = int(1000 / fps) if fps > 0 else 30
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("\n⏹️  Vídeo interrompido pelo usuário.")
            break

    captura.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Frames processados: {frame_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Caminho do vídeo não fornecido!")
        print("\n💡 Uso: python src/predict_video.py caminho/do/video.mp4")
        print("   Formatos aceitos: .mp4, .avi, .mkv")
        sys.exit(1)

    caminho = sys.argv[1]
    modelo_path = sys.argv[2] if len(sys.argv) > 2 else None
    predizer_video(caminho, modelo_path)
