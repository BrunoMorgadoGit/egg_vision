"""
EggVision - Predição em Webcam
===============================
Classifica ovos em tempo real usando a webcam.

Uso:
    python src/predict_webcam.py

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


def predizer_webcam(caminho_modelo: str = None):
    """Realiza a predição em tempo real usando a webcam."""
    print("=" * 55)
    print("   🥚 EggVision — Predição em Webcam")
    print("=" * 55)

    # Carregar modelo
    modelo = load_trained_model(caminho_modelo)

    # Abrir webcam
    print("\n📷 Abrindo webcam (dispositivo 0)...")
    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        print("\n❌ Não foi possível acessar a webcam!")
        print("💡 Verifique se a webcam está conectada e disponível.")
        sys.exit(1)

    print("✅ Webcam aberta!")
    print("Pressione 'q' para sair.\n")

    frame_count = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            print("\n❌ Falha ao capturar frame!")
            break

        frame_count += 1

        # Predição
        resultados = modelo(frame, verbose=False)
        classe_nome, confianca = obter_resultado(resultados[0])

        # Desenhar resultado
        draw_prediction_on_frame(frame, classe_nome, confianca)

        # Indicador LIVE
        cv2.circle(frame, (frame.shape[1] - 90, 25), 5, (0, 0, 255), -1)
        cv2.putText(frame, "LIVE", (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("EggVision - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹️  Webcam encerrada pelo usuário.")
            break

    captura.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Frames processados: {frame_count}")


if __name__ == "__main__":
    modelo_path = sys.argv[1] if len(sys.argv) > 1 else None
    predizer_webcam(modelo_path)
