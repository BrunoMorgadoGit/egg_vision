"""
EggVision - Módulo de Utilitários
==================================
Funções reutilizáveis para o projeto EggVision.

Inclui:
  - get_project_root()         → raiz do projeto
  - count_images_by_class()    → contagem de imagens por classe
  - validate_image_file()      → validação de arquivo de imagem
  - load_trained_model()       → carregamento do modelo treinado
  - format_confidence()        → formatação de confiança
  - draw_prediction_on_frame() → desenho de resultado na imagem
  - safe_mkdir()               → criação segura de diretório
  - clear_directory()          → limpeza segura de diretório
"""

import os
import sys
import shutil
import cv2
from ultralytics import YOLO


# =============================================
# Extensões de imagem aceitas pelo projeto
# =============================================
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# =============================================
# Funções de caminho e diretório
# =============================================

def get_project_root() -> str:
    """
    Retorna o caminho absoluto da raiz do projeto (eggvision/).

    Funciona independentemente de onde o script é executado,
    assumindo que este arquivo está em eggvision/src/utils.py.

    Returns:
        Caminho absoluto da raiz do projeto.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_mkdir(path: str) -> None:
    """
    Cria um diretório (e seus pais) de forma segura.
    Não gera erro se o diretório já existir.

    Args:
        path: Caminho do diretório a ser criado.
    """
    os.makedirs(path, exist_ok=True)


def clear_directory(path: str) -> None:
    """
    Remove todo o conteúdo de um diretório, mas mantém o diretório em si.
    Se o diretório não existir, cria-o.

    Args:
        path: Caminho do diretório a ser limpo.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =============================================
# Funções de validação
# =============================================

def validate_image_file(filepath: str) -> bool:
    """
    Valida se um arquivo é uma imagem aceita pelo projeto.

    Verifica:
      1. Se o arquivo existe
      2. Se a extensão está na lista de extensões válidas
      3. Se o arquivo não está vazio (tamanho > 0)

    Args:
        filepath: Caminho do arquivo a ser validado.

    Returns:
        True se o arquivo é uma imagem válida, False caso contrário.
    """
    if not os.path.isfile(filepath):
        return False

    _, ext = os.path.splitext(filepath)
    if ext.lower() not in VALID_IMAGE_EXTENSIONS:
        return False

    # Verificar se o arquivo não está vazio
    if os.path.getsize(filepath) == 0:
        return False

    return True


# =============================================
# Funções de contagem e estatísticas
# =============================================

def count_images_by_class(base_path: str) -> dict:
    """
    Conta o número de imagens válidas em cada subpasta (classe) de um diretório.

    Espera uma estrutura como:
        base_path/
          classe_a/
            img1.jpg
            img2.png
          classe_b/
            img3.jpg

    Args:
        base_path: Caminho do diretório contendo subpastas de classes.

    Returns:
        Dicionário {nome_classe: quantidade_de_imagens}.
        Retorna dicionário vazio se o caminho não existir.
    """
    result = {}

    if not os.path.isdir(base_path):
        return result

    for class_name in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            if validate_image_file(filepath):
                count += 1

        result[class_name] = count

    return result


# =============================================
# Funções de modelo
# =============================================

def load_trained_model(model_path: str = None) -> YOLO:
    """
    Carrega o modelo YOLO treinado para inferência.

    Se nenhum caminho for fornecido, usa o caminho padrão:
        runs/classify/eggvision_mvp/weights/best.pt

    Args:
        model_path: Caminho para o arquivo .pt do modelo.
                    Se None, usa o caminho padrão.

    Returns:
        Instância do modelo YOLO carregado.

    Raises:
        SystemExit: Se o modelo não for encontrado.
    """
    if model_path is None:
        model_path = os.path.join(
            get_project_root(),
            "runs", "classify", "eggvision_mvp", "weights", "best.pt"
        )

    model_path = os.path.abspath(model_path)

    if not os.path.isfile(model_path):
        print(f"\n❌ Erro: modelo treinado não encontrado!")
        print(f"   Caminho: {model_path}")
        print(f"\n💡 Dicas:")
        print(f"   1. Execute primeiro: python src/train.py")
        print(f"   2. Ou forneça o caminho do modelo como argumento.")
        sys.exit(1)

    print(f"\n🔄 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Modelo carregado com sucesso!")

    return model


# =============================================
# Funções de formatação
# =============================================

def format_confidence(confidence: float) -> str:
    """
    Formata o valor de confiança como porcentagem legível.

    Args:
        confidence: Valor float entre 0 e 1.

    Returns:
        String formatada, ex: "95.32%"
    """
    return f"{confidence * 100:.2f}%"


# =============================================
# Funções de visualização (OpenCV)
# =============================================

def draw_prediction_on_frame(frame, class_name: str, confidence: float) -> None:
    """
    Desenha o resultado da classificação sobre o frame/imagem.

    Adiciona textos com a classe prevista e a confiança na parte
    superior da imagem, com fundo semi-transparente para legibilidade.

    Args:
        frame: Imagem OpenCV (numpy array BGR).
        class_name: Nome da classe prevista ("normal" ou "defeituoso").
        confidence: Valor de confiança entre 0 e 1.
    """
    # Textos a serem exibidos
    texto_classe = f"Classe: {class_name}"
    texto_conf = f"Confianca: {format_confidence(confidence)}"

    # Definir cor baseada na classe
    if class_name.lower() == "normal":
        cor = (0, 200, 0)       # Verde para normal
    else:
        cor = (0, 0, 220)       # Vermelho para defeituoso

    # Configurações de fonte
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.9
    espessura = 2

    # Calcular tamanho do texto para o fundo
    (w1, h1), _ = cv2.getTextSize(texto_classe, fonte, escala, espessura)
    (w2, h2), _ = cv2.getTextSize(texto_conf, fonte, escala, espessura)

    # Desenhar fundo semi-transparente
    overlay = frame.copy()
    largura_fundo = max(w1, w2) + 30
    altura_fundo = h1 + h2 + 50
    cv2.rectangle(overlay, (10, 10), (10 + largura_fundo, 10 + altura_fundo),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Desenhar textos
    cv2.putText(frame, texto_classe, (20, 40), fonte, escala, cor, espessura)
    cv2.putText(frame, texto_conf, (20, 40 + h1 + 15), fonte, escala,
                (255, 255, 255), espessura)


def exibir_imagem(frame, titulo: str = "EggVision") -> None:
    """
    Exibe uma imagem em uma janela OpenCV e aguarda qualquer tecla.

    Args:
        frame: Imagem OpenCV (numpy array BGR).
        titulo: Título da janela.
    """
    cv2.imshow(titulo, frame)
    print("\n🖼️  Pressione qualquer tecla para fechar a janela...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def obter_resultado(resultado) -> tuple:
    """
    Extrai a classe prevista e a confiança do resultado YOLO.

    Args:
        resultado: Objeto de resultado retornado pelo modelo YOLO.

    Returns:
        Tupla (classe_nome, confianca) onde:
        - classe_nome é a string da classe prevista
        - confianca é o float da confiança (0 a 1)
    """
    probs = resultado.probs
    classe_idx = probs.top1
    confianca = probs.top1conf.item()
    classe_nome = resultado.names[classe_idx]

    return classe_nome, confianca
