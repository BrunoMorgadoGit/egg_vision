"""Funções reutilizáveis do EggVision."""

import hashlib
import os
import shutil
import sys

import cv2
from ultralytics import YOLO


# =============================================
# Configurações compartilhadas
# =============================================
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
PROJECT_CLASSES = ("normal", "rachado", "sujo")
REQUIRED_SPLITS = ("train", "val", "test")
IGNORED_FILENAMES = {".gitkeep", ".ds_store", "thumbs.db"}

MIN_TRAIN_IMAGES_PER_CLASS = 20
MIN_VAL_IMAGES_PER_CLASS = 5
IMBALANCE_WARNING_RATIO = 3.0
SIMILAR_IMAGE_HASH_DISTANCE = 4
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
UNCERTAIN_CLASS = "incerto"


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

def is_ignored_aux_file(filepath: str) -> bool:
    """Retorna True para arquivos auxiliares que não fazem parte do dataset."""
    return os.path.basename(filepath).lower() in IGNORED_FILENAMES


def inspect_image_file(filepath: str) -> tuple[bool, str]:
    """
    Valida se um arquivo é uma imagem aceita e legível pelo OpenCV.

    Verifica:
      1. Se o arquivo existe
      2. Se a extensão está na lista de extensões válidas
      3. Se o arquivo não está vazio (tamanho > 0)
      4. Se o OpenCV consegue decodificar a imagem

    Args:
        filepath: Caminho do arquivo a ser validado.

    Returns:
        Tupla (válido, motivo).
    """
    if not os.path.isfile(filepath):
        return False, "não é arquivo"

    if is_ignored_aux_file(filepath):
        return False, "arquivo auxiliar ignorado"

    _, ext = os.path.splitext(filepath)
    if ext.lower() not in VALID_IMAGE_EXTENSIONS:
        return False, f"extensão inválida ({ext or 'sem extensão'})"

    if os.path.getsize(filepath) == 0:
        return False, "arquivo vazio"

    image = cv2.imread(filepath)
    if image is None or image.size == 0:
        return False, "OpenCV não conseguiu ler"

    return True, "ok"


def validate_image_file(filepath: str) -> bool:
    """Retorna True se o arquivo é uma imagem válida para o projeto."""
    valid, _ = inspect_image_file(filepath)
    return valid


def scan_class_directory(class_path: str) -> dict:
    """Lista imagens válidas e arquivos inválidos de uma pasta de classe.

    A varredura é recursiva para tolerar subpastas como normal/Normal/*.jpg
    sem mudar o rótulo da classe.
    """
    result = {"images": [], "invalid": []}

    if not os.path.isdir(class_path):
        return result

    for current_dir, _, filenames in os.walk(class_path):
        for filename in sorted(filenames):
            filepath = os.path.join(current_dir, filename)
            if is_ignored_aux_file(filepath):
                continue

            valid, reason = inspect_image_file(filepath)
            if valid:
                result["images"].append(filepath)
            else:
                result["invalid"].append((filepath, reason))

    return result


def ensure_dataset_structure(dataset_path: str,
                             classes: tuple[str, ...] = PROJECT_CLASSES) -> None:
    """Cria dataset/{train,val,test}/{normal,rachado,sujo} se não existir."""
    for split in REQUIRED_SPLITS:
        for class_name in classes:
            safe_mkdir(os.path.join(dataset_path, split, class_name))


def get_split_counts(dataset_path: str,
                     classes: tuple[str, ...] = PROJECT_CLASSES) -> dict:
    """Retorna contagens por split e classe esperada."""
    counts = {}

    for split in REQUIRED_SPLITS:
        split_path = os.path.join(dataset_path, split)
        counts[split] = {}
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            scan = scan_class_directory(class_path)
            counts[split][class_name] = len(scan["images"])

    return counts


def get_active_training_classes(counts: dict,
                                classes: tuple[str, ...] = PROJECT_CLASSES) -> list[str]:
    """
    Classes treináveis são as que têm imagens em train e val.

    Teste pode ficar vazio em MVPs pequenos, mas train e val precisam existir
    para o YOLO treinar e medir generalização mínima.

    A classe sujo é tratada de forma mais conservadora: só é ativada quando
    também houver imagens em test, para não parecer reconhecível antes de o
    ciclo completo de avaliação existir.
    """
    active = []
    for class_name in classes:
        train_count = counts.get("train", {}).get(class_name, 0)
        val_count = counts.get("val", {}).get(class_name, 0)
        test_count = counts.get("test", {}).get(class_name, 0)
        if class_name == "sujo" and test_count == 0:
            continue
        if train_count > 0 and val_count > 0:
            active.append(class_name)
    return active


def collect_invalid_files(base_path: str,
                          classes: tuple[str, ...] = PROJECT_CLASSES,
                          splits: tuple[str, ...] | None = REQUIRED_SPLITS) -> list[tuple[str, str]]:
    """Coleta arquivos inválidos em raw_dataset ou dataset preparado."""
    invalid_files = []

    if splits is None:
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            scan = scan_class_directory(class_path)
            invalid_files.extend(scan["invalid"])
        return invalid_files

    for split in splits:
        for class_name in classes:
            class_path = os.path.join(base_path, split, class_name)
            scan = scan_class_directory(class_path)
            invalid_files.extend(scan["invalid"])

    return invalid_files


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

        result[class_name] = len(scan_class_directory(class_path)["images"])

    return result


def file_sha256(filepath: str) -> str:
    """Calcula hash SHA-256 para detectar duplicatas exatas."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def average_hash(filepath: str, hash_size: int = 8) -> int | None:
    """Calcula hash perceptual simples para achar imagens muito semelhantes."""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = resized > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    """Distância de Hamming entre dois hashes perceptuais."""
    return (left ^ right).bit_count()


def collect_dataset_image_records(dataset_path: str,
                                  classes: tuple[str, ...] = PROJECT_CLASSES) -> list[dict]:
    """Coleta imagens válidas do dataset com hashes para auditoria."""
    records = []

    for split in REQUIRED_SPLITS:
        for class_name in classes:
            class_path = os.path.join(dataset_path, split, class_name)
            scan = scan_class_directory(class_path)
            for filepath in scan["images"]:
                records.append({
                    "split": split,
                    "class": class_name,
                    "path": filepath,
                    "sha256": file_sha256(filepath),
                    "ahash": average_hash(filepath),
                })

    return records


def find_cross_split_similar_images(
    records: list[dict],
    max_pairs: int = 20,
    distance_threshold: int = SIMILAR_IMAGE_HASH_DISTANCE,
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict, int]]]:
    """Procura duplicatas exatas e imagens muito semelhantes entre splits."""
    exact_pairs = []
    similar_pairs = []

    for index, left in enumerate(records):
        for right in records[index + 1:]:
            if left["split"] == right["split"]:
                continue

            if left["sha256"] == right["sha256"]:
                exact_pairs.append((left, right))
                if len(exact_pairs) >= max_pairs:
                    break
                continue

            if left["ahash"] is None or right["ahash"] is None:
                continue

            distance = hamming_distance(left["ahash"], right["ahash"])
            if distance <= distance_threshold:
                similar_pairs.append((left, right, distance))
                if len(similar_pairs) >= max_pairs:
                    break

        if len(exact_pairs) >= max_pairs and len(similar_pairs) >= max_pairs:
            break

    return exact_pairs, similar_pairs


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


def get_model_class_names(model: YOLO) -> list[str]:
    """Retorna os nomes de classe conhecidos pelo modelo carregado."""
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)

    if isinstance(names, dict):
        return [names[key] for key in sorted(names.keys())]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def assert_classification_result(resultado) -> None:
    """Falha com mensagem clara quando um modelo de detecção é usado por engano."""
    if getattr(resultado, "probs", None) is None:
        raise ValueError(
            "O modelo carregado não retornou probabilidades de classificação. "
            "Use um modelo de classificação, como yolov8n-cls.pt ou um best.pt "
            "treinado com YOLO Classification."
        )


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
        class_name: Nome da classe prevista ou "incerto".
        confidence: Valor de confiança entre 0 e 1.
    """
    # Textos a serem exibidos
    texto_classe = f"Classe: {class_name}"
    texto_conf = f"Confianca: {format_confidence(confidence)}"

    # Definir cor baseada na classe
    lower_class = class_name.lower()
    if lower_class == "normal":
        cor = (0, 200, 0)       # Verde para normal
    elif lower_class == "rachado":
        cor = (0, 0, 220)       # Vermelho para rachado
    elif lower_class == "sujo":
        cor = (0, 140, 255)     # Laranja para sujo
    elif lower_class == UNCERTAIN_CLASS:
        cor = (0, 220, 220)     # Amarelo para incerto
    else:
        cor = (0, 0, 220)

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


def interpretar_resultado(resultado,
                          confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
    """
    Extrai classe, confiança e resultado final do YOLO Classification.

    Args:
        resultado: Objeto de resultado retornado pelo modelo YOLO.
        confidence_threshold: confiança mínima para aceitar a classe.

    Returns:
        Dicionário com classe prevista, confiança, resultado e flag incerto.
    """
    assert_classification_result(resultado)

    probs = resultado.probs
    classe_idx = probs.top1
    confianca = probs.top1conf.item()
    classe_nome = resultado.names[classe_idx]
    incerto = confianca < confidence_threshold

    return {
        "classe_prevista": classe_nome,
        "confianca": confianca,
        "resultado": UNCERTAIN_CLASS if incerto else classe_nome,
        "incerto": incerto,
    }


def obter_resultado(resultado,
                    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> tuple:
    """Mantém compatibilidade com os scripts de vídeo/webcam."""
    prediction = interpretar_resultado(resultado, confidence_threshold)
    return prediction["resultado"], prediction["confianca"]
