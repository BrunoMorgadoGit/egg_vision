# EggVision — Classificação de Ovos com YOLOv8

Sistema de visão computacional para classificar a imagem inteira de um ovo como:

| Classe | Situação atual |
|--------|----------------|
| `normal` | Treinável quando houver imagens em `train` e `val` |
| `rachado` | Treinável quando houver imagens em `train` e `val` |
| `sujo` | Preparada na estrutura, mas ignorada enquanto estiver vazia |

Importante: uma pasta vazia não representa uma classe treinada. O modelo só reconhecerá `sujo` depois que imagens reais dessa categoria forem adicionadas aos conjuntos de treino, validação e teste e o treinamento for executado novamente.

## Estrutura

```text
eggvision/
├── raw_dataset/
│   ├── normal/
│   ├── rachado/
│   └── sujo/
├── dataset/
│   ├── train/
│   │   ├── normal/
│   │   ├── rachado/
│   │   └── sujo/
│   ├── val/
│   │   ├── normal/
│   │   ├── rachado/
│   │   └── sujo/
│   └── test/
│       ├── normal/
│       ├── rachado/
│       └── sujo/
├── src/
│   ├── prepare_dataset.py
│   ├── check_dataset.py
│   ├── train.py
│   ├── validate_model.py
│   ├── predict_image.py
│   ├── predict_video.py
│   ├── predict_webcam.py
│   └── utils.py
└── runs/classify/eggvision_mvp/
```

## Instalação

Linux/macOS:

```bash
cd eggvision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd eggvision
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Organizar imagens

Coloque as imagens originais em:

```text
raw_dataset/normal/
raw_dataset/rachado/
raw_dataset/sujo/
```

Extensões aceitas: `.jpg`, `.jpeg`, `.png`, `.webp`.

Use nomes com o mesmo ID para fotos do mesmo ovo, por exemplo:

```text
normal_ovo01_foto01.jpg
normal_ovo01_foto02.jpg
rachado_ovo03_foto01.jpg
```

O `prepare_dataset.py` usa o padrão `ovoXX` para manter fotos do mesmo ovo no mesmo split e reduzir vazamento de dados.

## Preparar dataset

```bash
python src/prepare_dataset.py
```

O script:

- lê `raw_dataset/normal`, `raw_dataset/rachado` e `raw_dataset/sujo`;
- preserva as classes separadas;
- cria `dataset/train`, `dataset/val` e `dataset/test`;
- cria as pastas `sujo` mesmo que ainda estejam vazias;
- não cria imagens falsas.

Opções:

```bash
python src/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1
python src/prepare_dataset.py --seed 123
```

## Verificar dataset

```bash
python src/check_dataset.py
```

A verificação informa:

- pastas ausentes;
- quantidade de imagens por classe;
- classe vazia;
- formatos inválidos ou imagens ilegíveis;
- poucas imagens;
- desequilíbrio entre classes;
- possíveis duplicatas exatas ou imagens muito semelhantes entre treino, validação e teste.

## Treinar

```bash
python src/train.py
```

O treino usa `yolov8n-cls.pt`, ou seja, um modelo YOLO de classificação, não detecção.

Enquanto `sujo` estiver vazia, o script:

- exibe aviso claro;
- ignora `sujo` temporariamente;
- treina somente `normal` e `rachado`;
- informa que o modelo ainda não reconhece ovos sujos.

O melhor modelo fica em:

```text
runs/classify/eggvision_mvp/weights/best.pt
```

## Validar modelo

```bash
python src/validate_model.py --split val
```

Para avaliar no teste:

```bash
python src/validate_model.py --split test
```

O script salva métricas em:

```text
runs/classify/eggvision_mvp/validation_val/
runs/classify/eggvision_mvp/validation_test/
```

Ele calcula precision, recall, F1-score e matriz de confusão, com destaque para o erro crítico `rachado` classificado como `normal`.

## Testar uma imagem

```bash
python src/predict_image.py caminho/da/imagem.jpg
```

Com limite de confiança:

```bash
python src/predict_image.py caminho/da/imagem.jpg --conf 0.70
```

Sem abrir janela OpenCV, útil em servidores:

```bash
python src/predict_image.py caminho/da/imagem.jpg --no-show
```

A saída inclui:

- classe prevista;
- porcentagem de confiança;
- limite mínimo de confiança;
- resultado `incerto` quando a confiança fica abaixo do limite;
- aviso quando o modelo ainda não tem a classe `sujo`;
- imagem anotada salva em `outputs/predictions/`.

## Configurações de treino

Valores principais em `src/train.py`:

| Parâmetro | Valor |
|-----------|-------|
| `MODEL_NAME` | `yolov8n-cls.pt` |
| `IMGSZ` | `320` |
| `EPOCHS` | `80` |
| `PATIENCE` | `15` |
| `BATCH` | `8` |
| `SEED` | `42` |

O treino usa data augmentation moderado: rotação, espelhamento horizontal, translação, escala, variação de saturação/brilho e random erasing. Isso ajuda, mas não substitui imagens reais variadas.

## Para adicionar `sujo` corretamente

1. Adicione imagens reais em `raw_dataset/sujo/`.
2. Garanta amostras suficientes para `train`, `val` e `test`.
3. Execute `python src/prepare_dataset.py`.
4. Execute `python src/check_dataset.py`.
5. Execute `python src/train.py` novamente.
6. Confirme em `python src/validate_model.py --split val` que `sujo` aparece nas métricas.

Enquanto esses passos não forem feitos, o modelo não deve ser usado para reconhecer ovos sujos.
