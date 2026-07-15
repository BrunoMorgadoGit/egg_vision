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

## Segmentação de rachaduras

O EggVision possui dois pipelines independentes:

| Tarefa | O que responde | Modelo/saída |
|--------|----------------|--------------|
| Classificação | Avalia a imagem inteira como `normal` ou `rachado` | `runs/classify/eggvision_mvp/weights/best.pt` |
| Detecção | Usaria apenas caixas retangulares; não é usada para localizar a rachadura | — |
| Segmentação | Localiza os pixels da rachadura com máscara e contorno | `runs/segment/eggvision_crack_seg/weights/best.pt` |

O pipeline de segmentação tem uma única classe, `0: rachadura`. A classe `sujo` não participa desse modelo. Imagens normais continuam necessárias como exemplos negativos e recebem labels `.txt` vazias no dataset preparado.

### Estrutura

```text
annotations/egg_crack_seg/rachado/  # polígonos manuais das rachaduras
datasets/egg_crack_seg/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

As anotações espelham o caminho relativo dentro de `raw_dataset/rachado`. No layout atual, por exemplo:

```text
raw_dataset/rachado/Rachado/ovo01-foto1.jpg
annotations/egg_crack_seg/rachado/Rachado/ovo01-foto1.txt
```

Cada linha do label segue YOLO Segmentation:

```text
0 x1 y1 x2 y2 x3 y3 ...
```

As coordenadas são normalizadas entre 0 e 1 e cada polígono tem ao menos três pontos. Imagens rachadas exigem polígonos reais; o projeto não cria máscaras a partir do nome da pasta ou do contorno inteiro do ovo.

### Anotar e revisar

```bash
python src/annotate_cracks.py
python src/annotate_cracks.py --preview
```

O segundo comando salva overlays em `outputs/segmentation_annotation_preview/` sem alterar as imagens ou labels. Os atalhos da interface e o fluxo completo estão em [docs/SEGMENTACAO_RACHADURAS.md](docs/SEGMENTACAO_RACHADURAS.md).

### Validar e preparar

Antes dos splits, confira as anotações-fonte:

```bash
python src/validate_seg_dataset.py --source-annotations
```

Depois que todas as imagens rachadas tiverem labels válidas:

```bash
python src/prepare_seg_dataset.py
python src/validate_seg_dataset.py
```

O preparador usa seed 42, split aproximado 70/15/15, agrupa por `classe + ovoXX` e copia arquivos com `shutil.copy2`. Fotos como `ovo01-foto1.jpg` e `ovo01-foto2.jpg` da mesma classe permanecem no mesmo split. O identificador `normal:ovo01` é diferente de `rachado:ovo01`.

O script não remove o dataset gerado por padrão. Para recriá-lo conscientemente após mudar as anotações:

```bash
python src/prepare_seg_dataset.py --rebuild
```

### Testar o pipeline e treinar

Testes CPU-only, sem download de modelo:

```bash
python -m unittest discover -s tests -v
```

Smoke training, somente depois de o validador retornar código zero:

```bash
python src/train_seg.py --epochs 1 --imgsz 320 --batch 1 --name eggvision_crack_seg_smoke
```

Treinamento inicial completo:

```bash
python src/train_seg.py \
  --data datasets/egg_crack_seg/data.yaml \
  --epochs 100 \
  --patience 20 \
  --imgsz 1024 \
  --batch 4 \
  --name eggvision_crack_seg
```

O treino usa `yolov8n-seg.pt`, seed 42, early stopping e augmentations leves. `mosaic`, `mixup` e random erasing ficam desativados para não apagar ou distorcer rachaduras. Se faltar memória, tente explicitamente `--batch 2` e depois `--batch 1`; reduza `--imgsz` apenas se ainda for necessário.

Para retomar o `last.pt` da execução padrão:

```bash
python src/train_seg.py --name eggvision_crack_seg --resume
```

### Avaliar e inferir

A avaliação final usa o split de teste e registra métricas de caixa e máscara separadamente:

```bash
python src/evaluate_seg.py --split test
```

Inferência em uma imagem ou pasta:

```bash
python src/predict_seg.py --source caminho/ovo.jpg --conf 0.25 --save-json
python src/predict_seg.py --source caminho/pasta --conf 0.25 --save-json
```

Sem máscara da classe `rachadura` acima do limite, o resultado é `NORMAL`; com ao menos uma máscara válida, é `RACHADO`. A saída desenha máscara/contorno, sem caixa gigante ao redor do ovo.

### Artefatos

```text
runs/segment/eggvision_crack_seg/weights/best.pt
runs/segment/eggvision_crack_seg/weights/last.pt
runs/segment/eggvision_crack_seg/results.csv
runs/segment/eggvision_crack_seg/results.png
runs/segment/eggvision_crack_seg/evaluation/test_metrics.json
outputs/segmentation_predictions/*_seg.jpg
outputs/segmentation_predictions/*_seg.json
```

Para um diagnóstico auxiliar das curvas:

```bash
python src/analyze_seg_training.py
```

Esse diagnóstico considera tendências de várias épocas; não declara overfitting por uma época isolada. Compare `train/seg_loss`, `val/seg_loss`, métricas de máscara e `results.png`. Com poucos ovos, uma única falha na validação muda vários pontos percentuais.

Data augmentation não cria ovos fisicamente diferentes. Além disso, se fotos do mesmo ovo tiverem IDs diferentes, o agrupamento automático não consegue reconhecê-las e pode haver vazamento entre treino e teste. As métricas só ganham confiabilidade com mais ovos normais e rachados fisicamente distintos e, futuramente, dados separados para a classe `sujo` no classificador correspondente.
