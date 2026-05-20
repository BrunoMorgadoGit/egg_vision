# 🥚 EggVision — Classificação de Ovos com YOLO

Sistema MVP de visão computacional para classificar ovos como **normal** ou **defeituoso** usando Ultralytics YOLO Classification e OpenCV.

## 📋 O que é o EggVision?

O EggVision é um **MVP (Produto Mínimo Viável)** que utiliza inteligência artificial para identificar se um ovo está **normal** ou **defeituoso** a partir de imagens simples, sem esteira industrial.

### Classes de classificação:

| Classe | Descrição |
|--------|-----------|
| **normal** | Ovos sem defeitos visíveis |
| **defeituoso** | Ovos rachados **ou** sujos (agrupados em uma única classe) |

> **📌 Importante:** Este MVP classifica a **imagem inteira** — ele não localiza onde está a rachadura ou sujeira. A evolução futura será usar **YOLO Detection com bounding boxes** para localizar exatamente o defeito.

## 📁 Estrutura do Projeto

```
eggvision/
├── raw_dataset/                 ← suas imagens originais
│   ├── normal/                  ← fotos de ovos normais
│   ├── rachado/                 ← fotos de ovos rachados
│   └── sujo/                    ← fotos de ovos sujos
├── dataset/                     ← gerado automaticamente pelo prepare_dataset.py
│   ├── train/
│   │   ├── normal/
│   │   └── defeituoso/
│   ├── val/
│   │   ├── normal/
│   │   └── defeituoso/
│   └── test/
│       ├── normal/
│       └── defeituoso/
├── src/
│   ├── prepare_dataset.py       ← organiza raw_dataset → dataset
│   ├── check_dataset.py         ← verifica e mostra estatísticas
│   ├── train.py                 ← treina o modelo YOLO
│   ├── predict_image.py         ← predição em imagem
│   ├── predict_video.py         ← predição em vídeo
│   ├── predict_webcam.py        ← predição em webcam (tempo real)
│   └── utils.py                 ← funções utilitárias
├── runs/                        ← resultados de treinamento
├── requirements.txt
├── .gitignore
└── README.md
```

## 📷 Onde Colocar as Imagens

Coloque suas fotos originais nas pastas do `raw_dataset/`:

| Pasta | Conteúdo |
|-------|----------|
| `raw_dataset/normal/` | Fotos de ovos normais (sem defeitos) |
| `raw_dataset/rachado/` | Fotos de ovos rachados |
| `raw_dataset/sujo/` | Fotos de ovos sujos |

### Como Nomear as Imagens

Use o padrão `categoria_ovoXX_fotoYY.extensão`:

```
raw_dataset/
  normal/
    normal_ovo01_foto01.jpg
    normal_ovo01_foto02.jpg    ← mesma ovo, ângulo diferente
    normal_ovo02_foto01.jpg
  rachado/
    rachado_ovo01_foto01.jpg
    rachado_ovo01_foto02.jpg
  sujo/
    sujo_ovo01_foto01.jpg
    sujo_ovo02_foto01.jpg
```

> **⚠️ Importante sobre nomes:** O script `prepare_dataset.py` usa o padrão `ovoXX` para agrupar fotos do mesmo ovo. Isso evita que fotos do mesmo ovo caiam em splits diferentes (vazamento de dados). Se os nomes não seguirem esse padrão, cada arquivo será tratado como um ovo independente.

**Extensões aceitas:** `.jpg`, `.jpeg`, `.png`, `.webp`

### Quantidade Recomendada

| Classe | Mínimo MVP | Ideal |
|--------|-----------|-------|
| normal | ~100 fotos | 200+ |
| rachado | ~50 fotos | 100+ |
| sujo | ~50 fotos | 100+ |

## 🚀 Instalação

### 1. Acesse o projeto

```bash
cd eggvision
```

### 2. Crie e ative um ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> **Nota:** O PyTorch será instalado automaticamente pelo `ultralytics`. Se tiver GPU NVIDIA com CUDA, consulte [pytorch.org](https://pytorch.org/) para instalar a versão com suporte a GPU.

## 📦 Passo a Passo Completo

### 1. Preparar o dataset

Após colocar suas imagens em `raw_dataset/`, execute:

```bash
python src/prepare_dataset.py
```

Isso vai:
- Ler as imagens de `raw_dataset/normal/`, `rachado/` e `sujo/`
- Agrupar `rachado` e `sujo` na classe `defeituoso`
- Dividir em **train (70%)**, **val (15%)** e **test (15%)**
- Manter fotos do mesmo ovo no mesmo split (evita vazamento)
- Salvar em `dataset/train/`, `dataset/val/`, `dataset/test/`

**Opções:**
```bash
# Alterar proporções
python src/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1

# Alterar seed
python src/prepare_dataset.py --seed 123
```

> **⚠️ Atenção:** O `prepare_dataset.py` limpa a pasta `dataset/` antes de gerar. Se você adicionou imagens manualmente nela, elas serão perdidas!

### 2. Verificar o dataset

```bash
python src/check_dataset.py
```

Mostra estatísticas: total de imagens, quantidade por split e por classe, e alertas sobre desequilíbrios.

### 3. Treinar o modelo

```bash
python src/train.py
```

O script irá:
1. Verificar se o dataset está organizado
2. Carregar o modelo pré-treinado `yolo11n-cls.pt`
3. Treinar por 50 épocas com data augmentation
4. Salvar o melhor modelo em `runs/classify/eggvision_mvp/weights/best.pt`

### 4. Testar com imagem

```bash
python src/predict_image.py caminho/da/imagem.jpg
```

Exemplo:
```bash
python src/predict_image.py dataset/test/normal/normal_ovo01_foto01.jpg
```

### 5. Testar com vídeo

```bash
python src/predict_video.py caminho/do/video.mkv
```

Formatos aceitos: `.mp4`, `.avi`, `.mkv`. Pressione `q` para sair.

### 6. Testar com webcam (tempo real)

```bash
python src/predict_webcam.py
```

Abre a webcam padrão e classifica em tempo real. Pressione `q` para sair.

## ⚙️ Configurações de Treinamento

Ajustáveis no arquivo `src/train.py`:

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `MODEL_NAME` | `yolo11n-cls.pt` | Modelo base (trocar para `yolov8n-cls.pt` se YOLO11 não disponível) |
| `IMGSZ` | `224` | Tamanho da imagem de entrada |
| `EPOCHS` | `50` | Número de épocas |
| `BATCH` | `16` | Tamanho do batch (reduzir se faltar memória) |
| `SEED` | `42` | Seed para reprodutibilidade |

### Data Augmentation

O augmentation é aplicado **durante o treinamento** pelo YOLO (não cria imagens extras no disco):

| Augmentation | Valor | Descrição |
|-------------|-------|-----------|
| `degrees` | 25 | Rotação ±25 graus |
| `fliplr` | 0.5 | Espelhamento horizontal |
| `translate` | 0.1 | Translação 10% |
| `scale` | 0.2 | Zoom ±20% |
| `hsv_s` | 0.2 | Variação de saturação ±20% |
| `hsv_v` | 0.2 | Variação de brilho ±20% |
| `erasing` | 0.1 | Random erasing |

> **💡 Dica:** Data augmentation ajuda muito, mas **não substitui fotos reais variadas**. Quanto mais diversidade de ângulos, iluminações e fundos, melhor o modelo.

## 🔮 Evolução Futura

Este MVP é o **primeiro passo**. As próximas etapas planejadas são:

1. **YOLO Detection** — usar detecção com bounding boxes para **localizar exatamente onde está o defeito**
2. **Separar tipos de defeito** — classificar em rachado, sujo, manchado
3. **Mais dados** — expandir para 500-1000+ imagens
4. **Integração com esteira** — câmera fixa em esteira industrial
5. **API REST** — para integração com outros sistemas
6. **Dashboard** — interface para estatísticas de classificação

## ❓ Solução de Problemas

| Problema | Solução |
|----------|---------|
| Modelo treinado não encontrado | Execute `python src/train.py` primeiro |
| raw_dataset/ não encontrado | Crie as pastas e adicione suas imagens |
| Nenhuma imagem encontrada | Verifique extensões (.jpg, .jpeg, .png, .webp) |
| Webcam não abre | Verifique se está conectada e liberada |
| Erro de memória no treino | Reduza `BATCH` para 8 ou 4 em `train.py` |
| YOLO11 não encontrado | Troque `MODEL_NAME` para `yolov8n-cls.pt` |
| Parâmetro de augmentation incompatível | Comente a linha correspondente em `train.py` |

## 📄 Licença

Projeto educacional / MVP para validação de conceito.
