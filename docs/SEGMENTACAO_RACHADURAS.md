# Segmentação de rachaduras no EggVision

Este guia descreve o segundo modelo do EggVision. Ele não substitui nem altera o classificador em `runs/classify/eggvision_mvp/weights/best.pt`.

## O que cada modelo faz

- Classificação decide se a imagem inteira parece `normal` ou `rachado`.
- Detecção desenharia uma caixa retangular; ela não representa bem uma rachadura fina.
- Segmentação prevê uma máscara por pixel e permite destacar apenas a região rachada.

O modelo de segmentação possui somente a classe `0: rachadura`. `sujo` continua fora deste pipeline.

## Pré-requisitos

Confira o ambiente sem atualizar bibliotecas:

```bash
python --version
python -c "import ultralytics; print(ultralytics.__version__)"
```

As dependências são instaladas com:

```bash
pip install -r requirements.txt
```

O projeto usa a versão já instalada do Ultralytics e o checkpoint compatível `yolov8n-seg.pt`. O checkpoint pode ser baixado pelo Ultralytics na primeira execução de treino quando houver acesso à internet.

## Anotações reais obrigatórias

Somente colocar uma foto em `raw_dataset/rachado/` não ensina a posição da rachadura. Cada imagem rachada precisa de ao menos um polígono manual:

```text
0 x1 y1 x2 y2 x3 y3 ...
```

Regras:

- coordenadas entre 0 e 1;
- pelo menos três pontos por polígono;
- uma linha para cada rachadura separada;
- não usar a imagem inteira ou o ovo inteiro como máscara;
- não converter uma bounding box do ovo em label de rachadura;
- não salvar label vazio para uma imagem rachada.

As imagens normais não precisam de annotation-fonte. Durante a preparação, cada uma recebe um `.txt` vazio no dataset de segmentação para representar ausência de objetos.

## Ferramenta OpenCV

Abra a ferramenta local:

```bash
python src/annotate_cracks.py
```

Atalhos:

| Entrada | Ação |
|---------|------|
| Clique esquerdo | Adicionar ponto ao polígono atual |
| `Enter` | Finalizar o polígono atual, com no mínimo três pontos |
| `U` | Desfazer o último ponto |
| `R` | Limpar o polígono atual |
| `D` | Excluir o último polígono concluído |
| `S` | Salvar os polígonos da imagem |
| `N` | Avançar para a próxima imagem |
| `B` | Voltar para a imagem anterior |
| `Q` ou `Esc` | Sair sem apagar labels já salvas |

A ferramenta reabre labels existentes para edição. Ela impede a troca de imagem enquanto houver alteração não salva e se recusa a salvar uma imagem rachada sem polígono.

Para começar por uma imagem específica:

```bash
python src/annotate_cracks.py --start ovo12-foto1.jpg
```

Para gerar overlays das anotações existentes sem alterar o dataset:

```bash
python src/annotate_cracks.py --preview
```

Saída:

```text
outputs/segmentation_annotation_preview/
```

## Onde as labels são salvas

A árvore relativa das imagens é preservada:

```text
raw_dataset/rachado/Rachado/ovo01-foto1.jpg
annotations/egg_crack_seg/rachado/Rachado/ovo01-foto1.txt
```

Isso permite adicionar outras subpastas sem misturar labels com o dataset bruto.

## Validar antes da preparação

```bash
python src/validate_seg_dataset.py --source-annotations
```

O comando retorna código diferente de zero quando existe imagem rachada sem label, label vazio, polígono inválido, classe diferente de zero, coordenada fora de 0 a 1 ou imagem corrompida. O relatório é salvo em:

```text
runs/segment/dataset_validation_report.json
```

## Preparar os splits

Quando a validação das anotações passar:

```bash
python src/prepare_seg_dataset.py
```

O preparador:

- lê somente `normal` e `rachado`;
- ignora `sujo`;
- usa seed 42 e proporções 70/15/15;
- extrai `ovoXX` do nome;
- agrupa por classe e ovo;
- mantém todas as fotos do mesmo ovo no mesmo split;
- garante exemplos normais e rachados em cada split quando houver ao menos três grupos de cada origem;
- copia imagens e labels com `shutil.copy2`;
- não cria symlinks;
- não modifica `raw_dataset` nem as labels-fonte;
- cria labels vazios somente para imagens normais;
- bloqueia a cópia se alguma imagem rachada estiver sem polígono válido.

Como `ovo01` pode existir nas duas pastas, as chaves são diferentes: `normal:ovo01` e `rachado:ovo01`.

O dataset gerado usa nomes com proveniência explícita:

```text
images/train/normal__ovo01-foto1.jpg
images/train/rachado__ovo01-foto1.jpg
labels/train/normal__ovo01-foto1.txt
labels/train/rachado__ovo01-foto1.txt
```

Para recriar uma saída já preenchida, a remoção precisa ser explícita:

```bash
python src/prepare_seg_dataset.py --rebuild
```

Somente `datasets/egg_crack_seg/` é removido por essa opção. O dataset bruto e as annotations permanecem intactos.

## Validar o dataset preparado

```bash
python src/validate_seg_dataset.py
```

São verificados:

- YAML e caminhos de `train`, `val` e `test`;
- imagem sem label e label sem imagem;
- label vazio em rachado e label não vazio em normal;
- classe, número de coordenadas, pontos, limites e valores numéricos;
- imagens corrompidas e extensões não suportadas;
- duplicatas exatas por SHA-256;
- mesmo ovo em splits diferentes;
- presença de exemplos normais e rachados em cada split quando possível.

Erros impeditivos retornam código diferente de zero.

## Testes estruturais

```bash
python -m unittest discover -s tests -v
```

Os testes não exigem GPU nem modelo treinado. Eles usam diretórios temporários, imagens mínimas e mocks da resposta do YOLO.

Confira também as interfaces:

```bash
python src/prepare_seg_dataset.py --help
python src/validate_seg_dataset.py --help
python src/train_seg.py --help
python src/evaluate_seg.py --help
python src/predict_seg.py --help
python src/annotate_cracks.py --help
```

## Smoke training

Execute somente depois que o validador retornar sucesso:

```bash
python src/train_seg.py \
  --epochs 1 \
  --imgsz 320 \
  --batch 1 \
  --name eggvision_crack_seg_smoke
```

Esse teste confirma integração com o Ultralytics. Ele não mede qualidade e seus resultados não devem ser confundidos com o treino oficial.

## Treinamento inicial

```bash
python src/train_seg.py \
  --data datasets/egg_crack_seg/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 100 \
  --patience 20 \
  --imgsz 1024 \
  --batch 4 \
  --device auto \
  --workers 0 \
  --project runs/segment \
  --name eggvision_crack_seg \
  --seed 42
```

Parâmetros de proteção:

- transfer learning com `yolov8n-seg.pt`;
- early stopping com patience 20;
- seed fixa 42 e execução determinística;
- rotação de 5 graus, translação de 5% e escala de 10%;
- variações HSV leves;
- `mosaic=0` e `mixup=0`;
- random erasing desativado;
- gráficos e pesos `best.pt`/`last.pt`;
- recusa de sobrescrever uma execução que já possui artefatos.

Se houver falta de memória:

```bash
python src/train_seg.py --batch 2
python src/train_seg.py --batch 1
```

Reduza `--imgsz` somente depois e registre o valor usado, pois rachaduras finas perdem detalhes em resoluções baixas.

Retomada pelo caminho padrão:

```bash
python src/train_seg.py --name eggvision_crack_seg --resume
```

Ou por um checkpoint explícito:

```bash
python src/train_seg.py --resume runs/segment/eggvision_crack_seg/weights/last.pt
```

## Avaliação no teste

```bash
python src/evaluate_seg.py \
  --model runs/segment/eggvision_crack_seg/weights/best.pt \
  --split test
```

O JSON separa precision, recall, mAP50 e mAP50-95 de bounding box e máscara:

```text
runs/segment/eggvision_crack_seg/evaluation/test_metrics.json
```

O conjunto de teste atual será pequeno mesmo após as anotações. Uma única previsão incorreta pode alterar as métricas em muitos pontos percentuais; os números não serão conclusivos.

## Inferência

Imagem individual:

```bash
python src/predict_seg.py \
  --model runs/segment/eggvision_crack_seg/weights/best.pt \
  --source caminho/ovo.jpg \
  --conf 0.25 \
  --save-json
```

Pasta:

```bash
python src/predict_seg.py --source caminho/pasta --conf 0.25 --save-json
```

Use `--show` para abrir a imagem e `--no-text` para salvar somente máscara/contorno. Por padrão, não é desenhada bounding box.

Regra de decisão:

- nenhuma máscara da classe 0 acima de `--conf`: `NORMAL`;
- uma ou mais máscaras da classe 0 acima de `--conf`: `RACHADO`.

Cada JSON registra nome, status, threshold, número de rachaduras, confiança máxima, confiança individual, área em pixels e proporção da imagem.

Saídas:

```text
outputs/segmentation_predictions/*_seg.jpg
outputs/segmentation_predictions/*_seg.json
outputs/segmentation_predictions/predictions.json  # fonte em pasta
```

## Artefatos e overfitting

```text
runs/segment/eggvision_crack_seg/weights/best.pt
runs/segment/eggvision_crack_seg/weights/last.pt
runs/segment/eggvision_crack_seg/results.csv
runs/segment/eggvision_crack_seg/results.png
```

Gere um resumo auxiliar:

```bash
python src/analyze_seg_training.py
```

O script informa melhor e última época, melhor métrica de máscara, épocas sem melhora, diferença final entre perdas e uma tendência auxiliar de overfitting. Ele exige divergência em várias épocas e não conclui overfitting por um único ponto.

Analise também manualmente:

- perda de treino caindo enquanto a de validação sobe por várias épocas;
- métricas de validação instáveis;
- grande diferença entre treino e validação;
- piora no teste apesar de resultado excelente no treino.

## Limitações atuais

- Data augmentation gera variações da mesma foto, não novos ovos.
- Fotos do mesmo ovo com IDs diferentes ainda podem vazar entre splits.
- Rachaduras variam em espessura, iluminação e contraste; são necessários mais ovos físicos.
- Imagens normais são essenciais para medir falsos positivos.
- A classe `sujo` não faz parte da segmentação de rachaduras.
- O melhor peso deve ser usado na inferência, mas ele não compensa labels imprecisas.
- Métricas de validação/teste pequenos não comprovam generalização.
