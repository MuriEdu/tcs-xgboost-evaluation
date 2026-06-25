# Pipeline TSC + XGBoost para Previsão de Inadimplência

Pipeline modular para previsão de inadimplência de clientes combinando Classificação de Séries Temporais (TSC) com XGBoost.

Características principais:
- Interface TSC genérica com cinco implementações intercambiáveis (Null, CIF, ROCKET, LSTM, Shapelets)
- Engenharia de features com Polars (processamento colunar vetorizado)
- Ingestão via DuckDB (`.xlsx`, `.csv`, `.parquet`) e pandas+xlrd (`.xls` legado)
- Validação cruzada estratificada preservando distribuição de classes
- Métricas voltadas a dados desbalanceados: AUC-ROC, F1, G-Mean, Matriz de Confusão
- Otimizações de performance: transforms vetorizados, cache de features entre folds, paralelismo (joblib) e treino do LSTM em GPU — ver [Performance](#performance-e-otimizações)

Resultados do comparativo mais recente em [`RESULTS.md`](RESULTS.md).

## Estrutura do Repositório

```
tcs-xgboost-evaluation/
├── core_tsc_xgboost_pipeline.py   # Orquestração principal do pipeline
├── dataset_aumentado.xlsx          # Dataset usado por padrão (160k amostras)
├── RESULTS.md                      # Resultados e análise do comparativo
└── tsc_models/
    ├── base.py                     # Interface BaseTimeSeriesClassifier
    ├── null_classifier.py          # Baseline pass-through
    ├── canonical_interval_forest.py
    ├── rocket_classifier.py
    ├── lstm_classifier.py
    ├── shapelet_classifier.py
    ├── test_canonical_interval_forest.py
    ├── test_lstm_classifier.py
    └── test_rocket_classifier.py
```

## Arquitetura do Pipeline

O pipeline é composto por cinco camadas sequenciais:

### 1. Ingestão de Dados — `load_dataset`

Lê o arquivo de entrada e retorna um `DataFrame` Polars.

- **`.xlsx`**: lido via DuckDB com extensão Excel (SQL sobre arquivo)
- **`.xls`**: lido via pandas + xlrd (DuckDB não suporta formato XLS legado)
- **`.csv` / `.parquet`**: lidos diretamente pelo Polars

A coluna-alvo é normalizada para binário 0/1 por `normalize_target`, suportando entradas numéricas, booleanas e textuais (`"yes"/"no"`, `"true"/"false"`, `"default"/"non"`, etc.).

### 2. Separação de Features — `split_static_temporal_features`

Divide as colunas em dois grupos:

- **Features estáticas**: atributos sem variação temporal (ex: dados cadastrais)
- **Features temporais**: colunas que representam sequências no tempo (ex: histórico de pagamentos)

A detecção é automática por padrões de nome: `t1`, `time_1`, `lag_1`, `hist_1`, `month_01`, `week_01`, entre outros. Se nenhuma coluna temporal for detectada, todas as colunas numéricas são tratadas como candidatas temporais.

### 3. Geração de Meta-Features — `BaseTimeSeriesClassifier`

Aplica um classificador TSC sobre as features temporais de cada amostra para gerar meta-features compactas e discriminativas. O modelo é treinado no conjunto de treino de cada fold e aplicado ao conjunto de teste.

Interface obrigatória:

```python
model.fit(X_temporal_train, y_train)   # aprende representação da série
model.transform(X_temporal_test)       # → pl.DataFrame de meta-features
```

As meta-features são concatenadas com as features estáticas para formar a entrada final do XGBoost.

#### Modelos TSC disponíveis

| Modelo | Como gera meta-features | Saída |
|---|---|---|
| **Null** | Retorna as features temporais brutas sem transformação | `n_features` colunas |
| **CIF** | Treina 200 árvores de decisão sobre intervalos aleatórios da série; extrai média, desvio, inclinação e 22 features Catch22 por intervalo | `n_estimators × n_classes` colunas |
| **ROCKET** | Aplica 2.000 kernels convolucionais 1D aleatórios; extrai MAX e PPV (proporção de valores positivos) por kernel | `4.000` colunas |
| **LSTM** | Treina autoencoder LSTM para reconstruir a sequência temporal; o estado oculto final é a meta-feature | `hidden_size` colunas |
| **Shapelets** | Sorteia subsequências aleatórias da série (shapelets); extrai a distância mínima z-normalizada de cada amostra a cada shapelet | `num_shapelets` colunas |

**CIF — Canonical Interval Forest**
Sorteia intervalos aleatórios da série temporal e extrai 25 features por intervalo (média, desvio padrão, inclinação + 22 features não-lineares via biblioteca `pycatch22`). Cada árvore é treinada com bootstrap sobre um subconjunto de intervalos, formando um ensemble.

**ROCKET — RandOm Convolutional KErnel Transform**
Gera kernels com pesos, comprimentos, dilatações e paddings aleatórios. Aplica cada kernel como convolução 1D sobre a série e extrai dois valores: o máximo da saída (MAX) e a proporção de valores positivos (PPV). Os kernels não são treinados — a aleatoriedade cobre o espaço de padrões e o XGBoost seleciona os relevantes.

**LSTM**
Trata cada coluna temporal como um passo de tempo com dimensão de entrada 1. Treina um autoencoder (encoder LSTM + decoder linear) para reconstruir a série de entrada via MSE. O estado oculto do encoder no último passo de tempo é usado como representação compacta da série. Treina automaticamente em GPU quando há CUDA disponível.

**Shapelets — Random Shapelet Transform**
Sorteia subsequências aleatórias da série temporal (shapelets), cada uma z-normalizada. Para cada amostra, calcula a distância euclidiana mínima z-normalizada (via janela deslizante) entre a série e cada shapelet. As distâncias são as meta-features — o XGBoost aprende quais shapelets são discriminativos.

### 4. Classificação — `train_xgboost_classifier`

Treina um `XGBClassifier` sobre a matriz final `[features estáticas | meta-features]`.

- **Desbalanceamento de classes**: tratado via `scale_pos_weight = n_negativos / n_positivos`
- **Hiperparâmetros**: 200 estimadores, profundidade máxima 6, taxa de aprendizado 0,08, subsample 0,8, objetivo `binary:logistic`

### 5. Validação — `validate_pipeline`

Executa validação cruzada estratificada K-fold. Em cada fold:

1. Divide os dados mantendo a proporção de classes (estratificação)
2. Treina o modelo TSC e transforma as features temporais
3. Treina o XGBoost no conjunto de treino
4. Avalia no conjunto de teste

Retorna métricas por fold e agregadas (média e desvio padrão).

## Modelos TSC — Comparativo

```
         ┌─────────────────────────────────────────────────────┐
         │                 Features Temporais                   │
         │           [hist_1, hist_2, ..., hist_n]              │
         └────────────────────────┬────────────────────────────┘
                                  │
      ┌──────────────┬────────────┼────────────┬───────────────┐
      │              │            │            │               │
    Null            CIF        ROCKET        LSTM          Shapelets
 (pass-through) (200 árvores) (2k kernels) (autoencoder) (100 shapelets)
      │              │            │            │               │
  18 features   400 features  4.000 feat.  32 features    100 feat.
      │              │            │            │               │
      └──────────────┴────────────┴────────────┴───────────────┘
                                  │
                     concat(static + meta-features)
                                  │
                             XGBClassifier
                                  │
                        AUC-ROC · F1 · G-Mean
```

## Performance e Otimizações

O pipeline foi otimizado para rodar em datasets grandes (160k+ amostras) sem recomputar trabalho redundante entre folds. As principais técnicas:

| Otimização | Onde | Ganho |
|---|---|---|
| **Convolução vetorizada** | ROCKET (`_apply_kernel_batch`) | substitui laços Python por `sliding_window_view` + matmul (~100–1000×) |
| **Distância vetorizada** | Shapelets (`_min_dist_batch`) | janela deslizante z-normalizada em lote (~285×) |
| **`fit-once` global** | Null, ROCKET, Shapelets | transform não-supervisionado e determinístico roda **uma vez** sobre todas as linhas e é fatiado por fold (`supports_global_transform`) |
| **Cache de features entre folds** | CIF (`prepare_cache` + `fit_indexed`) | Catch22 por intervalo é independente do fold → extraído uma vez, só as árvores retreinam por fold (`supports_indexed_fit`) |
| **Paralelismo** | CIF (joblib) | extração Catch22 em chunks de linhas + treino de árvores em threads |
| **GPU** | LSTM | treino em CUDA automático; batches movidos com `pin_memory`/`non_blocking` |

Essas flags são lidas por `validate_pipeline`, que escolhe o caminho mais rápido por modelo. Para benchmarks concretos do run mais recente, ver [`RESULTS.md`](RESULTS.md).

## Instalação

Requer Python 3.10+.

```bash
pip install polars duckdb pyarrow pandas xlrd numpy scikit-learn xgboost pycatch22 torch
```

Para acelerar o LSTM em GPU NVIDIA, instale o torch com suporte CUDA (ex. CUDA 12.4):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

## Uso

**Execução padrão** (modelo CIF, validação cruzada estratificada):
```bash
python core_tsc_xgboost_pipeline.py
```

**Comparar todos os modelos TSC:**
```bash
python core_tsc_xgboost_pipeline.py --compare
```

**Executar testes:**
```bash
pytest tsc_models/
```

## Dataset

Base original: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (30.000 amostras, 23 features).

A execução padrão usa `dataset_aumentado.xlsx`, uma versão expandida:

- **160.000 amostras**, 24 colunas (5 estáticas + 18 temporais)
- Features temporais: histórico de pagamentos (PAY_0–PAY_6), valores de fatura (BILL_AMT1–6) e pagamentos realizados (PAY_AMT1–6)
- Features estáticas: limite de crédito, sexo, escolaridade, estado civil, idade
- **Target `default payment next month`**: 1 = inadimplente no mês seguinte, 0 = adimplente

> **Nota:** por ser um dataset aumentado, métricas muito altas podem refletir correlação entre amostras sintéticas em folds diferentes. Ver ressalvas em [`RESULTS.md`](RESULTS.md).

## Métricas de Avaliação

| Métrica | Justificativa |
|---|---|
| **AUC-ROC** | Avalia a capacidade de ranqueamento do modelo em todos os limiares de decisão |
| **F1** | Equilíbrio entre precisão e recall na classe minoritária (inadimplentes) |
| **G-Mean** | `√(Sensibilidade × Especificidade)` — penaliza modelos que ignoram a classe minoritária |
| **Matriz de Confusão** | Detalha TP, FP, TN, FN por fold para análise de erros |

## Adicionando Novos Modelos TSC

1. Criar `tsc_models/meu_modelo.py` herdando de `BaseTimeSeriesClassifier`
2. Implementar `fit(X: pl.DataFrame, y: pl.Series)` e `transform(X: pl.DataFrame) -> pl.DataFrame`
3. Exportar em `tsc_models/__init__.py`
4. Instanciar em `core_tsc_xgboost_pipeline.py`

## Big Data — Persistência Intermediária

Para datasets grandes, prefira Parquet ao Excel para recarregamentos:

```python
X_static.write_parquet("static_features.parquet")
X_temporal.write_parquet("temporal_features.parquet")
```

Exportação direta via DuckDB:

```python
con.execute("COPY (SELECT * FROM my_table) TO 'output.parquet' (FORMAT PARQUET)")
```

## Licença

Repositório disponibilizado para fins acadêmicos e de prototipagem.
