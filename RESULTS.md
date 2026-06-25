# Resultados — Comparativo de Modelos TSC + XGBoost

Execução de `python core_tsc_xgboost_pipeline.py --compare` em **25/06/2026**.

## Configuração do experimento

| Item | Valor |
|---|---|
| Dataset | `dataset_aumentado.xlsx` |
| Amostras | 160.000 |
| Colunas | 24 (5 estáticas + 18 temporais) |
| Alvo | `default payment next month` (1 = inadimplente) |
| Validação | Stratified 10-fold (shuffle, `random_state=42`) |
| Teste por fold | ~16.000 amostras |
| XGBoost | 200 árvores, `max_depth=6`, `lr=0.08`, `scale_pos_weight` automático |
| Hardware | CPU 16 núcleos + GPU NVIDIA RTX 4060 Laptop (apenas LSTM) |

## Métricas agregadas (média ± desvio, 10 folds)

| Modelo | AUC-ROC | F1 | G-Mean | Veredito |
|---|---|---|---|---|
| **CIF** | **0,9959 ± 0,0004** | **0,9742 ± 0,0013** | **0,9743 ± 0,0013** | 🥇 melhor e mais estável |
| Null (baseline) | 0,9546 ± 0,0014 | 0,8822 ± 0,0022 | 0,8847 ± 0,0021 | referência forte |
| ROCKET | 0,9438 ± 0,0021 | 0,8683 ± 0,0027 | 0,8706 ± 0,0026 | abaixo do baseline |
| Shapelets | 0,9198 ± 0,0028 | 0,8373 ± 0,0029 | 0,8399 ± 0,0030 | abaixo do baseline |
| LSTM | 0,8083 ± 0,0043 | 0,7293 ± 0,0034 | 0,7297 ± 0,0033 | 🔻 pior |

## Leitura dos resultados

### CIF vence com folga
O Canonical Interval Forest entrega AUC 0,996 e F1 0,974 — o **único modelo que supera o baseline Null**, e por margem grande (+4 p.p. de AUC, +9 p.p. de F1). Também é o mais estável (desvio de AUC de apenas 0,0004). As features Catch22 por intervalo (média, desvio, inclinação + 22 estatísticas não-lineares) capturam padrões temporais que os demais transforms perdem.

### O baseline Null é difícil de bater
O Null apenas repassa as 18 colunas temporais brutas ao XGBoost — e mesmo assim atinge AUC 0,955. Isso indica que **o XGBoost já extrai quase todo o sinal diretamente das séries cruas**. Qualquer meta-feature precisa agregar informação além do que o XGBoost já consegue sozinho — barra alta.

### ROCKET, Shapelets e LSTM perdem informação
Os três ficam **abaixo do Null**. A causa é compressão: cada um substitui as 18 colunas temporais por uma representação que descarta sinal discriminativo:
- **ROCKET** (kernels aleatórios → MAX/PPV): pooling global perde a posição temporal dos padrões.
- **Shapelets** (distância mínima a subsequências aleatórias): só 100 distâncias, subsequências não otimizadas para a classe.
- **LSTM** (autoencoder): o objetivo é **reconstruir** a série (MSE), não separar inadimplentes — o estado oculto otimiza fidelidade de reconstrução, não discriminação. Por isso é o pior (AUC 0,808).

### Matriz de confusão
Por fold (~16k amostras, classes ~balanceadas no dataset aumentado): o CIF erra ~430 amostras/fold (FP+FN ≈ 160+260), contra ~1.800 do Null e ~4.300 do LSTM. CIF tem FP especialmente baixo (~165/fold) → poucos falsos alarmes de inadimplência.

## Tempo de execução

Tempos derivados dos timestamps do log (pós-otimizações).

| Modelo | Tempo total | Cache (1×) | Por fold | Observação |
|---|---|---|---|---|
| Null | ~10 s | — | trivial | passthrough + XGBoost |
| CIF | ~6 min 14 s | ~87 s | ~29 s | cache Catch22 1× + 200 árvores/fold |
| ROCKET | ~32 min 24 s | ~76 s | ~187 s | **gargalo** = XGBoost em 4.000 features |
| LSTM | ~9 min 40 s | — | ~58 s | treino na GPU (`device=cuda`) |
| Shapelets | ~1 min | ~11 s | ~5 s | transform vetorizado |
| **Total** | **~49,5 min** | | | run completo das 5 variantes |

### Onde o tempo é gasto agora
As otimizações moveram o gargalo. O transform dos TSC deixou de dominar:
- **ROCKET = 65% do runtime total**, mas **não** pelo transform (cache uma vez = 76 s). O custo é o **XGBoost treinando sobre a matriz de 4.000 colunas** (2.000 kernels × 2) em cada um dos 10 folds (~187 s/fold). Reduzir `num_kernels` ou aplicar seleção de features atacaria isso diretamente.
- **CIF**: extração Catch22 agora roda **uma vez** sobre as 160k linhas (~87 s, paralela) em vez de 10×; o custo por fold é o treino das 200 árvores de decisão.
- **LSTM**: ~58 s/fold na GPU — antes seria ~24 min/fold em CPU com batch 64.
- **Shapelets**: transform vetorizado tornou o modelo praticamente gratuito (~1 min no total).

## Ressalvas

- **Dataset aumentado**: `dataset_aumentado.xlsx` (160k) é uma versão expandida do UCI original (30k). Métricas muito altas (CIF AUC 0,996) podem refletir **artefatos da augmentation** — se o processo duplicou/interpolou amostras, linhas correlacionadas podem cair em treino e teste de folds diferentes, inflando o desempenho. Validar com split por grupo/origem antes de concluir poder preditivo real.
- **LSTM subtreinado**: 30 épocas, `hidden_size=32`. O objetivo autoencoder não é alinhado à tarefa; um encoder supervisionado tenderia a superar este resultado.
- Números de tempo são de uma única execução (sem repetição); variam com carga da máquina e disputa da GPU.

## Conclusão

Para este dataset, **CIF é o único transform que agrega valor** sobre passar as séries cruas ao XGBoost. ROCKET/Shapelets/LSTM custam tempo e pioram a métrica — úteis apenas como comparação. Próximo passo de eficiência com maior retorno: **reduzir a dimensionalidade do ROCKET** (menos kernels ou seleção de features) para cortar o maior bloco de runtime.
