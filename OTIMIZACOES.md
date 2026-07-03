# Otimizações de Performance — Pipeline TSC + XGBoost

Este documento detalha as otimizações aplicadas ao pipeline para rodar em datasets grandes (160k+ amostras) sem recomputar trabalho redundante entre folds da validação cruzada. Cada seção explica **o problema**, **a técnica** e **por que é correta** (sem vazamento de dados).

## Visão geral

| # | Otimização | Onde | Ganho |
|---|---|---|---|
| 1 | Convolução vetorizada | ROCKET (`_apply_kernel_batch`) | ~100–1000× |
| 2 | Distância vetorizada | Shapelets (`_min_dist_batch`) | ~285× |
| 3 | `fit-once` global | Null, ROCKET, Shapelets | recomputa 1× em vez de `n_splits`× |
| 4 | Cache de features entre folds | CIF (`prepare_cache` + `fit_indexed`) | Catch22 extraído 1× |
| 5 | Paralelismo (joblib) | CIF | escala com núcleos de CPU |
| 6 | Treino em GPU | LSTM | ~4–6× por época |

As flags `supports_global_transform` e `supports_indexed_fit` (declaradas em `tsc_models/base.py`) são lidas por `validate_pipeline`, que escolhe o caminho mais rápido por modelo em tempo de execução.

---

## 1. Convolução vetorizada — ROCKET

**Arquivo:** `tsc_models/rocket_classifier.py` → `_apply_kernel_batch`

### Problema
ROCKET aplica 2.000 kernels convolucionais 1D sobre cada série temporal. A implementação ingênua percorre, em loops Python aninhados, cada amostra × cada posição de saída × cada elemento do kernel. Com 160k amostras e 2.000 kernels, isso são bilhões de iterações no interpretador Python — inviável.

### Técnica
Convolução dilatada reformulada como uma única multiplicação matricial (`matmul`) sobre janelas deslizantes:

```python
# windows: (n_samples, output_len, effective_len)
windows = np.lib.stride_tricks.sliding_window_view(data, effective_len, axis=1)
taps = windows[:, :, ::dilation]   # recupera as posições que o kernel toca
conv = taps @ weight + bias        # (n_samples, output_len) — vetorizado
max_vals = conv.max(axis=1)        # feature MAX
ppv_vals = (conv > 0).mean(axis=1) # feature PPV
```

Pontos-chave:
- `sliding_window_view` cria uma **view** das janelas sem copiar dados (zero alocação).
- O slicing `[:, :, ::dilation]` recupera os taps corretos para kernels dilatados.
- O `matmul` roda em BLAS otimizado (C/SIMD), aplicando o kernel a **todas as amostras e posições de uma vez**.
- Padding tratado com `np.pad` antes da janela quando `use_padding` está ativo.

### Ganho
~100–1000× vs. os loops Python originais. Loop restante em Python apenas sobre os 2.000 kernels (não sobre amostras/posições).

---

## 2. Distância vetorizada — Shapelets

**Arquivo:** `tsc_models/shapelet_classifier.py` → `_min_dist_batch`

### Problema
O Random Shapelet Transform calcula, para cada amostra, a distância euclidiana z-normalizada mínima entre a série e cada shapelet, via janela deslizante. A versão original iterava por amostra × por janela em Python.

### Técnica
Toda a operação — extração de janelas, z-normalização e distância — vetorizada sobre amostras e posições:

```python
windows = np.lib.stride_tricks.sliding_window_view(data, sh_len, axis=1)
mean = windows.mean(axis=2, keepdims=True)
std  = windows.std(axis=2, keepdims=True)
# z-normaliza cada janela; janelas constantes (std==0) ficam intactas
nonzero = std > 0
safe_std = np.where(nonzero, std, 1.0)
norm = np.where(nonzero, (windows - mean) / safe_std, windows)
dist = ((norm - shapelet) ** 2).mean(axis=2)  # (n_samples, n_windows)
return dist.min(axis=1)                        # mínimo por amostra
```

Pontos-chave:
- `safe_std` + `np.where` evitam divisão por zero em janelas constantes, **preservando exatamente** o comportamento da versão escalar original.
- Broadcasting faz a subtração `norm - shapelet` sem loops.

### Ganho
~285×. Loop restante só sobre os 100 shapelets.

---

## 3. `fit-once` global — Null, ROCKET, Shapelets

**Arquivos:** `tsc_models/base.py` (`supports_global_transform`), `core_tsc_xgboost_pipeline.py` (`validate_pipeline`)

### Problema
A validação cruzada faz `n_splits` folds (padrão 10). Para cada fold, o pipeline treina o modelo TSC no treino e transforma treino+teste. Para transforms **não-supervisionados e determinísticos**, isso recomputa a mesma coisa 10 vezes — a mesma linha sempre mapeia para as mesmas meta-features, independentemente da partição treino/teste.

### Técnica
Modelos cujo transform ignora os rótulos e a partição declaram `supports_global_transform = True`. `validate_pipeline` então computa as meta-features **uma vez sobre todas as linhas** e apenas fatia por fold:

```python
if getattr(tsc_model, "supports_global_transform", False):
    global_meta = tsc_model.fit_transform(X_temporal, y)  # 1× em todas as linhas
...
# dentro do fold:
X_meta_train = global_meta.gather(train_idx)
X_meta_test  = global_meta.gather(test_idx)
```

### Por que não há vazamento
O transform é **não-supervisionado** (não usa `y`) e **data-determinístico**: os kernels do ROCKET dependem só de `seq_len`; os shapelets são sorteados com semente fixa; o Null é pass-through. A meta-feature de uma linha de teste é idêntica quer seja computada isoladamente ou junto com as demais. Não há informação do treino "vazando" para o teste.

### Ganho
Transform roda 1× em vez de `n_splits`× (~10× menos trabalho de transform).

---

## 4. Cache de features entre folds — CIF

**Arquivo:** `tsc_models/canonical_interval_forest.py` (`prepare_cache`, `fit_indexed`, `transform_indexed`)

### Problema
O CIF extrai 25 features por intervalo (média, desvio, inclinação + 22 features Catch22 via `pycatch22`) para cada amostra, e treina 200 árvores de decisão. A extração Catch22 é a parte cara — e o CIF **é supervisionado** (as árvores usam `y`), então não pode usar o `fit-once` global da seção 3 diretamente.

### Técnica — desenho em dois estágios
A observação-chave: o conjunto de intervalos e as sementes por árvore derivam apenas de `random_state` e `n_timepoints` — **não dependem das linhas nem dos rótulos**. Logo, as features Catch22 por (linha, intervalo) são idênticas em todos os folds. Só o treino das árvores é supervisionado e precisa refazer por fold.

O modelo declara `supports_indexed_fit = True` e expõe:

- **`prepare_cache(X)`** — estágio 1, roda 1×: planeja os intervalos/sementes e constrói o cache Catch22 sobre **todas as linhas**.
- **`fit_indexed(row_idx, y)`** — estágio 2, por fold: treina as 200 árvores sobre o subconjunto de linhas do fold (com bootstrap), indexando o cache compartilhado. Parte barata.
- **`transform_indexed(row_idx)`** — inferência por fold contra o cache.

```python
# validate_pipeline:
tsc_model.prepare_cache(X_temporal)          # Catch22 1× em todas as linhas
...
# dentro do fold:
tsc_model.fit_indexed(train_idx, y_train)    # só retreina árvores (barato)
X_meta_train = tsc_model.transform_indexed(train_idx)
X_meta_test  = tsc_model.transform_indexed(test_idx)
```

### Por que não há vazamento
O cache guarda apenas features **por linha** derivadas dos dados brutos daquela linha (sem rótulos, sem estatísticas de fold). As árvores só treinam com `row_idx` do fold de treino via bootstrap. As linhas de teste nunca influenciam o treino.

### Ganho
Passa de ~`n_splits` passagens de Catch22 para **uma só**. A parte supervisionada (árvores, que liberam o GIL) continua por fold, mas é ordens de magnitude mais barata que a extração Catch22.

---

## 5. Paralelismo (joblib) — CIF

**Arquivo:** `tsc_models/canonical_interval_forest.py`

Dois pontos de paralelização:

**Construção do cache** — a extração Catch22 é dividida em chunks de linhas, um por worker de processo:

```python
n_chunks = max(1, min(effective_n_jobs(self.n_jobs), n))
chunks = [c for c in np.array_split(np.arange(n), n_chunks) if len(c) > 0]
parts = Parallel(n_jobs=self.n_jobs)(
    delayed(_interval_blocks_for_rows)(X_np[c], unique) for c in chunks
)
```
Cada worker recebe só sua fatia de linhas (barata de picklar) e computa todos os intervalos para elas.

**Treino das árvores** — paralelizado com backend de **threads** (não processos):

```python
self._trees = Parallel(n_jobs=self.n_jobs, prefer="threads")(
    delayed(self._fit_one_tree)(i, row_idx, y_np, n) for i in range(self.n_estimators)
)
```
Threads compartilham o cache grande sem picklá-lo, e o construtor de árvores do sklearn libera o GIL durante o fit — então escala bem em vários núcleos.

Além disso, dentro de `_interval_feature_block`, média/desvio/inclinação são vetorizados sobre as linhas (a inclinação por mínimos quadrados vira um `matmul`, equivalente a `np.polyfit` mas sem loop).

---

## 6. Treino em GPU — LSTM

**Arquivo:** `tsc_models/lstm_classifier.py`

### Técnicas combinadas

**Detecção automática de CUDA** — usa GPU quando disponível, sem configuração:
```python
self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
```

**Batch grande + escala de learning rate** — batch aumentado de 64 → 512 mantém a GPU ocupada (~4–6× mais rápido por época). Menos passos de gradiente por época seriam prejudiciais à convergência, então a LR é escalada pela **regra da raiz quadrada**:
```python
# batch 64->512 = 8x; lr 1e-3 * sqrt(8) ≈ 3e-3
batch_size: int = 512,
learning_rate: float = 3e-3,
```

**Transferência host→GPU sobreposta** — tensores ficam na CPU e são movidos por batch com `pin_memory` + `non_blocking`, permitindo que a cópia host→dispositivo se sobreponha à computação:
```python
loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=use_cuda)
...
x_batch = x_batch.to(self.device, non_blocking=use_cuda)
t_batch = t_batch.to(self.device, non_blocking=use_cuda)
```

### Ganho
~4–6× por época em GPU vs. batch pequeno. Para instalar torch com CUDA, ver seção de instalação do README.

---

## Como o pipeline escolhe o caminho

`validate_pipeline` (`core_tsc_xgboost_pipeline.py`) inspeciona as flags e ramifica:

1. `supports_global_transform` → **fit-once global** (Null, ROCKET, Shapelets).
2. senão `supports_indexed_fit` → **cache indexado** (CIF).
3. senão → caminho padrão (fit/transform por fold, ex.: LSTM).

Assim, cada modelo usa automaticamente a otimização mais forte que sua matemática permite, sem código condicional espalhado.

---

*Para benchmarks concretos do run mais recente, ver [`RESULTS.md`](RESULTS.md).*
