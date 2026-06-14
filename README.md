# Trabalho Mineração de Dados

Este repositório contém o trabalho de Mineração de Dados sobre a base **Spaceship Titanic**. O objetivo é prever a variável `Transported`, que indica se um passageiro foi transportado para outra dimensão.

O projeto está organizado em duas etapas principais:

1. [`notebooks/eda.ipynb`](notebooks/eda.ipynb): análise exploratória dos dados, valores ausentes, atributos compostos e relações com `Transported`.
2. [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb): engenharia de atributos, pré-processamento, otimização de hiperparâmetros, treinamento, validação cruzada e geração de artefatos.

A modelagem atual usa uma pipeline reprodutível com engenharia de atributos, imputação de valores ausentes, codificação de variáveis categóricas, validação cruzada estratificada com 10 folds e otimização de hiperparâmetros com Optuna.

## Estado Atual Do Projeto

O projeto compara três caminhos de modelagem:

| Caminho | Modelo | Função no projeto |
|---|---|---|
| A | Árvore de Decisão | Modelo interpretável baseado em regras. |
| B | Floresta Aleatória | Modelo principal, com melhor resultado atual. |
| C | Naive Bayes Gaussiano | Baseline probabilístico simples. |

Os melhores hiperparâmetros encontrados pelo Optuna estão salvos em [`models/params_path_a.json`](models/params_path_a.json), [`models/params_path_b.json`](models/params_path_b.json) e [`models/params_path_c.json`](models/params_path_c.json). Quando esses arquivos existem, o notebook reaproveita os parâmetros salvos em vez de executar a busca novamente.

Parâmetros versionados no estado atual:

| Caminho | Parâmetros |
|---|---|
| A | `max_depth=6`, `min_samples_split=43` |
| B | `n_estimators=216`, `min_samples_split=26`, `min_samples_leaf=11`, `max_features=None` |
| C | `var_smoothing=0.0009815405019274585` |

Na execução mais recente versionada em `reports/`, de `2026-06-05`, o ranking out-of-fold ficou:

| Posição | Caminho | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| 1 | Caminho B - Floresta Aleatória | 0.7988 | 0.8068 | 0.7896 | 0.7981 | 0.8866 |
| 2 | Caminho A - Árvore de Decisão | 0.7848 | 0.7681 | 0.8202 | 0.7933 | 0.8674 |
| 3 | Caminho C - Naive Bayes Gaussiano | 0.5809 | 0.5802 | 0.6071 | 0.5934 | 0.5876 |

Assim, o melhor caminho atual é o **Caminho B - Floresta Aleatória**.

## Requisitos

- [Python 3.12.x](https://www.python.org/)
- [Pip](https://pip.pypa.io/en/stable/)

## Instalação do ambiente virtual

No Linux, macOS ou Windows (PowerShell), execute o comando abaixo na raiz do projeto:

```bash
python3 -m venv .venv
```

Em seguida, ative o ambiente virtual:

- Linux / macOS:
  ```bash
  source .venv/bin/activate
  ```
- Windows (PowerShell):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- Windows (cmd):
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

> Se `python3` não estiver disponível, use `python` conforme a instalação do seu sistema.

## Instalar Dependências

Com o ambiente virtual ativado, instale os pacotes listados em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Entre as dependências principais estão `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib` e `optuna`.

## Executar O Projeto

Abra o Jupyter na raiz do projeto:

```bash
jupyter notebook
```

Depois, execute os notebooks nesta ordem:

1. [`notebooks/eda.ipynb`](notebooks/eda.ipynb): análise exploratória.
2. [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb): pré-processamento, Optuna, treinamento, avaliação e artefatos.

O notebook principal lê os dados de `data/`, usa os caminhos definidos em [`src/roots.py`](src/roots.py) e salva artefatos em `models/` e `reports/`.

## Documentação

- [`docs/eda.md`](docs/eda.md): explica a EDA e as hipóteses que orientam a modelagem.
- [`docs/notebook.md`](docs/notebook.md): explica o notebook principal, incluindo a otimização com Optuna e o benchmarking dos três caminhos.

## Estrutura Do Projeto

- `README.md` - documentação inicial do projeto.
- `requirements.txt` - dependências Python.
- `notebooks/` - notebooks de EDA e modelagem.
- `docs/` - explicações em Markdown dos notebooks.
- `src/roots.py` - caminhos centralizados do projeto.
- `data/` - dados de entrada:
  - `train.csv`;
  - `test.csv`;
  - `sample_submission.csv`.
- `models/` - arquivos `params_path_*.json` com os melhores hiperparâmetros versionados. O notebook também pode gerar modelos `.joblib` em execuções locais.
- `reports/` - métricas agregadas, históricos de validação cruzada e figuras.
- `reports/figures/` - curvas ROC em `.png` e `.csv`, além de visualizações de árvores em `.svg` para os caminhos baseados em árvore.

## Pipeline Atual De Modelagem

O notebook principal implementa o seguinte fluxo:

1. Carrega `data/train.csv` e `data/test.csv`.
2. Cria atributos derivados de `PassengerId`, `Cabin` e `Name`: `PassengerGroup`, `CabinDeck`, `CabinSide` e `FamilyName`.
3. Imputa `HomePlanet` usando regras por grupo de viagem, sobrenome e moda global do treino.
4. Imputa gastos individuais com `KNNImputer` antes de calcular `TotalSpend`.
5. Trata atributos numéricos, binários e categóricos com pipelines específicas.
6. Otimiza hiperparâmetros com Optuna, quando os JSONs de parâmetros não existem.
7. Avalia os modelos com `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
8. Salva métricas, histórico por fold, curvas ROC, parâmetros e modelos gerados pela execução.

As métricas calculadas são `accuracy`, `precision`, `recall`, `f1`, `f1_class_0`, `f1_class_1` e `roc_auc`.

## Artefatos Versionados

O repositório contém os seguintes tipos de artefatos já gerados:

- `reports/metrics_path_*_20260605_*.csv`: métricas agregadas mais recentes.
- `reports/history_path_*_20260605_*.csv`: métricas por fold mais recentes.
- `reports/figures/roc_path_*_20260605_*`: pontos e imagens das curvas ROC.
- `reports/figures/tree_path_*_20260605_*.svg`: visualizações de árvore para os
  caminhos A e B.
- `models/params_path_*.json`: melhores hiperparâmetros reaproveitados pelo notebook.

## Observações

- Mantenha o ambiente virtual ativado ao instalar pacotes ou ao executar os notebooks.
- O notebook de modelagem reaproveita `models/params_path_*.json` quando esses arquivos existem; se forem removidos, o Optuna roda novamente.
- Os modelos usam `random_state=42`, mas o sampler padrão do Optuna não recebe uma seed explícita. Se os JSONs forem apagados e a busca rodar de novo, os melhores parâmetros podem mudar.
- Os artefatos com timestamp em `reports/` representam execuções específicas do notebook.
- Arquivos `.joblib` de modelos treinados podem ser gerados localmente pelo notebook, mas não aparecem no estado versionado atual do repositório.

## Dupla Do Projeto

- Estevão Augusto da Fonseca Santos
- Gabriel Fagundes Mesquita Sousa
