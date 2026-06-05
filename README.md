# Trabalho Mineração de Dados

Este projeto contém uma análise e modelagem para a base Spaceship Titanic. A análise exploratória está em [`notebooks/eda.ipynb`](notebooks/eda.ipynb), e o notebook principal de modelagem está em [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb).

A modelagem usa uma pipeline de pré-processamento com engenharia de atributos, imputação de valores ausentes, validação cruzada estratificada e otimização de hiperparâmetros com Optuna.

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
- `models/` - modelos treinados e `params_path_*.json` com os melhores hiperparâmetros.
- `reports/` - métricas, históricos de validação cruzada e figuras.

## Observações

- Mantenha o ambiente virtual ativado ao instalar pacotes ou ao executar os notebooks.
- O notebook de modelagem reaproveita `models/params_path_*.json` quando esses arquivos existem; se forem removidos, o Optuna roda novamente.
- Os artefatos com timestamp em `models/` e `reports/` representam execuções específicas do notebook.

## Dupla Do Projeto

- Estevão Augusto da Fonseca Santos
- Gabriel Fagundes Mesquita Sousa
