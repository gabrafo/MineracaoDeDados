# Trabalho Mineração de Dados

Este projeto contém uma análise e modelagem de dados para uma tarefa de mineração de dados. O código principal está organizado em um notebook Jupyter (`notebook.ipynb`) e os dados de treino/teste estão na pasta `data/`.

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

## Instalar dependências

Com o ambiente virtual ativado, instale os pacotes listados em `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Executar o projeto

O notebook principal é `notebook.ipynb`.

Se você tiver Jupyter instalado, abra-o com:

```bash
jupyter notebook
```

Depois, carregue `notebook.ipynb` no navegador.

## Estrutura do projeto

- `README.md` — documentação do projeto
- `requirements.txt` — dependências Python
- `notebook.ipynb` — análise e experimentos
- `data/` — dados de entrada
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`
- `submission.csv` — possível arquivo de submissão gerado
- `insights.md` — anotações e conclusões do trabalho

## Observações

- Mantenha o ambiente virtual ativado ao instalar pacotes ou ao executar o notebook.
- Se desejar criar um ambiente virtual em outra pasta, substitua `.venv` pelo nome desejado.

## Dupla do Projeto

- Estevão Augusto da Fonseca Santos
- Gabriel Fagundes Mesquita Sousa