from pathlib import Path

# Diretório que contém este arquivo.
CURRENT_DIR = Path(__file__).resolve().parent

# Como este módulo fica em src/, a raiz do projeto é o diretório pai.
PROJECT_ROOT = CURRENT_DIR.parent

# Diretórios principais do projeto.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Arquivos de dados usados nos notebooks.
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Arquivo opcional de configuração do projeto.
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def create_required_directories():
    """Cria com segurança os diretórios de trabalho quando ainda não existem."""
    required_dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Raiz do projeto identificada em: {PROJECT_ROOT}")
    create_required_directories()
    print("Diretórios principais verificados/criados com sucesso.")
