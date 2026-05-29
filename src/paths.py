# src/paths.py
from pathlib import Path

# 1. Get the absolute path of the directory containing THIS file
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Derive the Project Root directory (adjust .parent as needed)
# If paths.py is inside 'src/', its parent is the root.
PROJECT_ROOT = CURRENT_DIR.parent

# 3. Define all your project subdirectories relative to the root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 4. Define specific important file paths
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# (Optional) Automatically create the directories if they don't exist yet
def create_required_directories():
    """Safely creates the directory structure if it is missing."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # If you run this script directly, it will verify your paths and build them
    print(f"Project Root identified as: {PROJECT_ROOT}")
    create_required_directories()
    print("All project directories verified/created successfully.")