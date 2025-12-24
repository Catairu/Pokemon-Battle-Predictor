import gdown
import sys
from pathlib import Path
from zipfile import ZipFile

# ==============================================================================
# 1. CONFIGURATION AND PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
zip_path = DATA_DIR / "data.zip"
url = "https://drive.google.com/uc?id=1a8URhjNIWn_2syEsaq95J50nlLZl9IDL"

# We define this file to check if the dataset is already installed
sentinel_file = DATA_DIR / "train.jsonl"

# ==============================================================================
# 2. SCRIPT EXECUTION
# ==============================================================================

if sentinel_file.exists():
    print("Dataset found. Skipping setup.")
    sys.exit()

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download the dataset and extract the zip file contents
gdown.download(url, str(zip_path), quiet=False)
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall()

print("Extraction complete! Removing the zip file...")
zip_path.unlink()

print("Setup completed successfully!")
