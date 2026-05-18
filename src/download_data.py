from pathlib import Path
import zipfile
import requests

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DATA_DIR / "bank_marketing.zip"
CSV_PATH = RAW_DATA_DIR / "bank-full.csv"


def download_dataset() -> Path:
    """Download and extract the UCI Bank Marketing dataset if it is missing."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        print(f"Dataset already exists: {CSV_PATH}")
        return CSV_PATH

    print("Downloading dataset from UCI...")
    response = requests.get(UCI_ZIP_URL, timeout=60)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as outer_zip:
        outer_zip.extractall(RAW_DATA_DIR)

    inner_zip_path = RAW_DATA_DIR / "bank.zip"
    if inner_zip_path.exists():
        with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
            inner_zip.extractall(RAW_DATA_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError("bank-full.csv was not found after extraction.")

    print(f"Dataset ready: {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    download_dataset()
