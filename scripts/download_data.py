import os
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset from {ZIP_URL}...")
    ssl_context = ssl._create_unverified_context()
    req = urllib.request.Request(
        ZIP_URL, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, context=ssl_context) as response, open(ZIP_PATH, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download completed.")

    print("Extracting dataset...")
    temp_extract_dir = DATA_DIR / "temp_extract"
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    extracted_files = list(temp_extract_dir.rglob("*.csv"))
    for file_path in extracted_files:
        dest_path = DATA_DIR / file_path.name
        shutil.move(str(file_path), str(dest_path))
        print(f"Moved {file_path.name} to {DATA_DIR}")

    if ZIP_PATH.exists():
        os.remove(ZIP_PATH)
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print("Data download and extraction successful! Cleaned up temporary files.")

if __name__ == "__main__":
    download_and_extract()

