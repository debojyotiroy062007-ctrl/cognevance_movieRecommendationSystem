import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def verify():
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"

    print("=== DATA VERIFICATION ===")
    
    if not ratings_path.exists() or not movies_path.exists():
        print(f"Error: Missing dataset files in {DATA_DIR}")
        return

    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    print("\n--- RATINGS DATASET ---")
    print(f"Total Rows: {len(ratings_df):,}")
    print(f"Columns: {list(ratings_df.columns)}")
    print("First 3 rows:")
    print(ratings_df.head(3).to_string(index=False))

    print("\n--- MOVIES DATASET ---")
    print(f"Total Rows: {len(movies_df):,}")
    print(f"Columns: {list(movies_df.columns)}")
    print("First 3 rows:")
    print(movies_df.head(3).to_string(index=False))

    print("\nSanity check passed successfully!")

if __name__ == "__main__":
    verify()
