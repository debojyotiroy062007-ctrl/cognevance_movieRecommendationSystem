from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "outputs" / "visualizations"
BENCHMARK_FILE = BASE_DIR / "outputs" / "benchmark_results.csv"
SAMPLE_RECS_FILE = BASE_DIR / "outputs" / "sample_recommendations.txt"
MODEL_FILE = BASE_DIR / "src" / "svd_model.pkl"

def verify():
    print("=== VERIFYING PIPELINE OUTPUTS ===")
    
    expected_charts = [
        "rating_distribution.png",
        "top_15_most_rated_movies.png",
        "genre_distribution.png",
        "user_activity_distribution.png"
    ]
    
    print("\n--- Visualizations Check ---")
    all_exist = True
    for chart in expected_charts:
        chart_path = VIS_DIR / chart
        exists = chart_path.exists()
        size_kb = chart_path.stat().st_size / 1024 if exists else 0
        print(f"File: {chart} | Exists: {exists} | Size: {size_kb:.1f} KB")
        if not exists:
            all_exist = False

    print(f"\n--- Model File Check ---")
    print(f"File: {MODEL_FILE.name} | Exists: {MODEL_FILE.exists()} | Size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    print("\n--- BENCHMARK RESULTS TABLE ---")
    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE, 'r') as f:
            print(f.read())
    else:
        print("Error: benchmark_results.csv not found!")

    print("--- SAMPLE RECOMMENDATIONS FOR USER 1 ---")
    if SAMPLE_RECS_FILE.exists():
        with open(SAMPLE_RECS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            user_1_lines = []
            capture = False
            for line in lines:
                if "FOR USER ID 1" in line:
                    capture = True
                elif "FOR USER ID 10" in line:
                    capture = False
                if capture and line.strip():
                    user_1_lines.append(line)
            print("\n".join(user_1_lines))
    else:
        print("Error: sample_recommendations.txt not found!")

    if all_exist and BENCHMARK_FILE.exists() and SAMPLE_RECS_FILE.exists() and MODEL_FILE.exists():
        print("\nAll verification checks PASSED successfully!")

if __name__ == "__main__":
    verify()
