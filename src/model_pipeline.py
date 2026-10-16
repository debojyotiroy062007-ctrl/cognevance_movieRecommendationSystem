import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, BaselineOnly, SVD
from surprise.model_selection import cross_validate

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_top_n_recommendations(user_id, model, ratings_df, movies_df, n=5):
    """
    Generate top-N movie recommendations for a user excluding already rated movies.
    """
    all_movie_ids = movies_df['movieId'].unique()
    rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    unrated_movie_ids = [m_id for m_id in all_movie_ids if m_id not in rated_movie_ids]
    
    predictions = []
    for m_id in unrated_movie_ids:
        pred = model.predict(uid=user_id, iid=m_id)
        predictions.append((m_id, pred.est))
    
    # Sort predictions by estimated rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_preds = predictions[:n]
    
    # Map to movie details
    recs = []
    movies_dict = movies_df.set_index('movieId').to_dict('index')
    for rank, (m_id, score) in enumerate(top_n_preds, 1):
        movie_info = movies_dict.get(m_id, {'title': 'Unknown', 'genres': 'Unknown'})
        recs.append({
            'rank': rank,
            'movieId': m_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'predicted_rating': round(score, 3)
        })
    return pd.DataFrame(recs)

def run_pipeline():
    print("=== STARTING MODEL BUILDING & EVALUATION PIPELINE ===")
    
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # Surprise reader and dataset setup
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # 1. Benchmark Evaluation (5-Fold Cross Validation)
    print("\n--- Running 5-Fold Cross Validation Benchmark ---")
    
    # Algorithm A: BaselineOnly
    print("Evaluating BaselineOnly algorithm...")
    algo_baseline = BaselineOnly(verbose=False)
    cv_baseline = cross_validate(algo_baseline, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    # Algorithm B: SVD (50 Latent Factors)
    print("Evaluating SVD algorithm (50 latent factors, lr_all=0.005, reg_all=0.02)...")
    algo_svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
    cv_svd = cross_validate(algo_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    # Summary benchmark dataframe
    benchmark_df = pd.DataFrame([
        {
            'Algorithm': 'BaselineOnly',
            'RMSE': round(float(np.mean(cv_baseline['test_rmse'])), 4),
            'MAE': round(float(np.mean(cv_baseline['test_mae'])), 4),
            'Fit Time (s)': round(float(np.mean(cv_baseline['fit_time'])), 4),
            'Test Time (s)': round(float(np.mean(cv_baseline['test_time'])), 4)
        },
        {
            'Algorithm': 'SVD (50 Latent Factors)',
            'RMSE': round(float(np.mean(cv_svd['test_rmse'])), 4),
            'MAE': round(float(np.mean(cv_svd['test_mae'])), 4),
            'Fit Time (s)': round(float(np.mean(cv_svd['fit_time'])), 4),
            'Test Time (s)': round(float(np.mean(cv_svd['test_time'])), 4)
        }
    ])
    
    benchmark_file = OUTPUT_DIR / "benchmark_results.csv"
    benchmark_df.to_csv(benchmark_file, index=False)
    print(f"\nSaved benchmark results to: {benchmark_file.name}")
    print("\n--- BENCHMARK RESULTS ---")
    print(benchmark_df.to_string(index=False))
    
    # 2. Full Model Training on 100k Dataset
    print("\n--- Training SVD Model on Full Dataset ---")
    full_trainset = data.build_full_trainset()
    svd_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
    svd_model.fit(full_trainset)
    
    model_pickle_path = SRC_DIR / "svd_model.pkl"
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(svd_model, f)
    print(f"Exported full SVD model to: {model_pickle_path.name}")
    
    # 3. Sample Inference for Users 1, 10, and 50
    sample_users = [1, 10, 50]
    sample_output_lines = []
    
    sample_output_lines.append("==================================================")
    sample_output_lines.append("      MOVIE RECOMMENDATION SYSTEM - SAMPLE OUTPUT")
    sample_output_lines.append("==================================================\n")
    
    for u_id in sample_users:
        recs_df = get_top_n_recommendations(u_id, svd_model, ratings_df, movies_df, n=5)
        
        user_header = f"--- TOP 5 RECOMMENDATIONS FOR USER ID {u_id} ---"
        sample_output_lines.append(user_header)
        for _, row in recs_df.iterrows():
            line = f"Rank {row['rank']}: [{row['movieId']}] {row['title']} | Genre: {row['genres']} | Predicted Rating: {row['predicted_rating']}"
            sample_output_lines.append(line)
        sample_output_lines.append("")
    
    formatted_recs_str = "\n".join(sample_output_lines)
    
    recommendations_file = OUTPUT_DIR / "sample_recommendations.txt"
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        f.write(formatted_recs_str)
    print(f"Saved sample recommendations to: {recommendations_file.name}")
    
    print("\nModel pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
