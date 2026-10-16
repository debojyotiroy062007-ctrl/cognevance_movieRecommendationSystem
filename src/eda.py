import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 18
})

def run_eda():
    print("=== STARTING EXPLORATORY DATA ANALYSIS ===")
    
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    total_ratings = len(ratings_df)
    mean_rating = ratings_df['rating'].mean()
    total_users = ratings_df['userId'].nunique()
    total_movies = movies_df['movieId'].nunique()
    rated_movies_count = ratings_df['movieId'].nunique()
    
    sparsity = (1.0 - (total_ratings / (total_users * total_movies))) * 100.0
    
    print(f"\n--- DATASET SUMMARY STATISTICS ---")
    print(f"Total Ratings: {total_ratings:,}")
    print(f"Mean Rating: {mean_rating:.3f}")
    print(f"Total Unique Users: {total_users:,}")
    print(f"Total Unique Movies (in catalog): {total_movies:,}")
    print(f"Total Rated Movies: {rated_movies_count:,}")
    print(f"User-Item Matrix Sparsity: {sparsity:.2f}%\n")
    
    # 1. Rating Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=ratings_df, x='rating', hue='rating', palette='crest', legend=False, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Distribution of User Ratings", pad=15, fontweight='bold')
    ax.set_xlabel("Rating Value")
    ax.set_ylabel("Count")
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    chart1_path = OUTPUT_DIR / "rating_distribution.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart1_path.name}")
    
    # 2. Top 15 Most Rated Movies
    merged_df = ratings_df.merge(movies_df, on='movieId')
    top_15_movies = merged_df['title'].value_counts().head(15).reset_index()
    top_15_movies.columns = ['title', 'rating_count']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_15_movies, y='title', x='rating_count', hue='title', palette='viridis', legend=False, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Top 15 Most-Rated Movies", pad=15, fontweight='bold')
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("")
    for p in ax.patches:
        width = int(p.get_width())
        if width > 0:
            ax.annotate(f'{width:,}', (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
    plt.tight_layout()
    chart2_path = OUTPUT_DIR / "top_15_most_rated_movies.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart2_path.name}")
    
    # 3. Genre Distribution
    movies_genres = movies_df.copy()
    movies_genres['genre_list'] = movies_genres['genres'].str.split('|')
    exploded_genres = movies_genres.explode('genre_list')
    exploded_genres = exploded_genres[exploded_genres['genre_list'] != '(no genres listed)']
    genre_counts = exploded_genres['genre_list'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=genre_counts, x='genre', y='count', hue='genre', palette='magma', legend=False, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Movie Count Across Individual Genres", pad=15, fontweight='bold')
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Movies")
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart3_path = OUTPUT_DIR / "genre_distribution.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart3_path.name}")
    
    # 4. User Activity Distribution (Log Scale)
    user_counts = ratings_df.groupby('userId').size().reset_index(name='rating_count')
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(user_counts['rating_count'], bins=50, log_scale=True, kde=True, color='teal', ax=ax, edgecolor='black')
    ax.set_title("User Activity Distribution (Log Scale)", pad=15, fontweight='bold')
    ax.set_xlabel("Number of Ratings per User (Log Scale)")
    ax.set_ylabel("Number of Users")
    plt.tight_layout()
    chart4_path = OUTPUT_DIR / "user_activity_distribution.png"
    plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart4_path.name}")
    
    print("\nEDA completed successfully.")

if __name__ == "__main__":
    run_eda()
