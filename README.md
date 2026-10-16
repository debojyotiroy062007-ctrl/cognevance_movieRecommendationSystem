# CineMatch – Intelligent Movie Recommendation System

An end-to-end Machine Learning recommendation system built using collaborative filtering (Matrix Factorization via SVD) and content-aware NLP vectorization to predict user ratings and suggest top-N movies. The project includes automated exploratory data analysis (EDA), model benchmarking, and an interactive Streamlit web dashboard.

---

## 📌 Project Overview

This project implements a personalized movie recommender using the MovieLens dataset. It addresses user preference modeling through matrix decomposition, benchmarks performance against baseline statistical models, and serves dynamic recommendations through a modern web UI.

### Key Capabilities
- **Collaborative Filtering:** Matrix Factorization utilizing Singular Value Decomposition (`SVD`) to learn latent user and item factors.
- **Statistical Baselines:** Benchmark evaluation against `BaselineOnly` models using cross-validation.
- **Content Similarity:** Text vectorization via `TfidfVectorizer` and linear kernel metrics for item-level content associations.
- **Automated EDA:** Distribution analysis of user ratings, sparsity matrices, and genre frequency visualization.
- **Interactive UI:** Real-time query matching, top-N recommendation generation, and score prediction via Streamlit.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Recommendation & Machine Learning:** `scikit-surprise`, `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Frontend / Deployment:** `streamlit`
- **Persistence & Utilities:** `pickle`, `pathlib`, `os`

---

## 📂 Project Structure

```text
movie_recommendation_system/
├── data/
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── notebooks/
├── outputs/
│   ├── visualizations/
│   ├── benchmark_results.csv
│   └── sample_recommendations.txt
├── report/
├── scripts/
│   ├── download_data.py
│   ├── verify_data.py
│   └── verify_pipeline.py
├── src/
│   ├── app.py                  # Streamlit web application
│   ├── eda.py                  # Exploratory Data Analysis routines
│   ├── model_pipeline.py       # Model training, cross-validation & SVD persistence
│   └── svd_model.pkl           # Serialized trained model
├── requirements.txt
└── README.md
