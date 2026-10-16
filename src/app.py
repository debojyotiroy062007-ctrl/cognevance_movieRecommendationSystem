import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"
VIS_DIR = OUTPUT_DIR / "visualizations"

# Page Configuration
st.set_page_config(
    page_title="CineMatch - Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Header Gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8c42 50%, #f9c74f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #90a4ae;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards & Containers */
    .movie-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 140, 66, 0.4);
    }
    
    .rating-badge {
        background-color: #ff8c42;
        color: #111;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.9rem;
        float: right;
    }

    .genre-badge {
        background: rgba(255, 255, 255, 0.1);
        color: #b0bec5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 4px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        color: #b0bec5;
        font-weight: 600;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8c42 100%) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model Loaders
@st.cache_data
def load_data():
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
    return movies_df, ratings_df

@st.cache_resource
def load_model():
    model_path = SRC_DIR / "svd_model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please run model pipeline first.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def compute_content_similarity(movies_df):
    movies_df_clean = movies_df.copy()
    movies_df_clean['genres_clean'] = movies_df_clean['genres'].str.replace('|', ' ', regex=False)
    movies_df_clean['genres_clean'] = movies_df_clean['genres_clean'].replace('(no genres listed)', '')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df_clean['genres_clean'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim, movies_df_clean

# Load resources
movies_df, ratings_df = load_data()
svd_model = load_model()
cosine_sim, movies_clean = compute_content_similarity(movies_df)

# Header
st.markdown('<div class="main-header">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industry-Grade AI-Powered Movie Recommendation System</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 User Collaborative Recommendations", 
    "🔍 Content & Similar Movie Finder", 
    "📊 Analytics & Insights Dashboard"
])

# ---------------------------------------------------------
# TAB 1: User-Based Collaborative Filtering
# ---------------------------------------------------------
with tab1:
    st.subheader("Personalized Movie Recommendations (SVD Collaborative Filtering)")
    st.write("Select a User ID to inspect their past movie ratings and view personalized AI-generated movie recommendations.")
    
    col_user, col_slider = st.columns([1, 2])
    
    all_users = sorted(ratings_df['userId'].unique())
    with col_user:
        selected_user_id = st.selectbox("Select User ID:", options=all_users, index=0)
    with col_slider:
        top_n = st.slider("Number of Recommendations:", min_value=5, max_value=15, value=5, step=1)
    
    col_history, col_recs = st.columns([1, 1])
    
    # User Rating History
    with col_history:
        st.markdown("### 🌟 User's Top Rated Movies")
        user_ratings = ratings_df[ratings_df['userId'] == selected_user_id]
        top_history = user_ratings.sort_values(by=['rating', 'timestamp'], ascending=[False, False]).head(5)
        top_history = top_history.merge(movies_df, on='movieId')
        
        for _, row in top_history.iterrows():
            genres_formatted = " ".join([f'<span class="genre-badge">{g}</span>' for g in row['genres'].split('|')])
            st.markdown(f"""
            <div class="movie-card">
                <span class="rating-badge">★ {row['rating']:.1f}</span>
                <div style="font-weight:700; font-size:1.05rem; color:#fff;">{row['title']}</div>
                <div style="margin-top:6px;">{genres_formatted}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # Model Recommendations
    with col_recs:
        st.markdown(f"### 🎯 Top {top_n} Recommended Movies for User {selected_user_id}")
        
        all_movie_ids = movies_df['movieId'].unique()
        rated_movie_ids = set(user_ratings['movieId'])
        unrated_movie_ids = [m_id for m_id in all_movie_ids if m_id not in rated_movie_ids]
        
        predictions = []
        for m_id in unrated_movie_ids:
            pred = svd_model.predict(uid=selected_user_id, iid=m_id)
            predictions.append((m_id, pred.est))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]
        
        movies_dict = movies_df.set_index('movieId').to_dict('index')
        
        for rank, (m_id, est_rating) in enumerate(top_predictions, 1):
            info = movies_dict.get(m_id, {'title': 'Unknown', 'genres': 'Unknown'})
            genres_formatted = " ".join([f'<span class="genre-badge">{g}</span>' for g in info['genres'].split('|')])
            st.markdown(f"""
            <div class="movie-card">
                <span class="rating-badge">Predicted ★ {est_rating:.2f}</span>
                <div style="font-weight:700; font-size:1.05rem; color:#fff;">#{rank}. {info['title']}</div>
                <div style="margin-top:6px;">{genres_formatted}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Content-Based Similar Movie Finder
# ---------------------------------------------------------
with tab2:
    st.subheader("Genre-Based Content Similarity Search")
    st.write("Find movies similar to any chosen title based on TF-IDF genre vector representations. Ideal for new users or cold-start scenarios.")
    
    movie_list = movies_df['title'].sort_values().tolist()
    selected_movie_title = st.selectbox("Search / Select a Movie Title:", options=movie_list, index=0)
    
    if selected_movie_title:
        idx_match = movies_df[movies_df['title'] == selected_movie_title].index
        if len(idx_match) > 0:
            idx = idx_match[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Exclude the movie itself
            similar_indices = [i for i, score in sim_scores if i != idx][:5]
            
            selected_info = movies_df.iloc[idx]
            st.info(f"**Selected Movie:** {selected_info['title']}  \n**Genres:** {selected_info['genres']}")
            
            st.markdown("### 🍿 Top 5 Similar Movies")
            col_sim1, col_sim2 = st.columns(2)
            
            for rank, sim_idx in enumerate(similar_indices, 1):
                sim_movie = movies_df.iloc[sim_idx]
                score = sim_scores[[i for i, s in sim_scores].index(sim_idx)][1]
                genres_formatted = " ".join([f'<span class="genre-badge">{g}</span>' for g in sim_movie['genres'].split('|')])
                
                target_col = col_sim1 if rank % 2 != 0 else col_sim2
                with target_col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <span class="rating-badge">Sim: {score*100:.0f}%</span>
                        <div style="font-weight:700; font-size:1.05rem; color:#fff;">#{rank}. {sim_movie['title']}</div>
                        <div style="margin-top:6px;">{genres_formatted}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Analytics Dashboard
# ---------------------------------------------------------
with tab3:
    st.subheader("Exploratory Data Analysis & System Analytics")
    st.write("Comprehensive dataset statistics and visual distributions generated during EDA.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ratings", f"{len(ratings_df):,}")
    c2.metric("Mean Rating", f"{ratings_df['rating'].mean():.2f} ★")
    c3.metric("Total Unique Users", f"{ratings_df['userId'].nunique():,}")
    c4.metric("Catalog Sparsity", f"{(1 - len(ratings_df)/(ratings_df['userId'].nunique()*movies_df['movieId'].nunique()))*100:.2f}%")
    
    st.markdown("---")
    
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.markdown("#### Rating Distribution")
        img1_path = VIS_DIR / "rating_distribution.png"
        if img1_path.exists():
            st.image(str(img1_path), use_container_width=True)
            
        st.markdown("#### Genre Distribution across Catalog")
        img3_path = VIS_DIR / "genre_distribution.png"
        if img3_path.exists():
            st.image(str(img3_path), use_container_width=True)
            
    with col_vis2:
        st.markdown("#### Top 15 Most-Rated Movies")
        img2_path = VIS_DIR / "top_15_most_rated_movies.png"
        if img2_path.exists():
            st.image(str(img2_path), use_container_width=True)
            
        st.markdown("#### User Activity Long-Tail Distribution")
        img4_path = VIS_DIR / "user_activity_distribution.png"
        if img4_path.exists():
            st.image(str(img4_path), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #78909c; font-size: 0.9rem;'>CineMatch Recommendation Engine • Built with Streamlit, Scikit-Surprise & Scikit-Learn</div>", unsafe_allow_html=True)
