import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------ Load Data ------------------------
@st.cache_data

def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Extract year from title if not available separately
if 'Year' not in movies.columns:
    movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)

st.title("🎬 Movie Recommendation System")
st.markdown("This app recommends movies using both **content-based** and **collaborative filtering**.")

# ------------------------ Sidebar Filters ------------------------
st.sidebar.header("Filters")
rec_type = st.sidebar.radio("Select Recommendation Type:", ["Content-Based", "Collaborative"])

all_genres = sorted(set(g for sublist in movies['Genres'].dropna().str.split('|') for g in sublist))
selected_genre = st.sidebar.selectbox("🎭 Genre", ["All"] + all_genres)

all_years = sorted(movies['Year'].unique())
selected_year = st.sidebar.selectbox("📅 Release Year", ["All"] + [str(y) for y in all_years if y != 0])

min_rating, max_rating = st.sidebar.slider("⭐ Average Rating Range (Internal)", 0.0, 5.0, (2.5, 5.0), 0.1)

# ------------------------ TF-IDF Model ------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Genres'].fillna(''))
model_knn_content = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_content.fit(tfidf_matrix)
indices = pd.Series(movies.index, index=movies['Title'])

# ------------------------ Collaborative Model ------------------------
user_movie_ratings = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
model_knn_collab = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_collab.fit(user_movie_ratings.values)

# ------------------------ TMDb API ------------------------
TMDB_API_KEY = "41ca3b314524a6f213a1165504f6c11a"

def fetch_tmdb_data(title):
    title = title.split(' (')[0]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    if data['results']:
        movie = data['results'][0]
        poster_path = movie.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        overview = movie.get('overview', '')
        release_date = movie.get('release_date', '')
        rating = movie.get('vote_average', 'N/A')
        return {
            "poster_url": poster_url,
            "overview": overview,
            "release_date": release_date,
            "rating": rating
        }
    return None

# ------------------------ Recommendation Functions ------------------------
def content_recommendations(title, num=10):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    distances, neighbors = model_knn_content.kneighbors(tfidf_matrix[idx], n_neighbors=num+1)
    movie_indices = neighbors.flatten()[1:]
    return movies.iloc[movie_indices]

def collaborative_recommendations(user_id, num=10):
    if user_id not in user_movie_ratings.index:
        return pd.DataFrame()
    distances, indices_ = model_knn_collab.kneighbors([user_movie_ratings.loc[user_id]], n_neighbors=num+1)
    neighbors = indices_.flatten()[1:]
    neighbor_ratings = ratings[ratings['UserID'].isin(user_movie_ratings.index[neighbors])]
    top_movies = (neighbor_ratings.groupby('MovieID')
                                .mean()['Rating']
                                .sort_values(ascending=False)
                                .head(num))
    return movies[movies['MovieID'].isin(top_movies.index)]

# ------------------------ Display Function ------------------------
def display_movie_info(row):
    title = row['Title']
    info = fetch_tmdb_data(title)
    st.markdown(f"### 🎬 {title}")
    poster_url = info.get("poster_url") if info else None
    st.image(poster_url or "https://via.placeholder.com/300x450?text=No+Image")

    if info:
        st.write(f"**TMDb Rating**: {info.get('rating', 'N/A')}")
        st.write(f"**Release Date**: {info.get('release_date', 'N/A')}")
        st.write(f"**Overview**: {info.get('overview', 'No description available.')}")
    else:
        st.write("ℹ️ No extra info available from TMDb.")

# ------------------------ User Interface ------------------------
if rec_type == "Content-Based":
    selected_movie = st.selectbox("🎞️ Select a movie", sorted(movies['Title'].unique()))
    if st.button("Get Recommendations"):
        recommended_df = content_recommendations(selected_movie)

        # Filter by genre, year, rating
        if selected_genre != "All":
            recommended_df = recommended_df[recommended_df['Genres'].str.contains(selected_genre, na=False)]
        if selected_year != "All":
            recommended_df = recommended_df[recommended_df['Year'] == int(selected_year)]

        avg_ratings = ratings.groupby('MovieID')['Rating'].mean().reset_index()
        recommended_df = pd.merge(recommended_df, avg_ratings, on='MovieID', how='left')
        recommended_df = recommended_df[(recommended_df['Rating'] >= min_rating) & (recommended_df['Rating'] <= max_rating)]

        if recommended_df.empty:
            st.warning("No recommendations match the selected filters.")
        else:
            for _, row in recommended_df.iterrows():
                display_movie_info(row)

elif rec_type == "Collaborative":
    user_ids = sorted(user_movie_ratings.index.tolist())
    selected_user = st.selectbox("👤 Select User ID", user_ids)
    if st.button("Get Recommendations"):
        recommended_df = collaborative_recommendations(selected_user)

        if selected_genre != "All":
            recommended_df = recommended_df[recommended_df['Genres'].str.contains(selected_genre, na=False)]
        if selected_year != "All":
            recommended_df = recommended_df[recommended_df['Year'] == int(selected_year)]

        avg_ratings = ratings.groupby('MovieID')['Rating'].mean().reset_index()
        recommended_df = pd.merge(recommended_df, avg_ratings, on='MovieID', how='left')
        recommended_df = recommended_df[(recommended_df['Rating'] >= min_rating) & (recommended_df['Rating'] <= max_rating)]

        if recommended_df.empty:
            st.warning("No recommendations match the selected filters.")
        else:
            for _, row in recommended_df.iterrows():
                display_movie_info(row)
