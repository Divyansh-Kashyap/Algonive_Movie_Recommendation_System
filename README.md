# 🎬 Movie Recommendation System

This project is a **hybrid Movie Recommendation System** that suggests films based on user preferences using both **Content-Based Filtering** and **Collaborative Filtering**. It includes:

- A well-structured **Jupyter Notebook** for model exploration and training
- A fully interactive **Streamlit Web App**
- **TMDb API integration** for rich movie metadata like:
  - Posters
  - Trailers
  - Overview
  - Release date
  - IMDb links
- A simple **user rating interface** to collect feedback on recommendations

---

## 🧠 Recommendation Techniques Used

### 1. 🎯 Content-Based Filtering
- Uses TF-IDF Vectorization on movie genres
- Measures cosine similarity between movies
- Recommends movies similar to the one selected

### 2. 👥 Collaborative Filtering
- Based on user rating behavior
- Implements a **K-Nearest Neighbors (KNN)** model on a user-item rating matrix
- Finds similar users and recommends highly rated movies

---

## 🛠 Tech Stack

| Component         | Tools Used                                  |
|------------------|----------------------------------------------|
| Language         | Python 3                                     |
| Libraries        | `pandas`, `numpy`, `scikit-learn`, `requests`, `streamlit` |
| Data Source      | [MovieLens Dataset (1M)](https://grouplens.org/datasets/movielens/) |
| External APIs    | [TMDb API](https://www.themoviedb.org/documentation/api) |

---

## 📁 Project Structure
### 📦 Movie-Recommender-System/
- app.py # Streamlit web app
- Movie_Recommender.ipynb # Jupyter notebook (model building)
- movies.csv # Sample movie metadata
- ratings.csv # Sample user ratings
- user_feedback.csv # Logs user ratings from the app
- README.md # Project documentation
