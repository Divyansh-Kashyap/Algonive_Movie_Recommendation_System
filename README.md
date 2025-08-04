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

```
📦 Movie-Recommender-System/
├── app.py                  # Streamlit web app
├── Movie_Recommender.ipynb # Jupyter notebook (model building)
├── movies.csv              # Sample movie metadata
├── ratings.csv             # Sample user ratings
├── user_feedback.csv       # Logs user ratings from the app
├── README.md               # Project documentation
```

---


## 🚀 Run the Streamlit Web App

```bash
streamlit run app.py
```

> ⚠️ You'll be prompted to enter your **TMDb API key** at runtime.

To get a TMDb API key:
- Register at [TMDb](https://www.themoviedb.org/signup)
- Go to your [API settings](https://www.themoviedb.org/settings/api) and create a developer key

---

## 📓 Explore the Jupyter Notebook

Open `Movie_Recommender.ipynb` to:
- Understand the logic behind both filtering methods
- Visualize similarities and performance
- Customize recommendations

Use:
```bash
jupyter notebook Movie_Recommendation_System.ipynb
```

---

## 🌟 Features

- ✅ TF-IDF-based genre similarity
- ✅ KNN-based user similarity
- ✅ TMDb integration:
  - Movie posters
  - Overviews and ratings
  - YouTube trailer links
  - IMDb page links
- ✅ User interface for feedback
- ✅ Feedback saved in `user_feedback.csv`

---



## 🔒 Privacy & Notes

- This app doesn't store any personal info or online activity
- Ratings provided in the app are saved locally for demo/logging purposes
- Replace `movies.csv` and `ratings.csv` with the full MovieLens 1M data for real-world performance

---

## 📌 Future Improvements

- Use deep learning for hybrid recommendations (e.g., Neural Collaborative Filtering)
- Add user registration and persistent feedback
- Integrate IMDb or TMDb login for personalization
- Host publicly on Streamlit Cloud or Hugging Face Spaces

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 🧑‍💻 Author

**Divyansh Kashyap**  
_BTech AIML | Machine Learning Intern | Open to Collaborations_  
📧 [LinkedIn]((https://www.linkedin.com/in/divyansh-kashyap-231270301/))

---


