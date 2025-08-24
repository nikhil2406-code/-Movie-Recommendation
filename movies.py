import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
movies = pd.read_csv('movies.csv.gz', compression='gzip')
credits = pd.read_csv('credits.csv.gz', compression='gzip')




movies = movies.merge(credits, left_on='title', right_on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# --- Data cleaning functions ---
def extract_names(obj):
    return [item['name'] for item in ast.literal_eval(obj)]

def extract_top_cast(obj):
    return [item['name'] for item in ast.literal_eval(obj)[:3]]

def extract_director(obj):
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            return [item['name']]
    return []

def clean_names(lst):
    return [i.replace(" ", "").lower() for i in lst]

# --- Apply cleaning ---
movies['genres'] = movies['genres'].apply(extract_names).apply(clean_names)
movies['keywords'] = movies['keywords'].apply(extract_names).apply(clean_names)
movies['cast'] = movies['cast'].apply(extract_top_cast).apply(clean_names)
movies['crew'] = movies['crew'].apply(extract_director).apply(clean_names)
movies['overview'] = movies['overview'].apply(lambda x: x.lower().split())

# Create tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new_df = movies[['movie_id', 'title', 'tags']]

# --- Text Vectorization ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# --- Stemming ---
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

# --- Cosine Similarity ---
similarity = cosine_similarity(vectors)

# --- Recommendation Function ---
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return []  # return empty list if movie not found

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

# --- Streamlit App ---
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie", new_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.write("### Recommended Movies:")
        for i in recommendations:
            st.write(f"🎥 {i}")
    else:
        st.warning("Movie not found in dataset!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2025 Nikhil Sriramoju. All rights reserved.  cc:theMoviedb.org </div>",
    unsafe_allow_html=True
)
