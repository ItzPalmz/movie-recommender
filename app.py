import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from streamlit_lottie import st_lottie
import json

# Function to load Lottie animation JSON
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load dataset
netflix_overall = pd.read_csv("netflix_titles.csv")
netflix_overall['description'] = netflix_overall['description'].fillna('')
netflix_overall['normalized_title'] = netflix_overall['title'].apply(lambda x: x.replace(" ", "").lower())

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(netflix_overall.index, index=netflix_overall['normalized_title']).drop_duplicates()

# recommendation function
def get_recommendations_new(title, cosine_sim=cosine_sim):
    title = title.replace(' ', '').lower()
    if title not in indices:
        return ["Title not found. Please try another."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_overall['title'].iloc[movie_indices].tolist()

st.set_page_config(page_title="Netflix Recommendations", layout="centered")
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, .css-10trblm, .css-qri22k, .stTextInput>div>div>input {
            color: white !important;
        }
        .css-1cpxqw2, .css-1d391kg, .css-1v0mbdj, .stSelectbox label {
            color: white !important;
        }
        .css-1wa3eu0, .stButton>button {
            color: black !important;
        }
        /* Make sidebar title "Navigation" black */
        section[data-testid="stSidebar"] h1 {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and animation
st.title("Netflix Movie and TV Show Recommendation")
lottie_coding = load_lottiefile("netflix-logo.json")
st_lottie(lottie_coding, speed=1, reverse=False, loop=True, quality="low", height=220)

# Sidebar
with st.sidebar:
    st.info("""
        This is a Netflix content analysis and recommendation system.
        - Get movie recommendations based on content similarity.
    """)

# Movie selection
movie_list = netflix_overall['title'].tolist()
selected_movie = st.selectbox("Select a movie or TV show", movie_list)

# Button for recommendations
if st.button('Get Recommendations'):
    recommended_movie_names = get_recommendations_new(selected_movie)
    st.subheader("Top 10 Recommended Movies/TV Shows")
    for i, movie in enumerate(recommended_movie_names, start=1):
        st.write(f"{i}. {movie}")
