import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"   # <-- replace

# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Netflix Style Movie Recommender", layout="wide")

# ------------------ NETFLIX CSS ------------------

st.markdown("""
<style>
body {
    background-color: #141414;
    color: white;
}
.main {
    background-color: #141414;
}
h1, h2, h3, h4 {
    color: #E50914;
    font-weight: 700;
}
.movie-card img {
    border-radius: 12px;
    transition: transform 0.3s ease;
}
.movie-card img:hover {
    transform: scale(1.08);
}
.movie-title {
    text-align: center;
    font-weight: 600;
    margin-top: 5px;
}
.rating {
    text-align: center;
    color: gold;
}
.trailer {
    text-align: center;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ DATA PREPROCESS ------------------

def extract_name(text):
    return [i['name'] for i in ast.literal_eval(text)]

def extract_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def extract_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on='title')

    movies['genres'] = movies['genres'].apply(extract_name)
    movies['keywords'] = movies['keywords'].apply(extract_name)
    movies['cast'] = movies['cast'].apply(extract_cast)
    movies['crew'] = movies['crew'].apply(extract_director)

    movies['overview'] = movies['overview'].fillna("").apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    return movies[['id', 'title', 'tags']]

movies = load_data()

# ------------------ VECTORIZE ------------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ------------------ TMDB API ------------------

def fetch_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()

        poster = "https://via.placeholder.com/500x750?text=No+Poster"
        rating = "N/A"

        if data.get('poster_path'):
            poster = "https://image.tmdb.org/t/p/w500" + data['poster_path']
        if data.get('vote_average'):
            rating = round(data['vote_average'], 1)

        return poster, rating
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster", "N/A"


def fetch_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
        data = requests.get(url, timeout=5).json()

        for v in data.get("results", []):
            if v['type'] == "Trailer" and v['site'] == "YouTube":
                return "https://www.youtube.com/watch?v=" + v['key']
    except:
        pass
    return None


def movie_page(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"

# ------------------ RECOMMEND ------------------

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names, posters, ratings, trailers, links = [], [], [], [], []

    for i in movies_list:
        row = movies.iloc[i[0]]

        poster, rating = fetch_details(row.id)

        names.append(row.title)
        posters.append(poster)
        ratings.append(rating)
        trailers.append(fetch_trailer(row.id))
        links.append(movie_page(row.id))

    return names, posters, ratings, trailers, links

# ------------------ UI ------------------

st.markdown("<h1>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h4>Netflix Style AI Recommendation System</h4>", unsafe_allow_html=True)

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters, ratings, trailers, links = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <a href="{links[i]}" target="_blank">
                        <img src="{posters[i]}" width="100%">
                    </a>
                    <div class="movie-title">{names[i]}</div>
                    <div class="rating">⭐ {ratings[i]}/10</div>
                    <div class="trailer">
                        <a href="{trailers[i]}" target="_blank" style="color:#E50914">▶ Watch Trailer</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
