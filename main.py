from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import requests
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()


movies = pickle.load(open("genre.pkl", "rb"))
g_movie = movies[["movie_id", "title", "genres"]]
g_movie = g_movie.reset_index(drop=True)

ps = PorterStemmer()
cv = CountVectorizer(max_features=20, stop_words='english')

API_KEY = os.getenv("TMDB_ID") 
BASE_URL = "https://api.themoviedb.org/3/movie/"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_movie_poster(movie_id):
    url = f"{BASE_URL}{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"{POSTER_BASE_URL}{poster_path}" 
        else:
            return None
    else:
        print(f"Error: Unable to fetch details for movie ID {movie_id}.")
        return None

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def recommend(genres):
    try:
        g_movie['genres'] = g_movie['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        g_movie.loc[len(g_movie)] = ["0", "movieU", " ".join(genres)]

        g_movie["genres"] = g_movie["genres"].apply(lambda x: x.lower() if isinstance(x, str) else x)
        g_movie["genres"] = g_movie["genres"].apply(stem)

        vector = cv.fit_transform(g_movie["genres"]).toarray()

        similarity = cosine_similarity(vector)

        movie_index = g_movie[g_movie['title'] == 'movieU'].index[0]
        distances = similarity[movie_index]

        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:36]

        recommendations = []
        for i in movie_list:
            if g_movie.iloc[i[0]].title != "movieU":  # Exclude "movieU" explicitly
                movie_id = g_movie.iloc[i[0]].movie_id
                recommendations.append({
                    "title": g_movie.iloc[i[0]].title,
                    "poster_url": fetch_movie_poster(movie_id)
                })

        g_movie.drop(index=movie_index, inplace=True)

        return recommendations

    except Exception as e:
        print(f"Error in recommendation: {e}")
        raise
        
class GenreRequest(BaseModel):
    genres: list[str]

@app.get("/")
def root():
    return {"message": "Welcome to the Movie Recommendation API. Use the /recommend endpoint to get movie recommendations based on genres."}

@app.post("/recommend")
def get_recommendations(request: GenreRequest):
    try:
        genres = request.genres
        if not genres:
            raise HTTPException(status_code=400, detail="Genres list cannot be empty.")
        recommendations = recommend(genres)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

  
