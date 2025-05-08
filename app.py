import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("imdb_top_1000.csv")
    df['combined'] = df['Genre'].fillna('') + " " + df['Director'].fillna('') + " " + df['Overview'].fillna('')
    return df

df = load_data()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# Simulated database
users = {}
ratings = {}
chats = {}

st.title("AI Movie Matchmaker")

# User registration
username = st.text_input("Enter username")
if st.button("Register"):
    if username in users:
        st.warning("User already exists.")
    else:
        users[username] = {"friends": [], "watched": [], "xp": 0, "level": 1}
        st.success(f"{username} registered.")

# Add a friend
friend = st.text_input("Add a friend")
if st.button("Add Friend"):
    if friend in users:
        users[username]["friends"].append(friend)
        st.success(f"{friend} added.")
    else:
        st.error("Friend not found.")

# Rate a movie
title = st.selectbox("Choose a movie", df["Series_Title"])
score = st.slider("Rate the movie", 1, 5)
if st.button("Rate"):
    ratings.setdefault(title, {})[username] = score
    if title not in users[username]["watched"]:
        users[username]["watched"].append(title)
        users[username]["xp"] += 10
        users[username]["level"] = 1 + users[username]["xp"] // 100
    st.success(f"Rated '{title}' {score}/5")

# AI Recommendation
query = st.text_input("Search for movie by keyword or director")
if st.button("Recommend"):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-5:][::-1]
    st.subheader("Top Matches:")
    for i in top_indices:
        row = df.iloc[i]
        st.markdown(f"**{row['Series_Title']}** ({row['Genre']}) - {row['IMDB_Rating']}")
        st.caption(row['Overview'])

# Friend Recommendations
if st.button("Get Friend-Based Suggestions"):
    watched_by_friends = {}
    for friend in users[username]["friends"]:
        for movie in users[friend]["watched"]:
            watched_by_friends[movie] = watched_by_friends.get(movie, 0) + 1
    top_movies = sorted(watched_by_friends.items(), key=lambda x: -x[1])[:5]
    st.subheader("Your Friends Recommend:")
    for m in top_movies:
        st.write(m[0])

# Profile
if st.button("View My Profile"):
    st.json(users[username])