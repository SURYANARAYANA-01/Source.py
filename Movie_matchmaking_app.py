# AI Movie Matchmaking Streamlit App

import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_movies():
    df = pd.read_csv("imdb_top_1000.csv")
    df['combined'] = df['Genre'].fillna('') + " " + df['Director'].fillna('') + " " + df['Overview'].fillna('')
    return df

@st.cache_resource
def vectorize(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    return tfidf, matrix

def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(data):
    with open("users.json", "w") as f:
        json.dump(data, f, indent=4)

df = load_movies()
tfidf, matrix = vectorize(df)
users = load_users()

if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Login"

def login_page():
    st.title("AI Movie Matchmaker")
    choice = st.radio("Choose Action", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button(choice):
        if choice == "Login":
            if username in users and users[username]["password"] == password:
                st.session_state.user = username
                st.session_state.page = "Home"
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")
        elif choice == "Register":
            if username in users:
                st.warning("Username exists.")
            else:
                users[username] = {
                    "password": password,
                    "friends": [],
                    "watched": [],
                    "continue": [],
                    "favorites": [],
                    "xp": 0,
                    "level": 1,
                    "ratings": {},
                    "chats": {}
                }
                save_users(users)
                st.success("Registered! Please login.")

def home_page():
    st.sidebar.title(f"Welcome, {st.session_state.user}")
    st.title("Home - Featured Movies")
    
    featured = df.sample(5)
    for _, row in featured.iterrows():
        st.subheader(row["Series_Title"])
        st.text(row["Genre"])
        st.caption(row["Overview"])

    st.subheader("Search Movies")
    query = st.text_input("Search by keyword or director")
    if query:
        query_vec = tfidf.transform([query])
        sims = cosine_similarity(query_vec, matrix).flatten()
        top = sims.argsort()[-5:][::-1]
        for i in top:
            movie = df.iloc[i]
            st.write(f"**{movie['Series_Title']}** - {movie['Genre']} - {movie['IMDB_Rating']}")
            if st.button(f"Watch {movie['Series_Title']}", key=movie['Series_Title']):
                users[st.session_state.user]["continue"].append(movie["Series_Title"])
                save_users(users)

    st.subheader("Continue Watching")
    continue_list = users[st.session_state.user].get("continue", [])
    for title in continue_list:
        st.write(f"- {title}")

    st.subheader("AI Movie Matchmaking Recommendations")
    if users[st.session_state.user]["watched"]:
        watched_titles = users[st.session_state.user]["watched"]
        indices = df[df["Series_Title"].isin(watched_titles)].index.tolist()
        if indices:
            watched_vecs = matrix[indices]
            sims = cosine_similarity(watched_vecs.mean(axis=0), matrix).flatten()
            top_indices = sims.argsort()[-6:][::-1]
            st.write("Because you watched:")
            for i in top_indices:
                if df.iloc[i]["Series_Title"] not in watched_titles:
                    m = df.iloc[i]
                    st.markdown(f"**{m['Series_Title']}** ({m['Genre']}) - {m['IMDB_Rating']}")
                    st.caption(m["Overview"])
    else:
        st.info("Watch some movies to get AI recommendations!")

def friends_page():
    st.title("Friends & Chat")
    username = st.session_state.user
    my_data = users[username]
    
    st.subheader("Your Friends")
    for friend in my_data["friends"]:
        st.markdown(f"**{friend}**")
        if st.button(f"Chat with {friend}"):
            st.session_state.chat_friend = friend
    
    st.subheader("Add a New Friend")
    new_friend = st.text_input("Enter friend's username")
    if st.button("Add Friend"):
        if new_friend in users and new_friend != username:
            if new_friend not in my_data["friends"]:
                my_data["friends"].append(new_friend)
                users[new_friend]["friends"].append(username)
                save_users(users)
                st.success(f"Added {new_friend} as a friend!")
            else:
                st.info("Already friends.")
        else:
            st.error("User not found.")
    
    if "chat_friend" in st.session_state:
        friend = st.session_state.chat_friend
        st.subheader(f"Chat with {friend}")
        chat_key = "-".join(sorted([username, friend]))
        messages = users[username]["chats"].get(chat_key, [])
        for msg in messages:
            st.write(f"{msg['sender']}: {msg['text']}")
        new_msg = st.text_input("Type your message", key="msg")
        if st.button("Send"):
            new_entry = {"sender": username, "text": new_msg}
            messages.append(new_entry)
            users[username]["chats"][chat_key] = messages
            users[friend]["chats"][chat_key] = messages
            save_users(users)
            st.experimental_rerun()

def profile_page():
    user = st.session_state.user
    data = users[user]

    st.title(f"{user}'s Profile")
    st.write(f"**XP:** {data['xp']} | **Level:** {data['level']}")

    st.subheader("Favorite Movies")
    for title in data["favorites"]:
        st.write(f"- {title}")

    st.subheader("Rate a Movie")
    movie = st.selectbox("Select a movie to rate", df["Series_Title"])
    rating = st.slider("Your Rating", 1, 10)
    if st.button("Submit Rating"):
        users[user]["ratings"][movie] = rating
        if movie not in users[user]["watched"]:
            users[user]["watched"].append(movie)
            users[user]["xp"] += 10
            if users[user]["xp"] >= 100:
                users[user]["xp"] = 0
                users[user]["level"] += 1
        save_users(users)
        st.success("Rating submitted!")

    st.subheader("Mark Favorite")
    fav_movie = st.selectbox("Choose movie to favorite", df["Series_Title"])
    if st.button("Add to Favorites"):
        if fav_movie not in users[user]["favorites"]:
            users[user]["favorites"].append(fav_movie)
            save_users(users)
            st.success("Added to favorites.")

def main():
    if st.session_state.user:
        pages = ["Home", "Friends", "Profile"]
        choice = st.sidebar.selectbox("Menu", pages)
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "Login"
            st.experimental_rerun()

        if choice == "Home":
            home_page()
        elif choice == "Friends":
            friends_page()
        elif choice == "Profile":
            profile_page()
    else:
        login_page()

main()