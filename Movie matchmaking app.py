#Step 1: Run the AI Movie Matchmaker App

import pandas as pd
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Load dataset (make sure to upload it to /content/)

df = pd.read_csv("/content/imdb_top_1000.csv")
df['combined'] = df['Genre'].fillna('') + " " + df['Director'].fillna('') + " " + df['Overview'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

Simulated in-memory data

users = {}
ratings = {}
chats = {}

Flask setup

app = Flask(name)
run_with_ngrok(app)

@app.route("/")
def home():
return "AI Movie Matchmaking System is Running!"

@app.route("/register", methods=["POST"])
def register():
data = request.json
username = data["username"]
if username in users:
return jsonify({"error": "User already exists"}), 400
users[username] = {"friends": [], "watched": [], "xp": 0, "level": 1}
return jsonify({"message": f"User '{username}' registered!"})

@app.route("/add_friend", methods=["POST"])
def add_friend():
data = request.json
user, friend = data["user"], data["friend"]
if friend not in users:
return jsonify({"error": "Friend not found"}), 404
users[user]["friends"].append(friend)
return jsonify({"message": f"{friend} added as friend!"})

@app.route("/rate", methods=["POST"])
def rate_movie():
data = request.json
user, title, score = data["user"], data["title"], data["score"]
ratings.setdefault(title, {})[user] = score
if title not in users[user]["watched"]:
users[user]["watched"].append(title)
users[user]["xp"] += 10
users[user]["level"] = 1 + users[user]["xp"] // 100
return jsonify({"message": f"Rated '{title}' with score {score}", "xp": users[user]["xp"], "level": users[user]["level"]})

@app.route("/recommend", methods=["POST"])
def recommend():
query = request.json.get("query", "")
query_vec = vectorizer.transform([query])
sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
top_indices = sims.argsort()[-10:][::-1]
results = df.iloc[top_indices][["Series_Title", "Genre", "IMDB_Rating", "Director", "Overview"]].to_dict(orient="records")
return jsonify(results)

@app.route("/friend_recommend", methods=["POST"])
def friend_recommend():
user = request.json.get("user")
friends = users[user]["friends"]
watched_by_friends = {}
for friend in friends:
for movie in users[friend]["watched"]:
watched_by_friends[movie] = watched_by_friends.get(movie, 0) + 1
top_movies = sorted(watched_by_friends.items(), key=lambda x: -x[1])[:5]
titles = [m[0] for m in top_movies]
results = df[df["Series_Title"].isin(titles)][["Series_Title", "Genre", "IMDB_Rating", "Director", "Overview"]].to_dict(orient="records")
return jsonify(results)

@app.route("/chat", methods=["POST"])
def chat():
data = request.json
movie, user, message = data["movie"], data["user"], data["message"]
chats.setdefault(movie, []).append(f"{user}: {message}")
return jsonify({"chat": chats[movie]})

@app.route("/profile/<username>")
def profile(username):
if username not in users:
return jsonify({"error": "User not found"}), 404
return jsonify(users[username])

app.run()
