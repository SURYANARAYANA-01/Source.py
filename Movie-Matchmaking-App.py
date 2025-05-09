import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
df = pd.read_csv('imdb_top_1000.csv')

# Data preprocessing
df['Genre'] = df['Genre'].str.replace(', ', ' ')
df['Overview'] = df['Overview'].fillna('')
df['Gross'] = df['Gross'].str.replace(',', '').str.replace('"', '').replace('', np.nan).astype(float)

# Create a combined feature for content-based filtering
df['Combined_Features'] = df['Genre'] + ' ' + df['Overview'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Series_Title']).drop_duplicates()

# Normalize numerical features for hybrid recommendation
scaler = MinMaxScaler()
numerical_features = df[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].fillna(0)
df[['IMDB_Rating_norm', 'Meta_score_norm', 'No_of_Votes_norm']] = scaler.fit_transform(numerical_features)

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices, weight_sim=0.7, weight_rating=0.2, weight_meta=0.1):
    """
    Get movie recommendations based on content similarity with optional hybrid scoring
    
    Parameters:
    - title: Movie title to get recommendations for
    - cosine_sim: Cosine similarity matrix
    - df: Movies dataframe
    - indices: Series mapping titles to indices
    - weights: How much to weight similarity vs ratings vs metascore
    
    Returns: DataFrame of recommended movies
    """
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores and indices of the top 30 most similar movies
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    
    # Get the movies and their similarity scores
    movies = df.iloc[movie_indices][['Series_Title', 'Genre', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Released_Year', 'Director']]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Create a DataFrame with the results
    recommendations = pd.DataFrame({
        'Title': movies['Series_Title'],
        'Genre': movies['Genre'],
        'IMDB_Rating': movies['IMDB_Rating'],
        'Meta_score': movies['Meta_score'],
        'Votes': movies['No_of_Votes'],
        'Year': movies['Released_Year'],
        'Director': movies['Director'],
        'Similarity_Score': similarity_scores
    })
    
    # Calculate hybrid score
    recommendations['Hybrid_Score'] = (
        weight_sim * recommendations['Similarity_Score'] +
        weight_rating * movies['IMDB_Rating_norm'] +
        weight_meta * movies['Meta_score_norm']
    )
    
    # Sort by hybrid score
    recommendations = recommendations.sort_values('Hybrid_Score', ascending=False)
    
    return recommendations.head(10)

def get_user_preferences():
    """
    Get user preferences for personalized recommendations
    """
    print("\nLet's customize your movie preferences:")
    
    # Genre preferences
    all_genres = set()
    for genres in df['Genre'].str.split(' '):
        all_genres.update(genres)
    
    print("\nAvailable genres:", ', '.join(all_genres))
    liked_genres = input("Enter genres you like (comma separated): ").strip().split(',')
    liked_genres = [g.strip() for g in liked_genres if g.strip()]
    
    # Director preferences
    print("\nSome top directors:", ', '.join(df['Director'].value_counts().head(10).index))
    liked_directors = input("Enter directors you like (comma separated, leave blank if none): ").strip().split(',')
    liked_directors = [d.strip() for d in liked_directors if d.strip()]
    
    # Star preferences
    print("\nSome top stars:", ', '.join(pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']]).value_counts().head(10).index))
    liked_stars = input("Enter stars you like (comma separated, leave blank if none): ").strip().split(',')
    liked_stars = [s.strip() for s in liked_stars if s.strip()]
    
    # Year range
    min_year = input("\nEarliest year you're interested in (leave blank for any): ").strip()
    max_year = input("Latest year you're interested in (leave blank for any): ").strip()
    
    # Rating threshold
    min_rating = input("\nMinimum IMDB rating you'd accept (0-10, leave blank for any): ").strip()
    
    return {
        'genres': liked_genres,
        'directors': liked_directors,
        'stars': liked_stars,
        'min_year': int(min_year) if min_year else None,
        'max_year': int(max_year) if max_year else None,
        'min_rating': float(min_rating) if min_rating else None
    }

def filter_by_preferences(recommendations, preferences):
    """
    Filter recommendations based on user preferences
    """
    if preferences['genres']:
        genre_mask = recommendations['Genre'].apply(
            lambda x: any(genre in x for genre in preferences['genres']))
        recommendations = recommendations[genre_mask]
    
    if preferences['directors']:
        director_mask = recommendations['Director'].apply(
            lambda x: any(director.lower() in x.lower() for director in preferences['directors']))
        recommendations = recommendations[director_mask]
    
    if preferences['min_year']:
        recommendations = recommendations[recommendations['Year'] >= preferences['min_year']]
    
    if preferences['max_year']:
        recommendations = recommendations[recommendations['Year'] <= preferences['max_year']]
    
    if preferences['min_rating']:
        recommendations = recommendations[recommendations['IMDB_Rating'] >= preferences['min_rating']]
    
    return recommendations

def recommend_based_on_likes(liked_movies, preferences=None):
    """
    Get recommendations based on multiple liked movies
    """
    all_recommendations = pd.DataFrame()
    
    for movie in liked_movies:
        try:
            recs = get_recommendations(movie)
            all_recommendations = pd.concat([all_recommendations, recs])
        except KeyError:
            print(f"Movie '{movie}' not found in database. Skipping.")
            continue
    
    # Average scores for movies recommended multiple times
    all_recommendations = all_recommendations.groupby('Title').agg({
        'Genre': 'first',
        'IMDB_Rating': 'first',
        'Meta_score': 'first',
        'Votes': 'first',
        'Year': 'first',
        'Director': 'first',
        'Similarity_Score': 'mean',
        'Hybrid_Score': 'mean'
    }).reset_index()
    
    # Remove movies that were in the input list
    all_recommendations = all_recommendations[~all_recommendations['Title'].isin(liked_movies)]
    
    # Apply user preferences if provided
    if preferences:
        all_recommendations = filter_by_preferences(all_recommendations, preferences)
    
    return all_recommendations.sort_values('Hybrid_Score', ascending=False).head(10)

def main():
    print("Welcome to the AI Movie Matchmaker!")
    print("We'll recommend movies based on your tastes.")
    
    # Get liked movies from user
    liked_movies = input("\nEnter movies you like (comma separated): ").strip().split(',')
    liked_movies = [m.strip() for m in liked_movies if m.strip()]
    
    # Get user preferences
    get_prefs = input("\nWould you like to specify additional preferences? (yes/no): ").strip().lower()
    preferences = None
    if get_prefs == 'yes':
        preferences = get_user_preferences()
    
    # Get recommendations
    print("\nGenerating recommendations...")
    recommendations = recommend_based_on_likes(liked_movies, preferences)
    
    # Display results
    if recommendations.empty:
        print("\nNo recommendations found matching your criteria. Try broadening your preferences.")
    else:
        print("\nHere are your personalized recommendations:")
        print(recommendations[['Title', 'Genre', 'IMDB_Rating', 'Year', 'Director']].to_string(index=False))
    
    # Option to explore more
    while True:
        explore = input("\nWould you like to explore recommendations for another movie? (yes/no): ").strip().lower()
        if explore != 'yes':
            break
        
        movie = input("Enter a movie title: ").strip()
        try:
            recs = get_recommendations(movie)
            print(f"\nMovies similar to '{movie}':")
            print(recs[['Title', 'Genre', 'IMDB_Rating', 'Year', 'Director']].to_string(index=False))
        except KeyError:
            print("Movie not found in database.")

    print("\nThank you for using the AI Movie Matchmaker!")

if __name__ == "__main__":
    main()
