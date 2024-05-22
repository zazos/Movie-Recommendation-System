import numpy as np
import psutil
from prediction_functions import *
from scipy.sparse import lil_matrix
import pandas as pd
from similarity_metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer


def similarity_functions_mapping(algorithm, similarity_metric):
    similarity_functions = {
        'cosine': cosine_similarity_sparse,
        'jaccard': jaccard_similarity_sparse,
        'pearson': pearson_similarity_sparse,
        'dice': dice_similarity_sparse
    }
    similarity_functions_movie = {
        'cosine': cosine_similarity_sparse_item,
        'jaccard': jaccard_similarity_sparse_item,
        'pearson': pearson_similarity_sparse_item,
        'dice': dice_similarity_sparse_item
    }
    if algorithm == 'item-item':
        return similarity_functions_movie[similarity_metric]
    else:
        return similarity_functions[similarity_metric]

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memmory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
def user_user_algorithm(input_id, directory, N, similarity_metric):
    print_memory_usage()
    if directory == 'ml-latest-small' or directory == "ml-latest":
        ratings = pd.read_csv(f'{directory}/ratings.csv', usecols=['userId', 'movieId', 'rating'])
        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        movies_mapping = {movie_id: index for index, movie_id in enumerate(movies['movieId'])}
        
        similarity_function = similarity_functions_mapping('user-user', similarity_metric)
        
        if input_id not in ratings['userId'].values:
            print(f"User with ID {input_id} cannot be found.")
            return
        
        user_item_matrix = lil_matrix((ratings['userId'].iloc[-1] + 1, len(movies_mapping)), dtype=np.float32)   
        for chunk in pd.read_csv(f'{directory}/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=10000):
            for row in chunk.itertuples():
                user_id, movie_id, rating = row.userId, row.movieId, row.rating
                user_item_matrix[user_id - 1, movies_mapping[movie_id]] = rating

        sparse_user_item_matrix = user_item_matrix.tocsr()
        
        similarity_matrix = similarity_function(sparse_user_item_matrix, input_id - 1)
        top_k_similar_users = similar_users(similarity_matrix, k=128)
        top_n_recommendations = predict_user_ratings(input_id - 1, top_k_similar_users, sparse_user_item_matrix, similarity_matrix, N, movies_mapping)
        
        return top_n_recommendations
    else:
        print(f"Directory with the name {directory} is not recognized")
        return []
    
def item_item_algorithm(input_id, directory, N, similarity_metric):
    if directory == 'ml-latest-small' or directory == "ml-latest":
        ratings = pd.read_csv(f'{directory}/ratings.csv', usecols=['userId', 'movieId', 'rating'])
        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        movies_mapping = {movie_id: index for index, movie_id in enumerate(movies['movieId'])}
        
        similarity_function = similarity_functions_mapping('item-item', similarity_metric)
        
        if input_id not in ratings['userId'].values:
            print(f"User with ID {input_id} cannot be found.")
            return        

        user_item_matrix = lil_matrix((ratings['userId'].iloc[-1] + 1, len(movies_mapping)), dtype=np.float32) 
        for chunk in pd.read_csv(f'{directory}/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=10000):
            for row in chunk.itertuples():
                user_id, movie_id, rating = row.userId, row.movieId, row.rating
                user_item_matrix[user_id - 1, movies_mapping[movie_id]] = rating
                
        sparse_item_user_matrix = user_item_matrix.T.tocsr()
        
        top_n_recommendations = predict_movie_ratings(input_id - 1, sparse_item_user_matrix, N, movies_mapping, similarity_function, k=128)
        
        return top_n_recommendations
    else:
        print(f"Directory with the name {directory} is not recognized")
        return []
    
def title_algorithm(input_id, directory, N, similarity_metric):
    if directory == 'ml-latest-small' or directory == "ml-latest":
        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        movies_mapping = {movie_id: index for index, movie_id in enumerate(movies['movieId'])}
        
        similarity_function = similarity_functions_mapping('title', similarity_metric)
        
        if input_id not in movies['movieId'].values:
            print(f"Movie with ID {input_id} cannot be found.")
            return

        movies['title'] = movies['title'].apply(remove_year_format)
        
        tfidf_titles = tfidf_computation(movies['title'])
        
        movie_index = movies_mapping[input_id]
        similarity_matrix = similarity_function(tfidf_titles, movie_index)
        
        similar_movies = similar_movies_mapping(similarity_matrix, N, movies_mapping)
        
        return similar_movies
    else:
        print(f"Directory with the name {directory} is not recognized")
        return []

def tag_algorithm(input_id, directory, N, similarity_metric):
    if directory == 'ml-latest-small' or directory == "ml-latest":
        tags = pd.read_csv(f'{directory}/tags.csv', usecols=['userId', 'movieId', 'tag'])
        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        
        unique_tags = tags['tag'].unique()
        
        similarity_function = similarity_functions_mapping('tag', similarity_metric)
        
        movies_mapping = {movie: index for index, movie in enumerate(movies['movieId'])} 
        tags_mapping = {tag: index for index, tag in enumerate(unique_tags)}
        
        if input_id not in tags['movieId'].values:
            print(f"Movie with ID {input_id} has to tags. Unable to make recommendations.")
            return []

        movies_tags_matrix = lil_matrix((len(movies), len(unique_tags)))
        for _, row in tags.iterrows():
            movie = movies_mapping[row['movieId']]
            tag = tags_mapping[row['tag']]
            movies_tags_matrix[movie, tag] += 1         
        
        if input_id not in movies_mapping:
            print(f"Movie with ID {input_id} cannot be found.")
            return []
        
        sparse_movies_tags_matrix = movies_tags_matrix.tocsr()

        movie_index = movies_mapping[input_id]
        similarity_matrix = similarity_function(sparse_movies_tags_matrix, movie_index)

        top_n_similar_movies = similar_movies_mapping(similarity_matrix, N, movies_mapping)
        
        return top_n_similar_movies
    else:
        print(f"Directory with the name {directory} is not recognized")
        return []
   
def user_preferred_tags(user_id, tags):
    user_tags = tags[tags['userId'] == user_id]['tag']
    if len(user_tags) == 0:
        print(f"User {user_id} hasn't tagged any movie")
        return []
    top_tags = user_tags.value_counts()
    return top_tags

def tag_relevance(combined_user_recommendations, user_top_tags, tags, N):
    relevance_scores = {}
    
    for movie_id in combined_user_recommendations:
        movie_tags = set(tags[tags['movieId'] == movie_id]['tag'])
        relevance_score = len(movie_tags.intersection(user_top_tags))
        relevance_scores[movie_id] = relevance_score
    
    return sorted(relevance_scores, key=relevance_scores.get, reverse=True)[:N]
    
     
def hybrid_algorithm(input_id, directory, N, similarity_metric):
    if directory == 'ml-latest-small' or directory == "ml-latest":
        tags = pd.read_csv(f'{directory}/tags.csv', usecols=['userId', 'movieId', 'tag'])

        print("Now running user-user algorithm ...")
        user_user_recommendations = user_user_algorithm(input_id, directory, N, similarity_metric)
        print("user-user algorithm successfully executed!")
        
        print("Now running item-item algorithm ...")
        item_item_recommendations = item_item_algorithm(input_id, directory, N, similarity_metric)
        print("item-item algorithm successfully executed!")
        
        combined_scores = {}
        weights = {'item-item': 0.7, 'user-user': 0.3}
        algorithms_weight_dict = zip([item_item_recommendations, user_user_recommendations], weights.values())
        for recommendation_list, weight in algorithms_weight_dict:
            for movie_id in recommendation_list:
                combined_scores[movie_id] = combined_scores.get(movie_id, 0) + weight
        top_n_combined_recommendations = sorted(combined_scores, key=combined_scores.get, reverse=True)[:(2*N)]
        
        if input_id not in tags['userId'].values:
            print(f"User {input_id} hasn't tagged any movie. Unable to make recommendations using tag algorithm.")
            print("Prining non tag-based movies:")
            
            return top_n_combined_recommendations[:N]
        else:
            user_top_tags = user_preferred_tags(input_id, tags)
            top_n_recommendations = tag_relevance(top_n_combined_recommendations, user_top_tags, tags, N)
        
            return top_n_recommendations
    else:
        print(f"Directory with the name {directory} is not recognized")
        return []