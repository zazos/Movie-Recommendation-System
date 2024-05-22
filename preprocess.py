from prediction_algorithms import *
from prediction_functions import *
from similarity_metrics import *
import pandas as pd
import argparse
import shutil
import time
import os

def preprocess_by_user(ratings, directory, subset, N=250):
    unique_users = ratings['userId'].unique()
    similarity_metrics = ['jaccard', 'dice', 'cosine', 'pearson']
    user_algorithms = ['user-user', 'item-item']
    directories = {'user-user': os.path.join(os.getcwd(), 'user_user'),
           'item-item': os.path.join(os.getcwd(), 'item_item')}
    
    for algorithm in user_algorithms:
        if os.path.exists(directories[algorithm]):
            shutil.rmtree(directories[algorithm])
        os.makedirs(directories[algorithm], exist_ok=True)
        for metric in similarity_metrics:
            recommendations = []
            for user_id in unique_users[:subset]:
                print(f"Processing user {user_id}/{len(unique_users[:subset])} with {metric} similarity")
                if algorithm == 'user-user':
                    user_recommendations = (user_user_algorithm(user_id, directory, N, metric))
                elif algorithm == 'item-item':
                    user_recommendations = (item_item_algorithm(user_id, directory, N, metric))
                recommendations.append({'userId': user_id, 'recommendations': user_recommendations})
            
            recommendations_df = pd.DataFrame(recommendations)      
            filename = f'{algorithm}_{metric}_{N}_recommendations.csv'        
            recommendations_df.to_csv(os.path.join(directories[algorithm], filename), index=False)

def preprocess_by_movies(movies, directory, subset, N=250):
    similarity_metrics = ['jaccard', 'dice', 'cosine', 'pearson']
    movies_algorithms = ['tag', 'title']
    directories = {'tag': os.path.join(os.getcwd(), 'tag'),
           'title': os.path.join(os.getcwd(), 'title')}    
    
    for algorithm in movies_algorithms:
        if os.path.exists(directories[algorithm]):
            shutil.rmtree(directories[algorithm])
        os.makedirs(directories[algorithm], exist_ok=True)
        for metric in similarity_metrics:
            recommendations = []
            for index, movie_id in enumerate(movies['movieId'][:subset]):
                movie_title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
                print(f"Processing movie {index + 1}/{len(movies[:subset])} with {metric} similarity")
                if algorithm == 'tag':
                    movies_recommendations = (tag_algorithm(movie_id, directory, N, metric))
                elif algorithm == 'title':
                    movies_recommendations = (title_algorithm(movie_id, directory, N, metric))
                recommendations.append({'movieId': movie_id, 'title': movie_title, 'recommendations': movies_recommendations})
                
            recommendations_df = pd.DataFrame(recommendations)      
            filename = f'{algorithm}_{metric}_{N}_recommendations.csv'        
            recommendations_df.to_csv(os.path.join(directories[algorithm], filename), index=False)

def main(directory):
    if directory == 'ml-latest' or directory == 'ml-latest-small':
        start_time = time.time()
        ratings = pd.read_csv(f'{directory}/ratings.csv', usecols=['userId', 'movieId', 'rating'])
        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        
        preprocess_by_user(ratings, directory, subset=25)
        
        preprocess_by_movies(movies, directory, subset=25)
        
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.2f} seconds")
    else:
        print(f"Directory named {directory} is not recognized")
pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    
    args = parser.parse_args()
    
    main(args.directory)