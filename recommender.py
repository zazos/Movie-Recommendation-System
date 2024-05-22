import argparse
from similarity_metrics import *
from prediction_algorithms import *
import time
import os
    
def main(directory, number_of_recommendations, similarity_metric, algorithm, input_id):
    if directory == 'ml-latest' or directory == 'ml-latest-small':
        start_time = time.time()
        
        algorithm_functions = {
            'user-user': user_user_algorithm,
            'item-item': item_item_algorithm,
            'title': title_algorithm,
            'tag': tag_algorithm,
            'hybrid': hybrid_algorithm
        }
        
        if algorithm in algorithm_functions:
            algorithm_function = algorithm_functions[algorithm]
            top_n_recommendations = algorithm_function(input_id, directory, number_of_recommendations, similarity_metric)
        else:
            print(f"Algorithm with name {algorithm} is not recognized")
            return
        
        if len(top_n_recommendations) == 0:
            return

        movies = pd.read_csv(f'{directory}/movies.csv', usecols=['movieId', 'title'])
        if algorithm == 'title' or algorithm == 'tag':
            print(f"Top {number_of_recommendations} similar movies for {movies.loc[movies['movieId'] == input_id, 'title'].values[0]}")
        else:
            print(f"Top {number_of_recommendations} recommendations for user {input_id}")
            
        for movie_id in top_n_recommendations:
            movie_title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
            print(f"(id_{movie_id}) Movie: {movie_title}")

        end_time = time.time()
        print(f"Elapsed time using {algorithm} algorithm and {similarity_metric} metric: {(end_time - start_time):.2f} seconds")
    elif directory == 'real_system':
        if number_of_recommendations > 250:
            print("Number of recommendation exceed the pre processed maximum number of 250")
            return

        algorithm_functions = ['user-user', 'item-item', 'tag', 'title']
        similarity_metrics = ['jaccard', 'dice', 'cosine', 'pearson']
        
        if algorithm in algorithm_functions:
            if similarity_metric in similarity_metrics:
                directories = {'user-user': os.path.join(os.getcwd(), 'user_user'),
                            'item-item': os.path.join(os.getcwd(), 'item_item'), 
                            'tag': os.path.join(os.getcwd(), 'tag'),
                            'title': os.path.join(os.getcwd(), 'title')}
                
                filename = f'{algorithm}_{similarity_metric}_250_recommendations.csv'
                recommendations_df = pd.read_csv(f'{directories[algorithm]}/{filename}')               
                movies = pd.read_csv(f'ml-latest-small/movies.csv', usecols=['movieId', 'title'])
                if algorithm == 'title' or algorithm == 'tag':
                    if input_id > recommendations_df['movieId'].iloc[-1]:
                        print(f"{input_id} exceeds the avaiable users/movies in the datasets")
                        return
                    
                    print(f"Top {number_of_recommendations} similar movies for {recommendations_df.loc[recommendations_df['movieId'] == input_id, 'title'].values[0]}")
                    reccomendations_str = recommendations_df.loc[recommendations_df['movieId'] == input_id, 'recommendations'].tolist()[0]
                    if reccomendations_str == '[]':
                        print(f"{recommendations_df.loc[recommendations_df['movieId'] == input_id, 'title'].values[0]} has to tags. Unable to make recommendations.")
                        return
                    recommendations_str = reccomendations_str.strip('[]')
                    recommendations_elements = recommendations_str.split(', ')
                    
                    recommendations_list = [int(element) for element in recommendations_elements]
                else:
                    if input_id > recommendations_df['userId'].iloc[-1]:
                        print(f"{input_id} exceeds the avaiable users/movies in the datasets")
                        return
                    print(f"Top {number_of_recommendations} recommendations for user {input_id}")
                    reccomendations_str = recommendations_df.loc[recommendations_df['userId'] == input_id, 'recommendations'].tolist()[0]
                    recommendations_str = reccomendations_str.strip('[]')
                    recommendations_elements = recommendations_str.split(', ')
                    
                    recommendations_list = [int(element) for element in recommendations_elements]
            
                for movie_id in recommendations_list[:number_of_recommendations]:
                    movie_title = movies.loc[movies['movieId'] == (movie_id), 'title'].values[0]
                    print(f"(id_{movie_id}) Movie: {movie_title}")
            else:
                print(f"Similarity metric with name {similarity_metric} is not recognized")
                return
        else:
            print(f"Algorithm with name {algorithm} is not recognized")
            return
    else:
        print(f"Directory named {directory} is not recognized")     
pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-n', '--number', type=int, required=True)
    parser.add_argument('-s', '--similarity', required=True, choices=['cosine', 'jaccard', 'pearson', 'dice'])
    parser.add_argument('-a', '--algorithm', required=True, choices=['user-user', 'item-item', 'title', 'tag', 'hybrid'])
    parser.add_argument('-i', '--input', type=int, required=True)
    
    args = parser.parse_args()
    
    main(args.directory, args.number, args.similarity, args.algorithm, args.input)