import numpy as np
import psutil
from scipy.sparse import csr_matrix
import re

def remove_year_format(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memmory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
def cosine_similarity_helper(i, j):
    dot_product = i.dot(j.T).data
    if dot_product.size == 0:
        return 0
    dot_product = dot_product[0]
    magnitude_product = np.sqrt(i.dot(i.T).data[0]) * np.sqrt(j.dot(j.T).data[0])
    if magnitude_product != 0:
        return dot_product / magnitude_product
    else:
        return 0

def subtract_row_means(matrix, row_means):
    nonzero_rows, nonzero_cols = matrix.nonzero()
    adjusted_data = matrix.data - row_means[nonzero_rows]
    adjusted_matrix = csr_matrix((adjusted_data, (nonzero_rows, nonzero_cols)), shape=matrix.shape)
    return adjusted_matrix
    
def jaccard_similarity_sparse(matrix, input_id):
    input_user_similarity_score = []
    set_input = set(matrix.getrow(input_id).indices)
    
    for user_id in range(matrix.shape[0]):
        if user_id != input_id:
            set_user = set(matrix.getrow(user_id).indices)
            intersection = len(set_input.intersection(set_user))
            union = len(set_input.union(set_user))
            if union != 0:
                input_user_similarity_score.append((user_id, intersection / union))
    return input_user_similarity_score

def dice_similarity_sparse(matrix, input_id):
    input_user_similarity_score = []
    set_input = set(matrix.getrow(input_id).indices)
    
    for user_id in range(matrix.shape[0]):
        if user_id != input_id:
            set_user = set(matrix.getrow(user_id).indices)
            intersection = len(set_input.intersection(set_user))
            divisor = len(set_input) + len(set_user)
            if divisor != 0:
                input_user_similarity_score.append((user_id, (2 * intersection) / divisor))
    return input_user_similarity_score

def cosine_similarity_sparse(matrix, input_id):
    input_user_similarity_score = []
    input_user_row = matrix.getrow(input_id)
    
    for user_id in range(matrix.shape[0]):
        if user_id != input_id:
            current_user_row = matrix.getrow(user_id)
            similarity_score = cosine_similarity_helper(input_user_row, current_user_row)
            input_user_similarity_score.append((user_id, similarity_score))
    
    return input_user_similarity_score

def pearson_similarity_sparse(matrix, input_id):
    input_user_similarity_scores = []
    row_means = np.array(matrix.mean(axis=1)).flatten()
    mean_centered_matrix = subtract_row_means(matrix, row_means)
    input_user_row = mean_centered_matrix.getrow(input_id)

    for user_id in range(matrix.shape[0]):
        if user_id != input_id:
            current_user_row = mean_centered_matrix.getrow(user_id)
            similarity_score = cosine_similarity_helper(input_user_row, current_user_row)
            input_user_similarity_scores.append((user_id, similarity_score))
            
    return input_user_similarity_scores

def cosine_similarity_sparse_item(matrix, unrated_movie, rated_movies):
    input_user_similarity_score = []
    input_user_row = matrix.getrow(unrated_movie)
    
    for movie_id in rated_movies:
        current_movie_row = matrix.getrow(movie_id)
        similarity_score = cosine_similarity_helper(input_user_row, current_movie_row)
        input_user_similarity_score.append((movie_id, similarity_score))
    
    return input_user_similarity_score
    
def jaccard_similarity_sparse_item(matrix, unrated_movie, rated_movies):
    input_user_similarity_score = []
    set_input_movie = set(matrix.getrow(unrated_movie).indices)
    
    for movie_id in rated_movies:
        set_current_movie = set(matrix.getrow(movie_id).indices)
        intersection = len(set_input_movie.intersection(set_current_movie))
        union = len(set_input_movie.union(set_current_movie))
        if union != 0:
            input_user_similarity_score.append((movie_id, intersection / union))
    return input_user_similarity_score

def dice_similarity_sparse_item(matrix, unrated_movie, rated_movies):
    input_user_similarity_score = []
    set_input_movie = set(matrix.getrow(unrated_movie).indices)
    
    for movie_id in rated_movies:
        set_current_movie = set(matrix.getrow(movie_id).indices)
        intersection = len(set_input_movie.intersection(set_current_movie))
        divisor = len(set_input_movie) + len(set_current_movie)
        if divisor != 0:
            input_user_similarity_score.append((movie_id, (2 * intersection) / divisor))
    return input_user_similarity_score

def pearson_similarity_sparse_item(matrix, unrated_movie, rated_movies):
    input_user_similarity_scores = []
    row_means = np.array(matrix.mean(axis=1)).flatten()
    mean_centered_matrix = subtract_row_means(matrix, row_means)
    input_movie_row = mean_centered_matrix.getrow(unrated_movie)

    for movie_id in rated_movies:
        current_movie_row = mean_centered_matrix.getrow(movie_id)
        similarity_score = cosine_similarity_helper(input_movie_row, current_movie_row)
        input_user_similarity_scores.append((movie_id, similarity_score))
            
    return input_user_similarity_scores
