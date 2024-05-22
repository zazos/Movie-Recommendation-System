import numpy as np
import re
import math
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

def similar_users(similarity_matrix, k):
    top_k = sorted(similarity_matrix, key=lambda x: x[1], reverse=True)[:k]
    return [user_id for user_id, _ in top_k]

def similar_movies(similarity_matrix, k):
    top_k = sorted(similarity_matrix, key=lambda x: x[1], reverse=True)[:k]
    return [(movie_id, similarity) for movie_id, similarity in top_k]

def similar_movies_mapping(similarity_matrix, k, movies_mapping):
    top_k = sorted(similarity_matrix, key=lambda x: x[1], reverse=True)[:k]
    reverse_mapping = {index: movie_id for movie_id, index in movies_mapping.items()}
    return [reverse_mapping[index] for index, _ in top_k]

def predict_user_ratings(input_user, top_k, sparse_user_item_matrix, similarity_matrix, N, movies_mapping):
    user_ratings = sparse_user_item_matrix.getrow(input_user).toarray().flatten()
    predictions = {}
    
    for i, movie_id in enumerate(movies_mapping.keys()):
        # print(f"Processing movie {i + 1}/{len(movies_mapping)}")
        if user_ratings[i] == 0:
            numerator = 0
            divisor = 0
            for similar_user, similarity in similarity_matrix:
                if similar_user in top_k:
                    similar_user_rating = sparse_user_item_matrix[similar_user, i]
                    if similar_user_rating != 0:
                        numerator += similarity * similar_user_rating
                        divisor += abs(similarity)
        
            if divisor != 0:
                predictions[movie_id] = numerator / divisor
    
    top_N_recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:N]
    return [movie_id for movie_id, _ in top_N_recommendations]

def predict_movie_ratings(input_user, sparse_item_user_matrix, N, movies_mapping, similarity_function, k):
    movie_ratings = sparse_item_user_matrix.getcol(input_user).toarray().flatten()
    rated_movies = np.where(movie_ratings != 0)[0]
    unrated_movies = np.where(movie_ratings == 0)[0]
    predictions = {}
    
    for i, unrated_movie in enumerate(unrated_movies):
        #  print(f"Processing unrated movie {i + 1}/{len(unrated_movies)}")
        movie_id = list(movies_mapping.keys())[unrated_movie]
        numerator = 0
        divisor = 0
        similarity_movies_matrix = similarity_function(sparse_item_user_matrix, unrated_movie, rated_movies)
        top_k_similar = similar_movies(similarity_movies_matrix, k)
        for similar_movie, similarity in top_k_similar:
            similar_movie_rating = movie_ratings[similar_movie]
            if similar_movie_rating != 0:
                numerator += similarity * similar_movie_rating
                divisor += similarity
    
        if divisor != 0:
            predictions[movie_id] = numerator / divisor
            
    top_N_recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:N]
    return [movie_id for movie_id, _ in top_N_recommendations]

def tf_computation(document):
    tf = {}
    for word in document:
        tf[word] = tf.get(word, 0) + 1
    
    for word in tf:
        tf[word] = tf[word] / len(document)
        
    return tf

def idf_computation(documents):
    idf = {}
    for document in documents:
        for word in set(document): # interested if a word appears, not how many times
            idf[word] = idf.get(word, 0) + 1
            
    for word in idf:
        idf[word] = math.log(len(documents) / idf[word])
    
    return idf

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(document):
    return [word for word in document if word not in stop_words]

def tfidf_computation(titles):
    documents = [remove_stopwords(doc.split()) for doc in titles]
    idf = idf_computation(documents)
    unique_words = set(word for document in documents for word in document)
    words_mapping = {word: index for index, word in enumerate(unique_words)}
    
    rows, cols, values = [], [], []
    for index, document in enumerate(documents):
        tf = tf_computation(document)
        for word in document:
            if word in idf and word in words_mapping:
                word_index = words_mapping[word]
                rows.append(index)
                cols.append(word_index)
                values.append(tf[word] * idf[word])
    
    tfidf_sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(documents), len(words_mapping)))
    
    l2_norms = np.sqrt(tfidf_sparse_matrix.power(2).sum(axis=1))
    l2_norms[l2_norms == 0] = 1e-10
    tfidf_matrix = tfidf_sparse_matrix.multiply(1 / l2_norms)
    
    return tfidf_matrix.tocsr()