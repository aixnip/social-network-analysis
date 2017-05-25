# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    token = []
    for index, row in movies.iterrows():
        token.append(tokenize_string(row.genres))
    movies['tokens'] = token
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})

    >>> movies = pd.DataFrame([[123, ['horror', 'romance']], [456, ['sci-fi']]], columns=['movieId', 'tokens'])
    >>> movies, vocab = featurize(movies)
    >>> sorted(vocab.items())
    [('horror', 0), ('romance', 1), ('sci-fi', 2)]
    >>> movies['features'][0].shape
    (1, 3)
    >>> movies['features'][0].toarray()
    array([[ 0.30103,  0.30103,  0.     ]])
    >>> movies['features'][1].toarray()
    array([[ 0.     ,  0.     ,  0.30103]])
    """
    tokens = movies['tokens'].tolist()
    total_counter = Counter()
    for t in tokens:
        uniq = set(t)
        total_counter.update(uniq)
    total = dict(total_counter)
    vocab = {}
    i = 0
    for g in sorted(total.keys()):
        vocab[g] = i
        i += 1
    features_array = []
    nfeatures = len(vocab)
    nmovies = len(movies)
    for t in tokens:
        this_count = dict(Counter(t))
        data = []
        col_index = []
        max_k = max(this_count.values())
        for k, v in this_count.items():
            data.append(v / max_k * np.log10(nmovies/total[k]))
            col_index.append(vocab[k])
        matrix = csr_matrix((np.array(data), (np.zeros(len(data)), np.array(col_index))), shape=(1, nfeatures))
        features_array.append(matrix)
    movies['features'] = features_array
    return movies, vocab

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.

        
    >>> a = csr_matrix((np.array([1]), (np.zeros(1), np.array([0]))), shape=(1, 5))
    >>> b = csr_matrix((np.array([1]), (np.zeros(1), np.array([0]))), shape=(1, 5))
    >>> c = csr_matrix((np.array([1]), (np.zeros(1), np.array([1]))), shape=(1, 5))
    >>> cosine_sim(a, b)
    1.0
    >>> cosine_sim(a, c)
    0.0
    """
    a_arr = a.toarray()
    b_arr = b.toarray()
    return np.dot(a_arr, b_arr.transpose())[0][0] / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    predict = []
    for index_test, row_test in ratings_test.iterrows():
        user_id = row_test['userId']
        movie_id = row_test['movieId']
        movie_feature = movies[movies.movieId==movie_id]['features'].iloc[0]
        num = 0
        denum = 0
        for index_train, row_train in ratings_train[ratings_train.userId==user_id].iterrows():
            train_movie_id = row_train['movieId']
            train_movie_feature = movies[movies.movieId==train_movie_id]['features'].iloc[0]
            similarity = cosine_sim(movie_feature, train_movie_feature)
            if similarity > 0:
                denum += similarity
                num += similarity * row_train.rating
        if denum == 0:
            predict.append(np.mean(ratings_train[ratings_train.userId==user_id]['rating']))
        else:
            predict.append(num / denum)
    return np.array(predict)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])



if __name__ == '__main__':
    main()
