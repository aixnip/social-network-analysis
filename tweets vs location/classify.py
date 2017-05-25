"""
classify.py
"""
import pickle
from sklearn.model_selection import train_test_split
import sklearn.feature_extraction.text as fet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn.feature_selection as fs
from sklearn.utils import shuffle
from nltk.stem.snowball import EnglishStemmer
import numpy as np
import re
from collections import Counter

def read_files():
    """
    this method assumes the files locates in the current folder...
    it reads all data from 4 files - tweets.pkl and tweets_test.pkl
    and combine them into a giant list of text

    returns X, y
    """
    #training data
    data1 = pickle.load(open('tweets_1.pkl', 'rb'))
    data2 = pickle.load(open('tweets_2.pkl', 'rb'))
    
    tweets1 = [t['text'] for t in data1 if 'text' in t.keys()]
    tweets2 = [t['text'] for t in data2 if 'text' in t.keys()]
    X_train = np.append(tweets1, tweets2)
    y_train = np.append(np.ones(len(tweets1)), np.zeros(len(tweets2)))

    #testing data
    data3 = pickle.load(open('tweets_1_test.pkl', 'rb'))
    data4 = pickle.load(open('tweets_2_test.pkl', 'rb'))

    tweets3 = [t['text'] for t in data3 if 'text' in t.keys()]
    tweets4 = [t['text'] for t in data4 if 'text' in t.keys()]
    X_test = np.append(tweets3, tweets4)
    y_test = np.append(np.ones(len(tweets3)), np.zeros(len(tweets4)))

    #shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=15)
    X_test, y_test = shuffle(X_test, y_test, random_state=15)
    
    return X_train, X_test, y_train, y_test

def preprocess(X):
    """
    this method removes url and @ username from the tweet text
    return a np array X
    """
    new_X = []
    for t in X:
        processed = re.sub('http\S+', '', t)
        processed = re.sub('@\S+', '', processed)
        new_X.append(processed)
    return np.array(new_X)

def report_accuracy(X_train, y_train, X_test, y_test, clf):
    """
    Args:
     X - features
     y - targets
     clf - classifier object

    this method takes X, y, and a classifier,
    then do a train test split,
    and report the accuracy scores using given classifier
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_pred, y_test)
    print('document-term matrix accuracy %0.3f'%acc_score)

class Tokenizer(object):
    """
    this class is necessary to for the vectorizer to stem the words
    """
    def __init__(self):
        self.stemmer = EnglishStemmer()
    def __call__(self, doc):
        doc= doc.lower()
        doc= re.sub('http\S+', '', doc)
        doc= re.sub('@\S+', '', doc)
        doc= re.sub('rt', '', doc)
        tokens = re.sub('\W+', ' ', doc).split()
        return [self.stemmer.stem(t) for t in tokens]

def feature_analyze(vec, X, y):
    """
    Args:
     vec - sklearn Vectorizer
     X - features
     y - targets

    this method analyzes feature chi-square values
    """
    print("\nFeature selection")
    chi, pval = fs.chi2(X, y)
    feats = vec.get_feature_names()
    print("\nTop chi2 values")
    for i in np.argsort(chi)[::-1][:10]:
        print('index=%d chisq=%.2f %s' % (i, chi[i], feats[i]))
    print("\nBottom chi2 values")
    for i in np.argsort(chi)[:10]:
        print('index=%d chisq=%.2f %s' % (i, chi[i], feats[i]))

def fit_train_transform(model, X_train, X_test):
    model.fit(X_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    return X_train_new, X_test_new

def test_penelty(penalty, X_train, y_train, X_test, y_test):
    """
    this method test the penalty - 'l1' or 'l2' with different C values

    Args:
     penalty - 'l1' or 'l2'
     X_train, y_train, X_test, y_test - training and testing data
    """
    C = [0.1, 1, 4, 16, 64]
    clf = LogisticRegression(penalty=penalty)
    print("\naccuracy on original x - %s regularization"%penalty)
    report_accuracy(X_train, y_train, X_test, y_test,  clf)
    print('experiment classifier with different c values')
    for const in C:
        clf_const = LogisticRegression(penalty=penalty,C=const)
        print("accuracy for c=%s"%(str(const)))
        report_accuracy(X_train, y_train, X_test, y_test,  clf_const)

def main():
    X_train_raw, X_test_raw, y_train, y_test = read_files()
    vec = fet.CountVectorizer(tokenizer=Tokenizer(), stop_words='english', ngram_range=(1,2), min_df=2)
    X_train, X_test = fit_train_transform(vec, X_train_raw, X_test_raw)
    print('training data shape')
    print(X_train.shape)
    print('testing data shape')
    print(X_test.shape)

    test_penelty('l2', X_train, y_train, X_test, y_test)
    test_penelty('l1', X_train, y_train, X_test, y_test)
    
    print('')
    clf = LogisticRegression(penalty='l1')
    tfidf = fet.TfidfTransformer()
    X_train_tfidf, X_test_tfidf = fit_train_transform(tfidf, X_train, X_test)
    print("accuracy on tf-idf x")
    report_accuracy(X_train_tfidf, y_train, X_test_tfidf, y_test, clf)

    feature_analyze(vec, X_train, y_train)
    
    clf.fit(X_train, y_train)
    sfm = fs.SelectFromModel(clf,prefit=True)
    X_selected1 = sfm.transform(X_train)
    X_selected_test1 = sfm.transform(X_test)
    print("\naccuracy on feature selected based on l1 penalties")
    print('selected %d features'%X_selected1.shape[1])
    report_accuracy(X_selected1, y_train, X_selected_test1, y_test,  clf)
    
    sbs = fs.SelectKBest(fs.chi2, k=200)
    sbs.fit(X_train, y_train)
    X_selected2 = sbs.transform(X_train)
    X_selected_test2 = sbs.transform(X_test)
    print("accuracy on feature selected x top 200 tfidf")
    report_accuracy(X_selected2, y_train, X_selected_test2, y_test,  clf)

    clf.fit(X_selected2, y_train)
    y_pred = clf.predict(X_selected_test2)
    y_stats = dict(Counter(y_pred))
    print(y_stats)
    acc_score = accuracy_score(y_pred, y_test)
    y_proba = clf.predict_proba(X_selected_test2)

    error= [(i, abs(y_proba[i][int(y_pred[i])])) for i in range(len(y_proba)) if y_pred[i] != y_test[i]]
    correct= [(i, 1- abs(y_proba[i][int(y_pred[i])])) for i in range(len(y_proba)) if y_pred[i] == y_test[i]]
    sorted_error = sorted(error, key=lambda x: -x[1])
    sorted_correct = sorted(correct, key=lambda x: x[1])
    most_error={}
    most_correct={}
    
    for i in range(len(sorted_error)):
        item = sorted_error[i]
        if y_test[item[0]] == 0 and y_pred[item[0]] == 1:
            most_error[0] = {'text':X_test_raw[item[0]], 'predict':y_pred[item[0]], 'error':item[1], 'true': y_test[item[0]]}
            break
    for i in range(len(sorted_error)):
        item = sorted_error[i]
        if y_test[item[0]] == 1 and y_pred[item[0]] == 0:
            most_error[1] = {'text':X_test_raw[item[0]], 'predict':y_pred[item[0]],'error':item[1], 'true': y_test[item[0]]}
            break
    for i in range(len(sorted_correct)):
        item = sorted_correct[i]
        if y_test[item[0]] == 0 and y_pred[item[0]] == 0:
            most_correct[0] = {'text':X_test_raw[item[0]], 'predict':y_pred[item[0]], 'error':item[1], 'true': y_test[item[0]]}
            break
    for i in range(len(sorted_correct)):
        item = sorted_correct[i]
        if y_test[item[0]] == 1 and y_pred[item[0]] == 1:
            most_correct[1] = {'text':X_test_raw[item[0]], 'predict':y_pred[item[0]], 'error':item[1], 'true': y_test[item[0]]}
            break
    print('top misclassified for each class')
    print(most_error)
    print('top correctly classified for each class')
    print(most_correct)
    results = {'num_training': len(y_train), 'num_testing': len(y_test), 'stats':y_stats, 'accuracy':acc_score, 'misclassified': most_error, 'correct_classified': most_correct}
    pickle.dump(results, open('classify_result.pkl', 'wb'))

if __name__ == '__main__':
    main()
