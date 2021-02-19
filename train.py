"""
Author: happygirlzt
Date: 18th Feb 2021

Feature extractor and training a SVM classifier
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from heapq import heapify, heappush, heappop, nlargest
from nltk.tokenize import RegexpTokenizer

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language = 'english')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from collections import Counter
import math
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
num_features = 54

def preprocess(original_text):
    """
    Preprocessing on one single cell (report)
    """
    # 1. tokenize
    tokenized_words = word_tokenize(original_text)
    
    # 2. remove stop words
    filtered_report = []
    for word in tokenized_words:
        if word not in stop_words:
            filtered_report.append(word)
    
    # 3. stemming
    stemmed_words = []
    for word in filtered_report:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words


def preprocess_all(original_series):
    """
    Apply the preprocessing function on all reports
    """
    original_series = original_series.apply(preprocess)


def extract_2_grams(report):
    """
    Extract 2-gram in a bug report
    """

    terms = tuple(report.lower().split())
    features = set()

    # 'hi', 'what', 'do' len = 3
    # 0 + 2, 1 + 2
    for i in range(len(terms)):
        if i + 2 <= len(terms):
            features.add(terms[i : i + 2])
    return features


def extract_1_gram(report):
    """
    Extract 1-gram in a bug report
    """

    terms = tuple(report.lower().split())
    features = set()

    for term in terms:
        features.add(term)

    return features


def calculate_sum_of_idf(corpus):
    """
    calcualte the similarity between bags
    """
    N = len(corpus) # total number of reports
    term_counter = Counter()

    for report in corpus:
        features = extract_features(report)
        for f in features:
            term_counter[' '.join(f)] += 1

    IDF = 0
    for (term, term_frequency) in term_counter.items():
        term_IDF = math.log(float(N) / term_frequency)
        IDF += term_IDF
    return IDF


def calculate_similarity_betweens_bags(bag_1, bag_2):
    for report in reports:

    
def generate_buckets():
    # TODO

def compute_similarity(query, report, corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    
def generate_features(report_1, report_2):
    """
    For each pair of reports, we extract the features
    """

    feature_vectors = np.zeros(num_features)
    feature_vec_idx = 0

    # 1-gram
    for scope_1 in [summary, description, both]:
        bag_1 = extract_1_gram(scope_1)

        for scope_2 in [summary, description, both]:
            bag_2 = extract_1_gram(scope_2)

            for corpus in [summary, description, both]:
                similarity = calculate_sum_of_idf()
                feature_vectors[feature_vec_idx] = similarity
                feature_vec_idx += 1

    # 2-gram
    for scope_1 in [summary, description, both]:
        bag_1 = extract_2_grams(scope_1)

        for scope_2 in [summary, description, both]:
            bag_2 = extract_2_grams(scope_2)

            for corpus in [summary, description, both]:
                similarity = calculate_sum_of_idf()
                feature_vectors[feature_vec_idx] = similarity
                feature_vec_idx += 1

    return feature_vectors


def generate_train_X_y(positive_samples, negative_samples):
    """
    Generate all positive and negative samples features
    """
    num_samples = len(positive_samples) + len(negative_samples)
    X = np.zeros((num_samples, num_features))
    y = np.zeros((num_samples, 1), dtype = int)
    col_idx = 0

    for report_1, report_2 in positive_samples:
        X[:, col_idx] = generate_features(report_1, report_2)
        y[col_idx] = 1
        col_idx += 1

    for report_1, report_2 in negative_samples:
        X[:, col_idx] = generate_features(report_1, report_2)
        y[:, col_idx] = 0
        col_idx += 1

    return X, y


def compute_similarity_bucket():
    """
    compute the similarity between a bug report and a bucket
    """
    max_prob = 0
    tests = [get_features(q, report) for report in bucket]

    for test in tests:
        probility = svm_predict(test)
        max_prob = max(probility, max_prob)

    return max_prob


def generate_candidate_reports(n):
    candidates = []
    heapify(candidates)

    for bucket in self.buckets:
        similarity = self.compute_similarity_bucket()
        master_bucket.similarity = similarity
        heappush(candidates, master_bucket)

    return nlargest(n, candidates)


if __name__ == '__main__':
    data = pd.read_csv(data_file)
    summary_corpus = data['short_desc']
    description_corpus = data['description']
    both_corpus = data['text']

    buckets = generate_buckets()

    generate_document_term_matrix(summary_corpus)
    # 1/ preprocess
    # 2/ 

    clf.fit(X, y)

    s = pickle.dumps(clf)