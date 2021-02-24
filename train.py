"""
Author: happygirlzt
Date: 18th Feb 2021

Feature extractor and training a SVM classifier
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from heapq import heapify, heappush, heappop, nlargest

from collections import Counter
import math
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pathlib import Path
import string

import logging
logging.basicConfig(
    handlers = [logging.FileHandler(filename = './train.log', 
                                    encoding='utf-8',
                                    mode='a+')],
                    format = '%(asctime)s %(name)s:%(levelname)s:%(message)s', 
                    datefmt = '%F %A %T', 
                    level = logging.INFO)

from config import *

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
num_features = 54


def calculate_similarity(corpus, term_corpus):
    """
    calcualte the similarity between bags
    corpus: [summary, description, both]
    term_corpus: one gram and bigrams in the two bug reports
    """

    N = corpus.shape[0] # total number of reports
    term_counter = Counter()

    for index, value in corpus.items():
        total_features = value

        for term in term_corpus:
            if term in total_features:
                term_counter[term] += 1

    similarity = 0
    # idf = log(num_documents / num_documents_contain_term + 1)
    for term, term_frequency in term_counter.items():
        term_IDF = math.log(float(N) / (term_frequency + 1))
        similarity += term_IDF
    return similarity



def extract_text(bug_id, df, is_one):
    """
    Extract the three text parts given a bug id and the data dataframe
    """

    row = df.loc[df['bug_id'] == int(bug_id)].iloc[0]

    if is_one:
        short_desc = row.one_short_desc
        description = row.one_desc
        both = row.one_both
        return [short_desc, description, both]
    else:
        short_desc = row.bi_short_desc
        description = row.bi_desc
        both = row.bi_both
        return [short_desc, description, both]


def generate_features(report_1_id, report_2_id, data_df):
    """
    For each pair of reports, we extract the features, tuple
    """

    feature_vectors = np.zeros(num_features)
    feature_vec_idx = 0

    # 1-gram
    for bag_1 in extract_text(report_1_id, data_df, True):
        for bag_2 in extract_text(report_2_id, data_df, True):

            term_corpus = bag_1.union(bag_2)

            print('     iterate on 3 typs of corpus ' + '=' * 10)
            for corpus in [short_desc_df['one_short_desc'], desc_df['one_desc'], both_df['one_both']]:
                # print('         starting calculate similarity ' + '=' * 10)
                similarity = calculate_similarity(corpus, term_corpus)
                print('         {} features'.format(feature_vec_idx) + '=' * 10)
                feature_vectors[feature_vec_idx] = similarity
                feature_vec_idx += 1

    # bigrams
    for bag_1 in extract_text(report_1_id, data_df, False):
        for bag_2 in extract_text(report_2_id, data_df, False):

            term_corpus = bag_1.union(bag_2)

            for corpus in [short_desc_df['bi_short_desc'], desc_df['bi_desc'], both_df['bi_both']]:
                similarity = calculate_similarity(corpus, term_corpus)
                print('         {} features'.format(feature_vec_idx) + '=' * 10)
                feature_vectors[feature_vec_idx] = similarity
                feature_vec_idx += 1

    print(feature_vectors)
    return feature_vectors


def generate_train_X_y(positive_samples, negative_samples):
    """
    Generate all positive and negative samples features
    """

    num_samples = len(positive_samples) + len(negative_samples)
    # ngram_df = pd.read_csv(ngram_file)
    ngram_df = pd.read_pickle(ngram_pickle)

    X = np.zeros((num_samples, num_features))
    y = np.zeros((num_samples, 1), dtype = int)

    col_idx = 0
    
    logging.basicConfig(format = '%(asctime)s %(message)s')
    logging.info('Generating positive samples ' + '=' * 20)

    for sample in tqdm(positive_samples):
        print('calculating a postive sample ' + '=' * 10)
        report_1, report_2 = int(sample[0]), int(sample[1])
        X[col_idx, ] = generate_features(report_1, report_2, ngram_df)
        y[col_idx] = 1
        col_idx += 1

    logging.basicConfig(format = '%(asctime)s %(message)s')
    logging.info('Generating negative samples ' + '=' * 20)
    for sample in tqdm(negative_samples):
        report_1, report_2 = int(sample[0]), int(sample[1])
        X[col_idx, ] = generate_features(report_1, report_2, ngram_df)
        y[col_idx] = 0
        col_idx += 1

    with open(X_file, 'wb') as handler:
        pickle.dump(X, handler)

    with open(y_file, 'wb') as handler:
        pickle.dump(y, handler)

    return X, y


if __name__ == '__main__':

    data = pd.read_csv(ngram_file)

    # should I use the whole corpus or just training data?

    with open(short_desc_corpus, 'rb') as handler:
        short_desc_df = pickle.load(handler)

    with open(desc_corpus, 'rb') as handler:
        desc_df = pickle.load(handler)

    with open(both_corpus, 'rb') as handler:
        both_df = pickle.load(handler)

    with open(positive_samples_file, 'rb') as handler:
        positive_pairs = pickle.load(handler)

    with open(negative_samples_file, 'rb') as handler:
        negative_pairs = pickle.load(handler)

    if not Path(X_file).is_file() or not Path(y_file).is_file():
        generate_train_X_y(positive_pairs, negative_pairs)