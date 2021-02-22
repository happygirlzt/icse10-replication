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

from pathlib import Path
import string

import ast

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
num_features = 54


def preprocess(original_text):
    """
    Preprocessing on one single cell (report)
    """
    # lowercase
    text = original_text.lower()
    # 1. tokenize
    tokenized_words = word_tokenize(original_text)
    
    # 2. remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped_words = [w.translate(table) for w in tokenized_words]

    # 3. remove stop words
    filtered_report = []
    for word in stripped_words:
        if word not in stop_words:
            filtered_report.append(word)
    
    # 4. stemming
    stemmed_words = []
    for word in filtered_report:
        if len(word) > 0:
            stemmed_words.append(stemmer.stem(word))

    return stemmed_words


def preprocess_all(new_data):
    """
    Apply the preprocessing function on all reports
    """
    data_df = pd.read_csv(processed_file)
    tqdm.pandas()

    # data_df = pd.read_csv(new_data)
    # data_df['short_desc'] = data_df['short_desc'].astype('str')
    # data_df['desc'] = data_df['desc'].astype('str')
    # data_df['both'] = data_df['both'].astype('str')

    # processed_df = pd.DataFrame({
    #     'short_desc_token': data_df['short_desc'].progress_apply(preprocess),
    #     'desc_token': data_df['desc'].progress_apply(preprocess),
    #     'both_token': data_df['both'].progress_apply(preprocess)
    # })
    # processed_df.to_csv(processed_file, index = False)

    # convert string to list
    data_df['desc_token'] = data_df['desc_token'].progress_apply(ast.literal_eval)
    data_df['short_desc_token'] = data_df['short_desc_token'].progress_apply(ast.literal_eval)
    data_df['both_token'] = data_df['both_token'].progress_apply(ast.literal_eval)

    ngram_df = pd.DataFrame({
        'one_short_desc' : data_df['short_desc_token'].progress_apply(extract_1_gram),
        'one_desc' : data_df['desc_token'].progress_apply(extract_1_gram),
        'one_both' : data_df['both_token'].progress_apply(extract_1_gram),
        'bi_short_desc' : data_df['short_desc_token'].progress_apply(extract_bigrams),
        'bi_desc' : data_df['desc_token'].progress_apply(extract_bigrams),
        'bi_both' : data_df['both_token'].progress_apply(extract_bigrams)
    })
    ngram_df.to_csv(ngram_file, index = False)

    print('finished')


def extract_bigrams(token_list):
    # terms = token_list[1:-1].split(',')
    bigram_features = set()

    # 'hi', 'what', 'do' len = 3
    # 0 + 2, 1 + 2
    for i in range(len(token_list) - 1):
        # 'hi what'
        bigram_features.add(token_list[i] + ' ' + token_list[i + 1])

    return bigram_features

    
def extract_1_gram(token_list):
    """
    Extract 1-gram in a bug report text part
    """

    features = set()

    for term in token_list:
        if len(term) > 0:
            features.add(term)    

    return features


def calculate_similarity(corpus, term_corpus):
    """
    calcualte the similarity between bags
    corpus: [summary, description, both]
    term_corpus: one gram and bigrams in the two bug reports
    """

    N = len(corpus) # total number of reports
    term_counter = Counter()

    for report in corpus:
        one_gram_features = extract_1_gram(report)
        bigram_features = extract_bigrams(report)
        total_features = one_gram_features.union(bigram_features)

        for term in term_corpus:
            if term in total_features:
                term_counter[term] += 1

    similarity = 0
    # idf = log(num_documents / num_documents_contain_term + 1)
    for (term, term_frequency) in term_counter.items():
        term_IDF = math.log(float(N) / (term_frequency + 1))
        similarity += term_IDF
    return similarity

    
def generate_buckets():
    # TODO
    pass
    

def extract_text(bug_id, df, is_one):
    """
    Extract the three text parts given a bug id and the data dataframe
    """

    row = df.loc[df['bug_id'] == int(bug_id)].iloc[0]
    if is_one:
        short_desc =  ast.literal_eval(row.one_short_desc)
        description =  ast.literal_eval(row.one_desc)
        both = ast.literal_eval(row.one_both)
        return [short_desc, description, both]
    else:
        short_desc = ast.literal_eval(row.bi_short_desc)
        description = ast.literal_eval(row.bi_desc)
        both = ast.literal_eval(row.bi_both)
        return [short_desc, description, both]


def build_corpus(data_df):
    # df['short_desc'].astype(str) + ...
    # data_df = data_df.replace('"','', regex = True)

    data_df = data_df.fillna('')
    # check NAN first
    # data_df.loc[data_df['desc'].isnull(), 'desc'] = ''
    # data_df.loc[data_df['short_desc'].isnull(), 'short_desc'] = ''
    data_df.to_csv(data_file, index = False, na_rep = '')


    data_df['both'] = data_df['short_desc'] + ' ' + data_df['desc']
    # data_df.loc[data_df['both'].isnull(), 'both'] = ''

    # return data_df['short_desc'], data_df['desc'], data_df['both']
    data_df.to_csv(new_data_file, index = False, na_rep = '')


def generate_features(report_1_id, report_2_id, data_df):
    """
    For each pair of reports, we extract the features, tuple
    """

    feature_vectors = np.zeros(num_features)
    feature_vec_idx = 0

    # 1-gram
    for text_1 in extract_text(report_1_id, data_df, true):
        bag_1 = extract_1_gram(text_1)

        for text_2 in extract_text(report_2_id, data_df, false):

            bag_2 = extract_1_gram(text_2)
            term_corpus = bag_1.union(bag_2)

            for corpus in [data_df['short_desc'], data_df['description'], data_df['both']]:
                similarity = calculate_similarity(corpus, term_corpus)

                feature_vectors[feature_vec_idx] = similarity
                feature_vec_idx += 1

    # bigrams
    for text_1 in extract_text(report_1_id, data, true):
        bag_1 = extract_bigrams(text_1)

        for text_2 in extract_text(report_2_id, data_df, false):

            bag_2 = extract_bigrams(text_2)
            term_corpus = bag_1.union(bag_2)

            for corpus in [data_df['short_desc'], data_df['description'], data_df['both']]:
                similarity = calculate_similarity(corpus, term_corpus)

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


if __name__ == '__main__':
    data_file = '/media/zt/dbrd/dataset/open_office_2001-2008_2010/generated/data.csv'
    new_data_file = '/media/zt/dbrd/dataset/open_office_2001-2008_2010/generated/new_data.csv'
    processed_file = '/media/zt/dbrd/dataset/open_office_2001-2008_2010/generated/processed_data.csv'
    ngram_file = '/media/zt/dbrd/dataset/open_office_2001-2008_2010/generated/ngram_data.csv'

    data = pd.read_csv(data_file)
    # should I use the whole corpus or just training data?

    summary_corpus = data['short_desc']
    # short_desc,desc
    description_corpus = data['desc']

    if not Path(new_data_file).is_file():
        build_corpus(data)
    
    preprocess_all(new_data_file)

    # buckets = generate_buckets()

    # # 1/ preprocess
    # # 2/ 

    # generate_train_X_y()
    # clf.fit(X, y)

    # s = pickle.dumps(clf)