"""
Author: happygirlzt
Date: 23 Feb 2021
"""

from pathlib import Path
import string
import ast
from config import *
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm

import pickle

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
stop_words = set(stopwords.words('english'))


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

    data_df = pd.read_csv(new_data)
    tqdm.pandas()

    # 1. tokenization -> generating processed data

    data_df['short_desc'] = data_df['short_desc'].astype('str')
    data_df['desc'] = data_df['desc'].astype('str')
    data_df['both'] = data_df['both'].astype('str')

    processed_df = pd.DataFrame({
        'bug_id': data_df['bug_id'],
        'short_desc_token': data_df['short_desc'].progress_apply(preprocess),
        'desc_token': data_df['desc'].progress_apply(preprocess),
        'both_token': data_df['both'].progress_apply(preprocess)
    })
    processed_df.to_csv(processed_file, index=False)

    # convert string to list
    data_df['desc_token'] = data_df['desc_token'].progress_apply(
        ast.literal_eval)
    data_df['short_desc_token'] = data_df['short_desc_token'].progress_apply(
        ast.literal_eval)
    data_df['both_token'] = data_df['both_token'].progress_apply(
        ast.literal_eval)

    ngram_df = pd.DataFrame({
        'bug_id': data_df['bug_id'],
        'one_short_desc': data_df['short_desc_token'].progress_apply(extract_1_gram),
        'one_desc': data_df['desc_token'].progress_apply(extract_1_gram),
        'one_both': data_df['both_token'].progress_apply(extract_1_gram),
        'bi_short_desc': data_df['short_desc_token'].progress_apply(extract_bigrams),
        'bi_desc': data_df['desc_token'].progress_apply(extract_bigrams),
        'bi_both': data_df['both_token'].progress_apply(extract_bigrams)
    })

    ngram_df.to_pickle(ngram_pickle)

    # ngram_df.to_csv(ngram_file, index = False)


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


def concatenate_summary_desc(data_file):
    """
    Generate a 'both' column by concatenating short_desc and desc
    """
    data_df = pd.read_csv(data_file)

    data_df = data_df.fillna('')
    # check NAN first
    # data_df.loc[data_df['desc'].isnull(), 'desc'] = ''
    # data_df.loc[data_df['short_desc'].isnull(), 'short_desc'] = ''
    data_df.to_csv(data_file, index=False, na_rep='')

    data_df['both'] = data_df['short_desc'] + ' ' + data_df['desc']
    # data_df.loc[data_df['both'].isnull(), 'both'] = ''

    # return data_df['short_desc'], data_df['desc'], data_df['both']
    data_df.to_csv(new_data_file, index=False, na_rep='')


def build_corpus(ngram_file):
    """
    Build and save three types of corpus, i.e., summary, description, and both
    """

    ngram_df = pd.read_pickle(ngram_pickle)
    tqdm.pandas()

    desc_df = pd.DataFrame({
        'bug_id': ngram_df['bug_id'],
        'one_desc': ngram_df['one_desc'],
        'bi_desc': ngram_df['bi_desc']
    })
    with open(desc_corpus, 'wb') as handler:
        pickle.dump(desc_df, handler, protocol=pickle.HIGHEST_PROTOCOL)

    short_desc_df = pd.DataFrame({
        'bug_id': ngram_df['bug_id'],
        'one_short_desc': ngram_df['one_short_desc'],
        'bi_short_desc': ngram_df['bi_short_desc']
    })
    with open(short_desc_corpus, 'wb') as handler:
        pickle.dump(short_desc_df, handler, protocol=pickle.HIGHEST_PROTOCOL)

    both_df = pd.DataFrame({
        'bug_id': ngram_df['bug_id'],
        'one_both': ngram_df['one_both'],
        'bi_both': ngram_df['bi_both']
    })
    with open(both_corpus, 'wb') as handler:
        pickle.dump(both_df, handler, protocol=pickle.HIGHEST_PROTOCOL)


def generate_train_pairs(train_pair_file):
    '''
    Save positive pair file and negative pair file
    '''
    with open(train_pair_file, 'r') as handler:
        lines = handler.readlines()

    positive_pairs = []
    negative_pairs = []

    for line in lines:
        splitted = line.split(',')

        first_id, second_id, label = int(splitted[0]), int(
            splitted[1]), int(splitted[2])
        if label == 1:
            positive_pairs.append([first_id, second_id])
        else:
            negative_pairs.append([first_id, second_id])

    with open(positive_samples_file, 'wb') as handler:
        pickle.dump(positive_pairs, handler)

    with open(negative_samples_file, 'wb') as handler:
        pickle.dump(negative_pairs, handler)


if __name__ == '__main__':
    if not Path(new_data_file).is_file():
        concatenate_summary_desc(data)

    if not Path(ngram_pickle).is_file():
        preprocess_all(new_data_file)

    if not Path(short_desc_corpus).is_file() 
    or not Path(desc_corpus).is_file()
    or not Path(both_corpus).is_file():
        build_corpus(ngram_file)

    if not Path(positive_samples_file).is_file() or not Path(negative_samples_file).is_file():
        generate_train_pairs(train_pair_file)
