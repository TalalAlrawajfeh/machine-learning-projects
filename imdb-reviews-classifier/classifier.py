#!/usr/bin/python3


import math
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


class ProgressCaclulator:
    def __init__(self, items_count, id):
        self.id = id
        self.items_count = items_count
        self.processed_items = 0
        self.old_progress = 0
        self.start_time = time.time()
        self.average_time = 0

    def progress(self):
        self.processed_items += 1
        current_progress = math.floor(
            self.processed_items * 1.0 / (self.items_count * 1.0) * 100.0)

        if current_progress > self.old_progress:
            time_interval_length = time.time() - self.start_time
            self.start_time = time.time()

            self.average_time = (self.average_time * self.old_progress +
                                 time_interval_length) / current_progress

            print('[worker-' + str(self.id) + '] normalizing reviews... ' + str(current_progress) +
                  '% [in ' + str(math.ceil(time_interval_length)) + ' seconds]')
            print('[worker-' + str(self.id) + '] estimated time left: ' + str(math.ceil(self.average_time *
                                                                                        (100 - current_progress))) + ' seconds')
            self.old_progress = current_progress


@lru_cache(maxsize=500000)
def fast_lemmatize(lemmatizer, word, tag):
    return lemmatizer.lemmatize(word, tag)


@lru_cache(maxsize=500000)
def is_excluded_word(word):
    return word in stopwords.words('english') or not word.isalpha()


def read_data(file):
    return pd.read_csv(file, encoding='ISO-8859-1')


def read_imdb_reviews_data():
    return read_data('imdb_master.csv').drop(columns='file')


def normalize_reviews(data, lemmatizer, id):
    tag_mapper = defaultdict(lambda: wn.NOUN)
    tag_mapper['J'] = wn.ADJ
    tag_mapper['V'] = wn.VERB
    tag_mapper['R'] = wn.ADV

    progress_calculator = ProgressCaclulator(len(data), id)

    for index, entry in enumerate(data['review']):
        lemmatized_words = []
        for word, tag in pos_tag(entry):
            if is_excluded_word(word):
                continue

            lemmatized_word = fast_lemmatize(
                lemmatizer, word, tag_mapper[tag[0]])
            lemmatized_words.append(lemmatized_word)
            data.loc[index, 'normalized_review'] = str(lemmatized_words)

        progress_calculator.progress()


def parallel_normalize_reviews(data):
    data['review'] = [word_tokenize(entry.lower())
                      for entry in data['review'].dropna()]

    lemmatizer = WordNetLemmatizer()

    chunk_size = 1000
    number_of_chunks = math.floor(len(data) / chunk_size)
    remainder = len(data) % chunk_size

    current_index = 0
    with ThreadPoolExecutor(max_workers=6) as e:
        for i in range(1, number_of_chunks + 1):
            e.submit(normalize_reviews,
                     data.iloc[current_index: current_index + chunk_size], lemmatizer, i)
            current_index += chunk_size

    if remainder != 0:
        normalize_reviews(
            data[current_index: current_index + remainder], lemmatizer, number_of_chunks + 1)

    return data


def load_data():
    if os.path.isfile('normalized_data.pkl'):
        data = pd.read_pickle('normalized_data.pkl')
    else:
        data = parallel_normalize_reviews(read_imdb_reviews_data())
        data.to_pickle('normalized_data.pkl')

    train = data.query(
        'type == "train" and (label == "pos" or label == "neg")')
    test = data.query('type == "test"')

    train_x = train['review'].tolist()
    test_x = test['review'].tolist()

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train['label'].tolist())
    test_y = label_encoder.fit_transform(test['label'].tolist())

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000, preprocessor=lambda x: x, tokenizer=lambda x: x)

    tfidf_vectorizer.fit(train_x + test_x)

    train_x_tfidf = tfidf_vectorizer.transform(train_x)
    test_x_tfidf = tfidf_vectorizer.transform(test_x)

    return train_x_tfidf, train_y, test_x_tfidf, test_y, label_encoder


def create_classifier():
    train_x_tfidf, train_y, test_x_tfidf, test_y, label_encoder = load_data()

    svm_model = svm.SVC(C=1.0, kernel='linear')
    svm_model.fit(train_x_tfidf, train_y)

    pickle.dump(svm_model, open('model.pkl', 'wb'))

    predictions = svm_model.predict(test_x_tfidf)

    print("SVM Accuracy Score: ",
          accuracy_score(predictions, test_y)*100)
    print(classification_report(test_y, predictions,
                                target_names=label_encoder.inverse_transform([0, 1])))


def load_classifier():
    train_x_tfidf, train_y, test_x_tfidf, test_y, label_encoder = load_data()

    svm_model = pickle.load(open('model.pkl', 'rb'))

    predictions = svm_model.predict(test_x_tfidf)

    print("SVM Accuracy Score: ",
          accuracy_score(predictions, test_y)*100)
    print(classification_report(test_y, predictions,
                                target_names=label_encoder.inverse_transform([0, 1])))


if __name__ == "__main__":
    if not os.path.isfile('model.pkl'):
        create_classifier()
    else:
        load_classifier()
