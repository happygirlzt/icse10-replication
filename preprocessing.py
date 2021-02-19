"""
Author: happygirlzt
Date: 16 Feb 2021
"""

# tokenizing, stemming, and stop words removal
import re
from typing import List

import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language = 'english')

class SpacyPreprocessor:
    def __init__(
        self,
        spacy_model = None
    ):

        if not spacy_model:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model

    @staticmethod
    def download_spacy_model(model = 'en_core_web_sm'):
        print('Downloading spaCy model {}'.format(model))
        spacy.cli.download(model)
        print('Finished downloading model')

    @staticmethod
    def load_model(model = 'en_core_web_sm'):

        return spacy.load(model, disable = ['ner', 'parser'])

    def tokenize(self, text) -> List[str]:

        doc = self.model(text)
        return [token.text for token in doc]

    def preprocess_text(self, text) -> str:

        # Remove non alphabetic characters
        text = re.sub(r'[^a-zA-Z\']', ' ', text)

        # remove non-Unicode characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # doc = self.tokenize(text)
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text_list(self, texts = List[str]) -> List[str]:

        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc: Doc) -> str:

        tokens = []

        for token in doc:
            if not token.is_stop and token.text.strip() != '':
                tokens.append(token)

        stems = []
        for token in tokens:
            stems.append(stemmer.stem(token.text))

        text = ' '.join(stems)
        text = text.lower()
        return text


if __name__ == "__main__":
    spacy_model = SpacyPreprocessor.load_model()
    preprocessor = SpacyPreprocessor(spacy_model = spacy_model)
    # cleaned_reports = preprocessor.preprocess_text_list(reports)
    cleaned_text = preprocessor.preprocess_text(
        'what are you talking...??? you are a silly girl'
    )

    print(cleaned_text)