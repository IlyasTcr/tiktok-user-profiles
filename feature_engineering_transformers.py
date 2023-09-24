import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from category_encoders import MEstimateEncoder
from functools import reduce

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def is_user_creator(biography, keywords):
    # Tokenize the biography and convert to lower case
    tokens = nltk.word_tokenize(biography.lower())

    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Check if any of the tokens are in your list of keywords
    return any(token in keywords for token in tokens)


# Define a string match transformer class
class StringMatchTransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with a list of keywords
    def __init__(self, keywords):
        self.keywords = keywords

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to check if any of the provided keywords are contained in the biography
    def transform(self, X):
        X = pd.DataFrame(X)
        results = X[0].apply(lambda biography: is_user_creator(biography, self.keywords))
        return results.to_numpy().reshape(-1,1)