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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def is_user_creator(biography, keywords):
    # Tokenize the biography and convert to lower case
    tokens = nltk.word_tokenize(biography.lower())

    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Check if any of the tokens are in your list of keywords
    return any(token in keywords for token in tokens)


# Define a string match transformer class
class BiographyTransformer(BaseEstimator, TransformerMixin):
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


# Define a PCA transformer class
class PCATransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with a standardscaler and pca
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.pca = PCA()

    # Fit method
    def fit(self, X, y=None):
        self.pca.fit(self.standard_scaler.fit_transform(X))
        return self

    # Transform method to return the principal components 
    def transform(self, X):
        return self.pca.transform(self.standard_scaler.transform(X))


# Define a missing value identifier class
class MissingValueIdentifier(BaseEstimator, TransformerMixin):
    # Initialize the class with a feature name
    def __init__(self, feature_name):
        self.feature_name = feature_name

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to check if the column values are null  
    def transform(self, X):
        return X[self.feature_name].isna().to_numpy().reshape(-1, 1)