from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass