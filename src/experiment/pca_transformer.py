import pandas as pd
from attr import dataclass
from sklearn.decomposition import PCA


class PCATransformer:
    def __init__(self, n_components: int, cols=None):
        self.cols = cols
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X: pd.DataFrame):
        fitted_cols = self.pca.fit(X[self.cols])
        X[self.cols] = fitted_cols
        return X

    def inverse_transform(self, X: pd.DataFrame):
        """
        Inverts the transform on predicted values
        :param X: the original dataframe
        :return: The inverted PCA values
        """
        return self.pca.inverse_transform(X)