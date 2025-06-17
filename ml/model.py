import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBWrapper(BaseEstimator, ClassifierMixin):
    __estimator_type__ = "classifier"  # <- Fix: tells sklearn this is a classifier

    def __init__(self, booster=None):
        self.booster = booster
        self._fitted = booster is not None
        self.feature_names = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        self.booster = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=100)
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise ValueError("Wrapper not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        prob = self.booster.predict(dmatrix)
        return np.vstack([1 - prob, prob]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.booster:
            self.booster.save_model(path)

    def load_model(self, path: str):
        self.booster = xgb.Booster()
        self.booster.load_model(path)
        self._fitted = True

    def get_feature_importance(self):
        if not self._fitted:
            return {}
        return self.booster.get_score(importance_type='gain')
