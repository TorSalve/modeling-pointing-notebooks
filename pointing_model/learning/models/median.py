import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import operator
from ..model_interface import SkLearnPointingMlModel


class MedianClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        unique, counts = np.unique(self.y_, return_counts=True)
        counts = dict(zip(unique, counts))
        return max(counts.items(), key=operator.itemgetter(1))[0]


class Median(SkLearnPointingMlModel):

    def model(self, **kwargs):
        return MedianClassifier()

    @property
    def name(self):
        return 'Median'

    @property
    def gridsearch_params(self):
        return []
