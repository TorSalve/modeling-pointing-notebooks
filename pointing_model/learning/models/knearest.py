from sklearn import neighbors, multioutput
from ..model_interface import SkLearnPointingMlModel


class KNearestNeighbors(SkLearnPointingMlModel):

    # 17.01 recall & presicion {'n_neighbors': 1}
    def model(self, **kwargs):
        n_neighbors = kwargs.get('n_neighbors', 1)
        return neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

    @property
    def name(self):
        return 'K-Nearest Neighbors'

    @property
    def gridsearch_params(self):
        return [
            {
                'n_neighbors': range(1, 11),
            }
        ]


class KNearestNeighborsRegressor(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    # 20.01 recall & presicion {'estimator__n_neighbors': 2}
    def model(self, **kwargs):
        n_neighbors = kwargs.get('n_neighbors', 2)
        return multioutput.MultiOutputRegressor(
            neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
        )

    @property
    def name(self):
        return 'K-Nearest Neighbors'

    @property
    def gridsearch_params(self):
        return [
            {
                'estimator__n_neighbors': range(1, 11),
            }
        ]
