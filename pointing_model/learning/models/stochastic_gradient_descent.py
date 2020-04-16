from sklearn import linear_model, multioutput
from ..model_interface import SkLearnPointingMlModel


class StochasticGradientDescent(SkLearnPointingMlModel):

    def model(self, **kwargs):
        loss = kwargs.get('loss', 'hinge')
        penalty = kwargs.get('penalty', 'l2')
        max_iter = kwargs.get('max_iter', 5)
        return linear_model.SGDClassifier(
            penalty=penalty, loss=loss, max_iter=max_iter
        )

    @property
    def name(self):
        return 'SGD'

    @property
    def gridsearch_params(self):
        return [
            {
                'loss': ['hinge', 'log', 'modified_huber'],
                'penalty': ['l2', 'elasticnet'],
                'max_iter': [1, 10, 10e1, 10e2, 10e3],
                'tol': [1e-2, 1e-3, 1e-4],
            }
        ]


class StochasticGradientDescentRegression(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    def model(self, **kwargs):
        loss = kwargs.get('loss', 'squared_loss')
        penalty = kwargs.get('penalty', 'l2')
        max_iter = kwargs.get('max_iter', 5)
        return multioutput.MultiOutputRegressor(
            linear_model.SGDRegressor(
                penalty=penalty, loss=loss, max_iter=max_iter
            )
        )

    @property
    def name(self):
        return 'SGD'

    @property
    def gridsearch_params(self):
        return [
            {
                'estimator__loss': ['squared_loss', 'huber'],
                'estimator__penalty': ['l2', 'elasticnet'],
                'estimator__max_iter': [1, 10, 10e1, 10e2],
                'estimator__tol': [1e-2, 1e-3, 1e-4],
            }
        ]
