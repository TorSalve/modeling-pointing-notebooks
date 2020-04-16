from sklearn import linear_model
from ..model_interface import SkLearnPointingMlModel


class Perceptron(SkLearnPointingMlModel):

    # {'alpha': 0.1, 'max_iter': 100.0, 'random_state': 0, 'tol': 0.01}
    def model(self, **kwargs):
        alpha = kwargs.get('alpha', 0.1)
        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 0.01)
        return linear_model.Perceptron(
            alpha=alpha, max_iter=max_iter, tol=tol
        )

    @property
    def name(self):
        return 'Perceptron'

    @property
    def gridsearch_params(self):
        return [
            {
                'alpha': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                'max_iter': [1, 10, 10e1, 10e2, 10e3],
                'tol': [1e-2, 1e-3, 1e-4],
                'random_state': [0]
            }
        ]
