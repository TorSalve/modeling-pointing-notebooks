from sklearn import linear_model, multioutput
from ..model_interface import SkLearnPointingMlModel


class LogisticRegression(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    # 17.01 precision & recall {'C': 1, 'tol': 0.0001, 'penalty': 'l2'}
    def model(self, **kwargs):
        C = kwargs.get('C', 1)
        tol = kwargs.get('tol', 1e-4)
        penalty = kwargs.get('penalty', 'l2')
        return multioutput.MultiOutputRegressor(
            linear_model.LogisticRegression(C=C, tol=tol, penalty=penalty)
        )

    @property
    def name(self):
        return 'Logistic Regression'

    @property
    def gridsearch_params(self):
        return [
            {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'tol': [1e-2, 1e-3, 1e-4, 1e-5],
                'C': [1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000]
            },
        ]
