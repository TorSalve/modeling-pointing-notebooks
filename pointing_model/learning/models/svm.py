from sklearn import svm, multioutput
from ..model_interface import SkLearnPointingMlModel


class SupportVectorMachine(SkLearnPointingMlModel):

    # {'C': 30000, 'gamma': 0.0001, 'kernel': 'rbf'}
    def model(self, **kwargs):
        C = kwargs.get('C', 20000)
        gamma = kwargs.get('gamma', 1e-3)
        kernel = kwargs.get('kernel', 'rbf')

        # C = kwargs.get('C', 10)
        # kernel = kwargs.get('kernel', 'linear')

        # C = kwargs.get('C', 30000)
        # gamma = kwargs.get('gamma', 0.0001)
        # kernel = kwargs.get('kernel', 'rbf')
        return svm.SVC(C=C, gamma=gamma, kernel=kernel)

    @property
    def name(self):
        return 'SVM'

    @property
    def gridsearch_params(self):
        return [
            {
                'kernel': ['rbf'],
                'gamma': [1e-2, 1e-3, 1e-4],
                'C': [
                    1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000
                ]
            },
            {
                'kernel': ['linear'],
                'C': [1, 10, 100, 1000, 5000, 10000]
            }
        ]


class SupportVectorMachineMultiOutput(SupportVectorMachine):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='classification_multi', **kwargs)

    def model(self, **kwargs):
        return multioutput.MultiOutputClassifier(
            super().model(**kwargs)
        )

    @property
    def name(self):
        return "%s-mo" % super().name

    @property
    def gridsearch_params(self):
        return [
            {
                'estimator__kernel': ['rbf'],
                'estimator__gamma': [1e-2, 1e-3, 1e-4],
                'estimator__C': [
                    1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000
                ]
            },
            {
                'estimator__kernel': ['linear'],
                'estimator__C': [1, 10, 100, 1000, 5000, 10000]
            }
        ]


class SupportVectorMachineRegression(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    def model(self, **kwargs):
        C = kwargs.get('C', 100)
        gamma = kwargs.get('gamma', 0.01)
        kernel = kwargs.get('kernel', 'rbf')
        return multioutput.MultiOutputRegressor(
            svm.SVR(C=C, gamma=gamma, kernel=kernel)
        )

    @property
    def name(self):
        return 'SVM'

    @property
    def gridsearch_params(self):
        return [
            {
                'estimator__kernel': ['rbf'],
                'estimator__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                'estimator__C': [10e0, 10e1, 10e2]
            },
            # {
            #     'estimator__kernel': ['linear'],
            #     'estimator__C': [10e0, 10e1, 10e2]
            # }
        ]
