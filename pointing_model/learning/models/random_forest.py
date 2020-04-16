from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ..model_interface import SkLearnPointingMlModel


class RandomForest(SkLearnPointingMlModel):

    def model(self, **kwargs):
        n_estimators = kwargs.get('n_estimators', 150)
        max_features = kwargs.get('max_features', 'auto')
        max_depth = kwargs.get('max_depth', 17)
        criterion = kwargs.get('criterion', 'entropy')
        random_state = kwargs.get('random_state', 0)
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, criterion=criterion,
            random_state=random_state
        )

    @property
    def name(self):
        return 'Random Forest'

    @property
    def gridsearch_params(self):
        return [
            {
                'n_estimators': [100, 125, 150, 175],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': range(10, 20),
                'criterion': ['entropy'],
                'random_state': [0]
            }
        ]


class RandomForestRegression(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    def model(self, **kwargs):
        n_estimators = kwargs.get('n_estimators', 200)
        max_features = kwargs.get('max_features', 'auto')
        max_depth = kwargs.get('max_depth', 13)
        criterion = kwargs.get('criterion', 'mse')
        random_state = kwargs.get('random_state', 0)
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, criterion=criterion,
            random_state=random_state
        )

    def preprocess(self, X, y=None):
        return X, y

    @property
    def name(self):
        return 'Random Forest Regression'

    @property
    def gridsearch_params(self):
        return [
            {
                'n_estimators': [100, 125, 150, 175],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': range(10, 20),
                'criterion': ['mse', 'mae'],
                'random_state': [0]
            },
            # {
            #     'n_estimators': [175],
            #     'max_features': ['auto'],
            #     'max_depth': [11],
            #     'criterion': ['mse'],
            #     'random_state': [0]
            # }
        ]
