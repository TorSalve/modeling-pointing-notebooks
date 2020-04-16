from sklearn.feature_selection import RFE, RFECV
from ..model_interface import SkLearnPointingMlModel
from sklearn.model_selection import StratifiedKFold


class RecursiveFeatureElimination(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        model = kwargs.get('rfe_model', None)
        if model is None:
            raise Exception('add RFE ml model')
        self.rfe_model = model(X, y)
        if not isinstance(self.rfe_model, SkLearnPointingMlModel):
            raise Exception('RFE ml model not of right type')

        super().__init__(X, y, _type='RFE', **kwargs)

    def model(self, **kwargs):
        n_features_to_select = kwargs.get('n_features_to_select', 40)
        step = kwargs.get('step', 1)
        return RFE(
            estimator=self.rfe_model.model(**kwargs),
            verbose=5,
            step=step,
            n_features_to_select=n_features_to_select
        )

    def preprocess(self, X, y=None):
        return X, y

    @property
    def name(self):
        return 'Recursive Feature Elimination'


class RecursiveFeatureEliminationCV(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        model = kwargs.get('rfe_model', None)
        if model is None:
            raise Exception('add RFE ml model')
        self.rfe_model = model(X, y)
        if not isinstance(self.rfe_model, SkLearnPointingMlModel):
            raise Exception('RFE ml model not of right type')

        super().__init__(X, y, _type='RFE', **kwargs)

    def model(self, **kwargs):
        n_features_to_select = kwargs.get('n_features_to_select', 40)
        step = kwargs.get('step', 1)
        kfold = kwargs.get('k_fold', 5)
        return RFECV(
            estimator=self.rfe_model.model(**kwargs),
            step=step,
            cv=StratifiedKFold(kfold),
            verbose=10
        )

    def preprocess(self, X, y=None):
        return X, y

    @property
    def name(self):
        return 'Recursive Feature Elimination CV'