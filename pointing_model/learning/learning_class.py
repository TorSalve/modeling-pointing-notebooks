
from pointing_model import PointingModelBase, utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import copy
# from .simple_nn import SimpleNN
from .model_interface import PointingMlModel, SkLearnPointingMlModel
from .models import *


class Learning(PointingModelBase):

    def __init__(self):
        super().__init__()
        self.pca = None
        self.correlation_matrix = None
        self.pairplot_data = None

    def preprocess(
        self, participants=None, X=None,
        calibrations=None, y=None, features=None, **kwargs
    ):
        use_preloaded = kwargs.get('use_preloaded', True)
        use_provided = kwargs.get('use_provided', True)
        exclude_features = kwargs.get('exclude_features', [])
        if features is None:
            features = kwargs.get('include', None)
        features = features if features is not None else utils.all_features()
        kwargs['include'] = features

        if use_preloaded:
            participants, X, calibrations, y = self.normalized
        elif use_provided:
            pass
        else:
            participants, X, calibrations, y = self.normalize_data(
                participants, X, calibrations, y, **kwargs
            )
        X = self.load_all_features(participants, X, calibrations, y, **kwargs)
        if y is not None:
            X = self.attach_target(X, y)

        if self.use_dynamic_features:
            X = utils.only_endpoints(X)

        add_features = []
        for f in features:
            add_features += self.multi_features.get(f, [f])
        self.add_features = [
            f for f in add_features
            if f not in exclude_features
        ]

        # print(self.add_features), exit()
        # self.add_features = list(set(add_features) - set(exclude_features))

        X = X.set_index('cid')[self.add_features]
        X = X.fillna(X.mean())
        if any(X.isna().any().values) > 0:
            X = X.fillna(0)
        return participants, X, calibrations, y

    def machine_learning_gridsearch(self, model, **kwargs):
        kwargs['gridsearch'] = True
        kwargs['validate'] = False
        kwargs['kfold'] = False
        return self.machine_learning_training(model, **kwargs)
    
    def machine_learning_validate(self, model, **kwargs):
        kwargs['gridsearch'] = False
        kwargs['validate'] = True
        kwargs['kfold'] = False
        return self.machine_learning_training(model, **kwargs)
    
    def machine_learning_kfold(self, model, **kwargs):
        kwargs['gridsearch'] = False
        kwargs['validate'] = False
        kwargs['kfold'] = True
        return self.machine_learning_training(model, **kwargs)

    def machine_learning_training(self, model, **kwargs):
        validate = kwargs.get('validate', False)
        gridsearch = kwargs.get('gridsearch', False)
        kfold = kwargs.get('kfold', False)
        rfe = kwargs.get('rfe', False)
        features = kwargs.get('features', None)
        ml_kwargs = kwargs.get('ml', self.config.get('ml'))
        cleanup = kwargs.get('cleanup', False)
        model_kwargs = kwargs.get('model_kwargs', {})
        preprocess_kwargs = kwargs.get('preprocess_kwargs', {})

        if cleanup:
            self.model = None

        _, X, _, y = self.preprocess(include=features, **preprocess_kwargs)
        # participants, X, calibrations, y = self.normalized
        # X = X.drop(columns=['id', 'pid', 'cid', 'time'])
        y = y.drop(columns=['pid'])

        self.model = model(X, y, **ml_kwargs, **model_kwargs)
        # if not isinstance(self.model, PointingMlModel):
        #     raise Exception('ml model not of right type: %s' % type(self.model))

        print('', '****', '', sep='\n')
        print("Using:", self.model.name)

        if gridsearch:
            _, p = self.model.gridsearch(**kwargs)
            return self.model, p
        elif kfold:
            scores = self.model.better_kfold_cross_validation(**kwargs)
            return self.model, scores
        elif rfe:
            if self.model.type != 'RFE':
                raise Exception(
                    'ml model not of type RFE, but of %s' % self.model.type
                )
            self.model.recursive_feature_elimination(**kwargs)

        if validate:
            self.model.train()
            self.model.validate(**kwargs)

        return self.model

    def machine_learning_predict(self, participants, X, calibrations):
        _, X, _, _ =\
            self.preprocess(participants, X, calibrations, use_preloaded=False)
        return self.model.predict(X)

    def compute_pca(self, **kwargs):
        base_fields = kwargs.get('base_fields', utils.all_body_fields())
        n_components = kwargs.get('n_components', 'mle')
        exclude_features = kwargs.get('exclude_features', [])
        include_features = kwargs.get(
            'include_features', utils.all_features()
        )
        load_features = kwargs.get('load_features', True)

        fields = base_fields
        if load_features:
            feature_functions = {*include_features} - set(exclude_features)
            fields = list(set(list(feature_functions) + fields))

        _, X, _, y = self.preprocess(
            features=fields, include=fields
        )
        X = X.fillna(X.mean())
        if any(X.isna().any().values) > 0:
            X = X.fillna(0)

        X = X[self.add_features].values
        X = StandardScaler().fit_transform(X)
        # print('computing new PCA, fields: %s' % ', '.join(fields))
        pca = PCA(n_components=n_components)
        self.pca = pca.fit(X)
        return self.pca, fields

    def compute_correlation_matrix(self, **kwargs):
        additional_fields = kwargs.get('additional_fields', [])
        method = kwargs.get('corr_method', 'pearson')
        base_fields = kwargs.get('base_fields', utils.all_body_fields())
        fields = base_fields + additional_fields
        kwargs['features'] = fields
        _, X, _, _ = self.preprocess(include=base_fields, **kwargs)
        self.correlation_matrix = X.corr(method=method)

    def compute_pairplot(self, **kwargs):
        additional_fields = kwargs.get('additional_fields', [])
        base_fields = kwargs.get('base_fields', utils.all_body_fields())
        fields = base_fields + utils.target_fields() + additional_fields
        _, X, _, y = self.preprocess(features=fields)
        X['target'] = X[utils.target_fields()]\
            .apply(lambda xs: tuple(xs), axis=1)
        self.pairplot_data = X.drop(columns=utils.target_fields())

    @property
    def classification_models(self):
        return [
            SupportVectorMachine,
            # KNearestNeighbors,
            RandomForest,
            # Perceptron,
            # MultiLayerPerceptron,
            NaiveBayes,
        ]
    
    @property
    def regression_models(self):
        return [
            SupportVectorMachineRegression,
            # KNearestNeighborsRegressor,
            RandomForestRegression,
            LinearRegression,
            # StochasticGradientDescentRegression,
        ]