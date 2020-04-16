from pointing_model import features, utils, PointingModelBase
import pandas as pd
import numpy as np
import warnings
import itertools
import copy
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import featuretools as ft
import matplotlib.pyplot as plt


class Features(PointingModelBase):

    def __init__(self):
        super().__init__()
        self.feature_functions = {}
        self.features = {}
        self.__features()
        self.multi_features = {}

    def __features(self):
        functions = {
            # 'completion_time':
            # lambda p, X, c, y: features.compute_field_value(
            #     X, 'completion_time', 'time'
            # ),
            'elbow_flexion':
            lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_elbow_flexion,
                'elbow_flexion'
            ),
            'shoulder_abduction':
            lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_shoulder_abduction,
                'shoulder_abduction'
            ),
            'shoulder_horizontal':
            lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_shoulder_horizontal,
                'shoulder_horizontal'
            ),
            'indexfinger_vertical':
            lambda p, X, c, y: features.compute_field_value(
                X, 'indexfinger_vertical', 'indexfinger.Y'
            ),
            'indexfinger_horizontal':
            lambda p, X, c, y: features.compute_field_value(
                X, 'indexfinger_horizontal', 'indexfinger.X'
            ),
            'indexfinger_depth':
            lambda p, X, c, y: features.compute_field_value(
                X, 'indexfinger_depth', 'indexfinger.Z'
            ),
            'hmd_vertical':
            lambda p, X, c, y: features.compute_field_value(
                X, 'hmd_vertical', 'hmd.Y'
            ),
            'hmd_horizontal':
            lambda p, X, c, y: features.compute_field_value(
                X, 'hmd_horizontal', 'hmd.X'
            ),
            'hmd_depth':
            lambda p, X, c, y: features.compute_field_value(
                X, 'hmd_depth', 'hmd.Z'
            ),
            'rightShoulder_vertical':
            lambda p, X, c, y: features.compute_field_value(
                X, 'rightShoulder_vertical', 'rightShoulder.Y'
            ),
            'rightShoulder_horizontal':
            lambda p, X, c, y: features.compute_field_value(
                X, 'rightShoulder_horizontal', 'rightShoulder.X'
            ),
            'rightShoulder_depth':
            lambda p, X, c, y: features.compute_field_value(
                X, 'rightShoulder_depth', 'rightShoulder.Z'
            ),
            # 'handO_y_relative':
            # lambda p, X, c, y: features.compute_orientation_relative(
            #     X, 'handO.Y', key='handO_y_relative'
            # ),
            # 'indexfingerO_y_relative':
            # lambda p, X, c, y: features.compute_orientation_relative(
            #     X, 'indexfingerO.Y', key='indexfingerO_y_relative'
            # ),
            # 'upperarmO_y_relative':
            # lambda p, X, c, y: features.compute_orientation_relative(
            #     X, 'upperarmO.Y', key='upperarmO_y_relative'
            # ),
            # 'indexfinger_velocity':
            # lambda p, X, c, y: features.compute_velocity(
            #     X, field="indexfinger"
            # ),
            'above_head': lambda p, X, c, y: features.compute_above_head(
                X, 'above_head'
            ),
            'above_hand': lambda p, X, c, y: features.compute_above_hand(
                X, 'above_hand'
            ),
            'indexfinger_body_position_y': lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_indexfinger_body_position_y,
                key='indexfinger_body_position_y'
            ),
            'indexfinger_body_position_x': lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_indexfinger_body_position_x,
                key='indexfinger_body_position_x'
            ),
            'indexfinger_body_position_z': lambda p, X, c, y:
                features.compute(
                    p, X, c, y, features.compute_indexfinger_body_position_z,
                    key='indexfinger_body_position_z', og_p=self.participants
                ),
            'indexfinger_body_position_z': lambda p, X, c, y:
                features.compute(
                    p, X, c, y, features.compute_indexfinger_body_position_z2,
                    key='indexfinger_body_position_z',
                ),
            'shoulder_xz_slope': lambda p, X, c, y: features.compute(
                p, X, c, y, features.compute_shoulder_xz_slope,
                key='shoulder_xz_slope'
            ),
        }

        for key in functions:
            self.add_feature_function(key, functions[key])

        distance_perm = utils.distance_permutations()
        for a, b in distance_perm:
            def distance_function(_key, _a, _b):
                return lambda p, X, c, y:\
                    features.compute_distance(X, _key, _a, _b)

            key = utils.distance_key(a, b)
            self.add_feature_function(key, distance_function(key, a, b))

        orientation_fields = utils.flatten(
            list(map(utils.body_orientation_field, ['upperarm', 'hmd']))
        )
        for field in orientation_fields:
            def orientation_function(_key, _field):
                return lambda p, X, c, y:\
                    features.compute_field_value(X, _key, _field)

            key = utils.field_to_orientation_key(field)
            self.add_feature_function(key, orientation_function(key, field))

    def add_feature_function(self, key, function):
        if key in self.feature_functions:
            warnings.warn("overwriting %s feature function" % key)
        self.feature_functions[key] = function

    def compute_feature(self, participants, X, calibrations, y, key):
        if key not in self.feature_functions:
            # warnings.warn("feature function not found: %s" % key)
            return
        function = self.feature_functions[key]
        self.features[key] = function(
            p=participants, X=X, c=calibrations, y=y
        )
        return self.features[key]

    def attach_target(self, X, y):
        if all(elem in X.columns for elem in utils.target_fields()):
            return X
        Xs = X\
            .reset_index()\
            .merge(
                y.reset_index(), right_on=['id', 'pid'],
                left_on=['cid', 'pid'], how="left"
            )\
            .drop(columns=['id_y'])\
            .rename(columns={'id_x': 'id', 'index_x': 'index'})\
            .set_index('index')
        if 'index_y' in list(Xs.columns):
            Xs = Xs.drop(columns=['index_y'])
        utils.check_dimensions(X, Xs, additional_columns=3, name='target')
        for target in utils.target_fields():
            Xs[target] = Xs[target].astype(float)
        return Xs

    def load_all_features(self, participants, X, calibrations, y, **kwargs):
        exclude = kwargs.get('exclude', [])
        include = kwargs.get('include', utils.all_features())
        for key in include:
            if key not in exclude:
                X = self.attach_feature(participants, X, calibrations, y, key)
        return X

    def attach_feature(self, participants, X, calibrations, y, key):
        if key not in self.features:
            self.compute_feature(participants, X, calibrations, y, key)
        if key in X.columns or key not in self.features:
            return X
        feature = self.features[key]
        Xs = X\
            .reset_index()\
            .merge(
                feature, right_on=['pid', 'cid', 'sid'],
                left_on=['pid', 'cid', 'index'], how="left"
            )\
            .drop(columns=['sid'])\
            .set_index('index')
        additional_columns = len(feature.columns) - 3
        utils.check_dimensions(
            X, Xs, additional_columns=additional_columns, name=key
        )
        if additional_columns > 1:
            mf = list(set(feature.columns) - {'pid', 'cid', 'sid'})
            mf.sort(reverse=True)
            self.multi_features[key] = mf
        return Xs

    def find_best_features(self, participants, X, calibrations, y, **kwargs):
        default_fields =\
            utils.all_body_fields() + utils.all_body_orientation_fields()
        fields = kwargs.get('fields', default_fields)
        load_features = kwargs.get('load_features', False)
        k = kwargs.get('k', 'all')
        regression = kwargs.get('regression', False)
        target_fields = kwargs.get('target_fields', utils.target_fields())

        if load_features:
            X = self.load_all_features(
                participants, X, calibrations, y,
                include=fields, **kwargs
            )

        X = X[self.add_features]
        with pd.option_context('mode.use_inf_as_na', True):
            X = X.fillna(X.mean())
        if any(X.isna().any().values) > 0:
            X = X.fillna(0)

        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        y = y[target_fields]
        if not regression:
            y = y.apply(lambda xs: str(tuple(xs)), axis=1)
            function = chi2
            sKbest = SelectKBest(function, k=k)
            sKbest.fit_transform(Xs, y)
            support = sKbest.get_support(indices=True)

            col_names = X.columns[support].values
            scores = sKbest.scores_[support]
            pvalues = sKbest.pvalues_[support]
            zipped = list(zip(col_names, pvalues))
            zipped.sort(key=lambda t: t[1])

            return pd.DataFrame(
                zipped, columns=['feature', 'p-value']
            ).round(5)
        else:
            scores_dict = {}
            for t in target_fields:
                ys = y[t]
                function = mutual_info_regression

                sKbest = SelectKBest(function, k=k)
                sKbest.fit_transform(Xs, ys)
                support = sKbest.get_support(indices=True)

                col_names = X.columns[support].values
                scores = sKbest.scores_[support]
                zipped = list(zip(col_names, scores))
                zipped.sort(key=lambda t: t[0])

                idx, sorted_scores = list(zip(*zipped))
                scores_dict[t] = sorted_scores

            return pd.DataFrame(scores_dict, index=idx)

    def find_best_features_extratree(
        self, participants, X, calibrations, y, **kwargs
    ):
        from sklearn import ensemble, preprocessing
        default_fields =\
            utils.all_body_fields() + utils.all_body_orientation_fields()
        fields = kwargs.get('fields', default_fields)
        load_features = kwargs.get('load_features', False)

        if load_features:
            X = self.load_all_features(
                participants, X, calibrations, y,
                include=fields
            )

        X = X[self.add_features]
        columns = copy.deepcopy(X.columns)
        with pd.option_context('mode.use_inf_as_na', True):
            X = X.fillna(X.mean())

        labelEncoder = preprocessing.LabelEncoder()
        y['target'] = y[utils.target_fields()]\
            .apply(lambda xs: str(tuple(xs)), axis=1)
        y = labelEncoder.fit_transform(y['target'])

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = ensemble.ExtraTreesClassifier()
        model.fit(X, y)
        feat_importances = pd.Series(
            model.feature_importances_, index=columns
        )
        return feat_importances.sort_values()

    def analyze_all_features(self, path='./feature_analysis', **kwargs):
        base_pairplot = kwargs.get('base_pairplot', False)
        save = kwargs.get('save', True)

        utils.ensure_dir_exists(path)

        participants, X, calibrations, y = self.normalized
        featureX = self.load_all_features(participants, X, calibrations, y)
        featureX = self.attach_target(featureX, y)

        features = utils.all_features()

        describe_path = "%s/describe.csv" % path
        featureX[features].describe().to_csv(describe_path)

        for key in features:
            self.analyze_feature(
                participants, featureX, calibrations, y,
                key, path=path, **kwargs
            )

        self.plot_correlation_matrix(
            force=True, save=True, additional_fields=features,
            save_path="%s/correlation_matrix.png" % path
        )

        self.plot_correlation_matrix(
            force=True, save=True,
            additional_fields=features + utils.target_fields(),
            save_path="%s/correlation_matrix_with_targets.png" % path
        )

        self.plot_correlation_matrix(
            force=True, save=True,
            base_fields=features + utils.target_fields(),
            save_path="%s/correlation_matrix_features.png" % path
        )

        self.plot_correlation_matrix(
            force=True, save=True,
            include=list(self.feature_functions.keys()),
            base_fields=(
                list(self.feature_functions.keys()) + utils.target_fields()
            ),
            save_path=(
                "%s/correlation_matrix_all_features_with_targets.png" % path
            )
        )

        self.plot_correlation_matrix(
            force=True, save=True, additional_fields=utils.target_fields(),
            save_path="%s/correlation_matrix_base.png" % path
        )

        self.plot_pca_coefficients(
            force=True, save=True, base_fields=utils.all_body_fields(),
            save_path="%s/pca_coefficients_base.png" % path
        )

        self.plot_pca_coefficients(
            force=True, save=True, base_fields=features,
            save_path="%s/pca_coefficients_features.png" % path
        )

        save_path = "%s/pca_base.png" % path
        utils.ensure_dir_exists(save_path, is_file=True)
        self.plot_pca(
            load_features=False, force=True,
            save=save, save_path=save_path
        )

        save_path = "%s/pca_all.png" % path
        utils.ensure_dir_exists(save_path, is_file=True)
        self.plot_pca(
            include_features=features, force=True,
            save=save, save_path=save_path
        )

        if base_pairplot:
            print('plotting pairplots:')

            print('\t start features')
            self.plot_pairplot(
                force=True, base_fields=features+utils.target_fields(),
                save=save, save_path="%s/pairplot_features.png" % path
            )

            print('\t start base')
            self.plot_pairplot(
                force=True, save=save,
                base_fields=utils.all_body_fields()+utils.target_fields(),
                save_path="%s/pairplot_base.png" % path
            )

            print('\t start orientations')
            self.plot_pairplot(
                force=True, save=save,
                base_fields=(
                    utils.all_body_orientation_fields()
                    + utils.target_fields()
                ),
                save_path="%s/pairplot_orientations.png" % path
            )

    def analyze_feature(
        self, participants, featureX, calibrations, y,
        key, path='./feature_analysis', **kwargs
    ):
        only_final = kwargs.get('only_final', True)
        normalize = kwargs.get('normalize', True)

        utils.ensure_dir_exists(path)

        featureX = self.attach_feature(
            participants, featureX, calibrations, y, key
        )

        keys = [
            'kde_%s' % key,
            'boxplot_%s' % key,
            'count_hist_plot_%s' % key,
            key
        ]
        for k in keys:
            plot_config = self.plotConfig.plotConfig.get(k, None)
            if plot_config is None:
                continue

            plot_config = dict(utils.merge({
                'plt_type': 'feature_boxplot',
                'feature_key': key,
                'X': featureX, 'participants': participants,
                'calibrations': calibrations, 'y': y,
            }, plot_config))

            self.plot(
                save=True, save_path="%s/%s.png" % (path, k),
                **plot_config
            )

    def automatic_features(
        self, participants, X, calibrations, y, model, **kwargs
    ):
        # X = self.attach_target(X, y)
        # X = self.load_all_features(participants, X, calibrations, y)
        X['cid_pid'] = X.apply(
            lambda xs: "%s_%s" % (xs['cid'], xs['pid']), axis=1
        )

        # print(participants.head(), X.head(), y.head(), sep='\n')
        body_types = {
            k: ft.variable_types.Numeric for k in utils.all_body_fields()
        }
        target_types = {
            k: ft.variable_types.Id for k in utils.target_fields()
        }
        vt = {
            'cid_pid': ft.variable_types.Index,
            'id': ft.variable_types.Id,
            'cid': ft.variable_types.Id,
            'pid': ft.variable_types.Id,
            'time': ft.variable_types.Numeric,
            **body_types,
            # **target_types
        }

        es = ft.EntitySet(id="pointing_movement")

        es = es.entity_from_dataframe(
            entity_id='collections',
            dataframe=X, index='cid_pid',
            variable_types=vt
        )

        selected = ['hmd', 'indexfinger']
        for f in utils.flatten(map(utils.body_field, selected)):
            es = es.normalize_entity(
                base_entity_id='collections',
                new_entity_id=f, index=f
            )

        # es.plot(to_file='./plot.png')

        feature_matrix, features_defs = ft.dfs(
            entityset=es, entities=es,
            target_entity="collections", verbose=1
        )

        # feature_matrix.to_csv('./export.csv')

        # feature_matrix = self.load_all_features(
        #     participants, feature_matrix, calibrations, y
        # )

        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        modl = model(feature_matrix, y)
        # modl.gridsearch()
        modl.better_kfold_cross_validation()

        # # reg: \[([^\]]+)\]\n5-fold cross validation score: ([0-9]*\.?[0-9]*)\naccuracy: ([0-9]*\.?[0-9]*)\nf1_score: ([0-9]*\.?[0-9]*)\n==========`====
        # perms = utils.unique_cominations(utils.all_features())
        # for p in perms:
        #     for i in list(p):

        #         feature_matrix = self.load_all_features(
        #             participants, X, calibrations, y, include=list(i)
        #         )

        #         feature_matrix = feature_matrix.fillna(feature_matrix.mean())
        #         if any(feature_matrix.isna().any().values) > 0:
        #             feature_matrix = feature_matrix.fillna(0)
        #         modl = model(feature_matrix, y)
        #         # model.gridsearch()
        #         print(list(i))
        #         modl.better_kfold_cross_validation()
        #         print('==========`====')
