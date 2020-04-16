# import torch
import pandas as pd
import numpy as np
import pointing_model.utils as utils
import pointing_model.plotting as plotting
from sklearn import model_selection, metrics, preprocessing, decomposition
from scipy import stats
import warnings
import matplotlib.pyplot as plt


class PointingMlModel():

    def __init__(self, X, y):
        self.X, self.y = X, y

    @property
    def name(self):
        return type(self.trained_model).__name__

    @property
    def short_name(self):
        return ''.join([c for c in self.name if c.isupper()])

    def model(self, **kwargs):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def plot_confusion_matrix(self, y_test, y_pred, labels, classes, **kwargs):
        matrix = metrics.confusion_matrix(y_test, y_pred, labels=classes)
        matrix = pd.DataFrame(matrix, columns=classes, index=classes)
        matrix.index.name, matrix.columns.name = 'Actual', 'Predicted'
        kwargs['xlabel'] = 'Predicted target'
        default_kwargs = plotting.learning_plots().get('confusion_matrix', {})
        kwargs = {**default_kwargs, **kwargs}
        return plotting.plot_learning_confusion_matrix(matrix, **kwargs)


class PyTorchPointingMlModel(PointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y)
        # self.torch_dtype = torch.float
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.torch_device = torch.device(device)


class TFPointingMlModel(PointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y)


def encode_target(xs):
    return str(tuple(xs))
    # return xs['trueTarget.X']
    # return xs['trueTarget.Y']
    # return xs['trueTarget.Z']


class SkLearnPointingMlModel(PointingMlModel):

    def __init__(self, X, y, _type='classification', **kwargs):
        super().__init__(X, y)

        # allowed = utils.all_features()
        # for x in X.columns:
        #     if x not in allowed:
        #         warnings.warn("some columns in X are not in features: %s" % x)
        # X = X.loc[:, X.columns.isin(utils.all_features())]

        self.verbose = kwargs.get('verbose', 0)
        self.target_fields = kwargs.get('target_fields', utils.target_fields())
        self.type = _type
        self.available_targets = kwargs.get(
            'available_targets', utils.get_targets()
        )
        test_size = kwargs.get('test_size', .2)

        if self.type == 'classification':
            self.labelEncoder = preprocessing.LabelEncoder()
            y['target'] = y[self.target_fields]\
                .apply(lambda xs: encode_target(xs), axis=1)
            self.enc_y = self.labelEncoder.fit(y['target'])
        elif self.type == 'classification_multi':
            self.labelEncoderX = preprocessing.LabelEncoder()
            self.labelEncoderX.fit(y['trueTarget.X'])
            self.labelEncoderY = preprocessing.LabelEncoder()
            self.labelEncoderY.fit(y['trueTarget.Y'])
            self.labelEncoderZ = preprocessing.LabelEncoder()
            self.labelEncoderZ.fit(y['trueTarget.Z'])

        self.standardScaler = preprocessing.StandardScaler()
        self.enc_X = self.standardScaler.fit(X.values)

        X, y = self._preprocess(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test =\
            model_selection.train_test_split(
                X, y, test_size=test_size, random_state=0
            )

        self.trained_model = self.model(**kwargs)

        # self.pca = decomposition.PCA(n_components=.99)
        # self.pca.fit(self.X_train)

    def predict(self, X):
        # X, _ = self._preprocess(X)
        return self.trained_model.predict(X)

    def _preprocess(self, X, y=None):
        if y is not None:
            if self.type == 'classification':
                if 'target' not in y:
                    y['target'] = y[self.target_fields]\
                        .apply(lambda xs: encode_target(xs), axis=1)
                y = self.enc_y.transform(y['target'])
            elif self.type == 'regression':
                y = y[self.target_fields].values
            elif self.type == 'classification_multi':
                y = y[utils.target_fields()].values.astype(float)
                y[:, 0] = self.labelEncoderX.transform(y[:, 0])
                y[:, 1] = self.labelEncoderY.transform(y[:, 1])
                y[:, 2] = self.labelEncoderZ.transform(y[:, 2])
            elif self.type == 'RFE':
                X, y = self.rfe_model._preprocess(X, y)

        # fields = utils.difference(
        #     utils.all_features(),
        #     [
        #         'above_head', 'above_hand', 'indexfinger_body_position_x',
        #         'indexfinger_body_position_y', 'indexfinger_body_position_z',
        #     ]
        # )
        # median = X[fields].median()gridsearch
        # std = X[fields].std()
        # outliers = (X[fields] - median).abs() > 2 * std
        # X[outliers] = np.nan
        # X = X.fillna(X.mean())

        # q1 = X.quantile(.25)
        # q3 = X.quantile(.75)
        # iqr = q3 - q1
        # outliers =\
        #   (X[fields] < (q1 - 1.5 * iqr))\
        #   | (X[fields] > (q3 + 1.5 * iqr))
        # X[outliers] = np.nan
        # X = X.fillna(X.median())

        # z = np.abs(stats.zscore(X[fields]))
        # outliers = np.where(z > 3)
        # X.values[outliers] = np.nan
        # X = X.fillna(X.median())

        # print(X.isna().sum()), exit()

        if self.type != 'RFE':
            X = self.standardScaler.transform(X.values)

        # median = np.median(X, axis=0)
        # print(median), exit()
        # std = np.std(X, axis=0)
        # outliers = np.where(
        #     np.abs(np.subtract(X, median)) > 2 * std
        # )
        # X[outliers] = np.take(median, outliers[1])
        # print(
        #     median, std, outliers,
        #     X[outliers], X[outliers].shape,
        #     X.shape,
        # sep='\n'), exit()

        return X, y

    def train(self):
        self.trained_model.fit(self.X_train, self.y_train)

    def validate(self, **kwargs):
        X = self.X_test
        y_test, y_pred = self.y_test, self.predict(X)

        if self.type == 'classification':
            self.print_classification_report(y_test, y_pred, **kwargs)
        elif self.type == 'regression':
            self.print_regression_report(y_test, y_pred, **kwargs)

    def recursive_feature_elimination(self, **kwargs):
        rfe_selector = self.model(**kwargs)
        rfe_selector.fit(self.X_train, self.y_train)

        print('--------------------------')
        print(rfe_selector.support_)
        print(rfe_selector.ranking_)
        print('--------------------------')
        rfe_support = rfe_selector.get_support()
        rfe_feature = self.X.loc[:, rfe_support].columns.tolist()
        print(rfe_feature)
        print(str(len(rfe_feature)), 'selected features')
        print('--------------------------')

        if self.name.endswith('CV'):
            print("Optimal number of features : %d" % rfe_selector.n_features_)
            # Plot number of features VS. cross-validation scores
            # plt.figure()
            # plt.xlabel("Number of features selected")
            # plt.ylabel(
            #     "Cross validation score (nb of correct classifications)"
            # )
            # plt.plot(
            #     range(1, len(rfe_selector.grid_scores_) + 1),
            #     rfe_selector.grid_scores_
            # )
            # plt.show()

        # rfe_feature = X.loc[:, rfe_support].columns.tolist()
        # print(str(len(rfe_feature)), 'selected features')

    def better_kfold_cross_validation(self, **kwargs):
        n_splits = kwargs.get('n_splits', 5)
        print_report = kwargs.get('print_report', False)
        base_save_path = (
            "./export/content/pointing_movement/figures/generated/"
            + "learning"
        )
        base_save_path = kwargs.get('base_save_path', base_save_path)
        clf = self.model(**kwargs)
        clf.fit(self.X_train, self.y_train)

        kf = model_selection.KFold(
            n_splits=n_splits, shuffle=True, random_state=0
        )

        scores = {}

        if not self.type == 'classification_multi':
            scoring = 'neg_root_mean_squared_error' \
                if self.type == 'regression' \
                else 'accuracy'

            cv = np.mean(
                np.abs(
                    model_selection.cross_val_score(
                        clf, self.X_train, self.y_train, cv=kf,
                        scoring=scoring
                    )
                )
            )
            print("%s-fold cross validation score: %s" % (n_splits, cv))
            scores['cross-validation'] = cv
        y_pred = clf.predict(self.X_test)

        if self.type == 'classification':
            self.print_classification_report(self.y_test, y_pred, **kwargs)

            ys = range(len(self.enc_y.classes_))
            f1_scores = metrics.f1_score(
                self.y_test, y_pred, labels=ys,
                average=None
            )
            d = dict(zip(self.enc_y.classes_, f1_scores))
            plt_kws = {
                "title": (
                    'Targets colored according to their F1'
                    + ' score archieved by a %s.'
                    % self.name
                ),
                "save_path": ( 
                    "%s/plot_f1score_targets_%s.png"
                    % (base_save_path, self.short_name)
                ),
                # 'cbar_label': 'F1 score',
                'vmax': None,
                'vmin': None,
            }
            accuracy = metrics.accuracy_score(self.y_test, y_pred)
            scores['accuracy'] = accuracy
            f1_score = metrics.f1_score(self.y_test, y_pred, average='macro')
            scores['f1-score'] = f1_score
            # print("accuracy: %s" % accuracy)
            # print("f1_score: %s" % f1_score)
            # return
        elif self.type == 'regression':
            self.print_regression_report(self.y_test, y_pred, scores=scores, **kwargs)

            distances = utils.distance(
                np.array(self.y_test).astype(float),
                np.array(y_pred).astype(float)
            )
            # scores['mean-distance'] = np.mean(distances)

            rmse = metrics.mean_squared_error(
                self.y_test, y_pred, squared=False
            )
            scores['rmse'] = rmse

            keys = ['true', 'pred']
            positions = utils.get_positions_from_targets(self.target_fields)
            columns = [(k, p) for k in keys for p in positions]
            t = y_pred.shape[1] if len(y_pred.shape) >= 2 else 1
            s = self.y_test.shape[1] if len(self.y_test.shape) >= 2 else 1
            y_pred = y_pred.reshape(self.y_test.shape)
            rmse = pd.DataFrame(
                np.hstack((self.y_test, y_pred)).flatten().reshape((-1, s+t)),
                columns=pd.MultiIndex.from_tuples(columns)
            )\
                .groupby([('true', p) for p in positions])\
                .apply(lambda xs: metrics.mean_squared_error(
                    xs['true'][positions], xs['pred'][positions],
                    squared=False
                ))\
                .to_dict()

            d = {
                str(utils.float_or_list_to_tuple(k)): v
                for k, v in rmse.items()
            }

            plt_kws = {
                "save_path": (
                    "%s/plot_rmse_targets_%s.png"
                    % (base_save_path, self.short_name)
                ),
                "title": (
                    'Targets colored according to their '
                    + 'RMSE to the predicted target (using %s).'
                    % self.name
                ),
                # 'cbar_label': '[cm]'
                'vmax': None,
                'vmin': None,
            }

        plotting.plot_target_grid(
            fscores=d, label_target=False, save=True,
            cmap='rainbow', available_targets=self.available_targets,
            target_fields=self.target_fields,
            **plt_kws
        )
        return scores

    def kfold_cross_validation(self, **kwargs):
        n_splits = kwargs.get('n_splits', 5)
        print_report = kwargs.get('print_report', False)
        kf = model_selection.KFold(
            n_splits=n_splits, shuffle=True, random_state=0
        )
        i = 1
        scores = []
        fscores = {}
        classes = self.labelEncoder.classes_
        c_matrix = np.zeros((len(classes), len(classes)))

        X, y = self._preprocess(self.X, self.y)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.model(**kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print("K-Fold cross validation: fold %s/%s" % (i, n_splits))
            if self.type == 'classification':
                if print_report:
                    self.print_classification_report(
                        y_test, y_pred,
                        score='kfold_%s' % i
                    )

                accuracy = metrics.accuracy_score(y_test, y_pred)
                fscore = metrics.f1_score(y_test, y_pred, average='weighted')
                scores.append([accuracy, fscore])

                precision, recall, fscore, support =\
                    metrics.precision_recall_fscore_support(y_test, y_pred)

                ys = self.enc_y.inverse_transform(y_test)
                for y_t, fs in zip(ys, fscore):
                    fscores.update({y_t: fscores.get(y_t, []) + [fs]})

                c_matrix += metrics.confusion_matrix(
                    y_test, y_pred, labels=range(len(classes))
                )

            elif self.type == 'regression':
                if print_report:
                    self.print_regression_report(y_test, y_pred)
                scores.append([
                    metrics.mean_squared_error(y_test, y_pred, squared=False),
                    metrics.r2_score(y_test, y_pred)
                ])

                distances = utils.distance(
                    np.array(y_test).astype(float),
                    np.array(y_pred).astype(float)
                )
                distances = np.array(
                    np.hstack((
                        y_test,
                        [np.array([round(d, 2)]) for d in distances]
                    ))
                ).flatten().reshape((-1, 4))

                for _x, _y, _z,  in distances:
                    y_t = str((_x, _y, _z))
                    fscores.update({y_t: fscores.get(y_t, []) + [d]})
            i += 1

        if self.type == 'classification':
            print(
                '---------',
                "Scores pr. split: %s" % str(scores),
                "Mean scores (accuracy, f1_score): %s"
                % np.mean(scores, axis=0),
                sep='\n'
            )
            plotting.plot_target_grid(
                fscores=fscores, label_target=False, save=True,
                cmap='rainbow',
                title=(
                    'Targets colored according to their F1'
                    + ' score archieved by a %s.'
                    % self.name
                ),
                save_path=(
                    "./export/content/pointing_movement/figures/generated/"
                    + "learning/plot_f1score_targets_%s.png"
                    % self.short_name
                )
            )
            matrix_kwargs = {
                "save": True,
                "save_path": (
                    "./export/content/pointing_movement/figures/generated/"
                    + "learning/plot_confusion_matrix_kfold_%s.png"
                    % self.short_name
                ),
                "title": (
                    "Confusion matrix of a %s using %s-fold cross validation"
                    % (self.name, n_splits)
                )

            }
            c_matrix = pd.DataFrame(c_matrix, columns=classes, index=classes)
            c_matrix.index.name, c_matrix.columns.name = 'Actual', 'Predicted'
            default_kwargs = plotting.learning_plots()\
                .get('confusion_matrix', {})
            matrix_kwargs = {**default_kwargs, **matrix_kwargs}
            plotting.plot_learning_confusion_matrix(c_matrix, **matrix_kwargs)
        elif self.type == 'regression':
            print(
                '---------',
                "RMSE pr. split: %s" % str(scores),
                "Mean scores (RMSE, R2): %s" % np.mean(scores, axis=0),
                sep='\n'
            )
            plotting.plot_target_grid(
                fscores=fscores, label_target=False, vmax=None, vmin=None,
                cmap='rainbow', save=True,
                title=(
                    'Targets colored according to their '
                    + 'average distance to the predicted target.'
                ),
                save_path=(
                    "./export/content/pointing_movement/figures/generated/"
                    + "learning/plot_distance_targets_%s.png"
                    % self.short_name
                )
            )

    def print_classification_report(self, y_test, y_pred, **kwargs):
        score = kwargs.get('score', None)
        base_save_path = (
            "./export/content/pointing_movement/figures/generated/"
            + "learning"
        )
        base_save_path = kwargs.get('base_save_path', base_save_path)
        save = kwargs.get('save', True)
        y_test = self.enc_y.inverse_transform(y_test)
        y_pred = self.enc_y.inverse_transform(y_pred)
        print('-----------------------')
        print(
            "Report:",
            metrics.classification_report(y_test, y_pred),
            sep='\n'
        )
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        f1_score = metrics.f1_score(y_test, y_pred, average='macro')
        print("F1 score:", f1_score)
        title_addition =\
            " optimized towards %s" % score\
            if score is not None\
            else ''

        file_addition = "_%s" % score if score is not None else '_validation'
        self.plot_confusion_matrix(
            y_test, y_pred, save=save,
            title=(
                "Confusion matrix of %s%s" % (self.short_name, title_addition)
            ),
            save_path=(
                "%s/plot_confusion_%s%s.png"
                % (base_save_path, self.short_name, file_addition)
            ),
        )
        print('-----------------------')
        return f1_score

    def print_regression_report(self, y_test, y_pred, **kwargs):
        base_save_path = (
            "./export/content/pointing_movement/figures/generated/"
            + "learning"
        )
        base_save_path = kwargs.get('base_save_path', base_save_path)
        save = kwargs.get('save', True)
        analyze_distances = kwargs.get('analyze_distances', utils.save_distances)
        scores = kwargs.get('scores', {})

        distances = utils.distance(
            np.array(y_test).astype(float),
            np.array(y_pred).astype(float)
        )

        keys = ['true', 'pred']
        positions = [t.split('.')[1].lower() for t in self.target_fields]
        columns = [(k, p) for k in keys for p in positions] + [('dist', '')]

        y_t = y_test.astype(float)
        ds = np.array([np.array([round(d, 2)]) for d in distances])
        if y_t.shape[1] == 1:
            y_t = y_t.ravel()
            ds = ds.ravel()

        data = np.array(
            np.hstack((y_t, np.round(y_pred.astype(float), 2), ds))
        ).flatten().reshape((-1, len(columns)))
        dist = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))


        csv_path = '%s/%s_raw_distances.csv' % (base_save_path, self.name)
        analyze_distances(dist, csv_path=csv_path, model_name=self.name, save=save, scores=scores)

        truth = [('true', p) for p in positions]
        plotting.boxplot_feature(
            dist, 'dist', rename=False,
            groupby=truth, save=save,
            title=(
                'Boxplot of distance between predicted '
                + 'target and true target (%s)'
                % self.name
            ),
            ylabel='Distance to the predicted target', xlabel='True target',
            save_path=(
                '%s/boxplot_distances_%s.png' % (base_save_path, self.name)
            )
        )

        dist = dist.sort_values(truth).groupby(truth).mean().round(2)
        columns = [(k, p) for k in keys for p in positions]
        data = np.array(
            np.hstack((y_t, np.round(y_pred.astype(float), 2)))
        ).flatten().reshape((-1, len(columns)))
        frame = pd.DataFrame(
            data, columns=pd.MultiIndex.from_tuples(columns)
        )

        def grp_func(xs, func, **kwargs):
            y_pred = xs[[('pred', p) for p in positions]].values
            y_true = xs[[('true', p) for p in positions]].values
            return func(y_true, y_pred, **kwargs)

        print(
            '-----------------------',
            'distances:', dist,
            '***', 'RMSE:',
            metrics.mean_squared_error(y_test, y_pred, squared=False),
            metrics.mean_squared_error(
                y_test, y_pred, multioutput='raw_values', squared=False
            ),
            frame.groupby(truth).apply(
                lambda xs: grp_func(
                    xs, metrics.mean_squared_error, squared=False
                )
            ),
            # '***', 'MSE:',
            # metrics.mean_squared_error(y_test, y_pred),
            # metrics.mean_squared_error(
            #     y_test, y_pred, multioutput='raw_values'
            # ),
            # frame.groupby(truth).apply(
            #     lambda xs: grp_func(xs, metrics.mean_squared_error)
            # ),
            # '***', 'EVS:',
            # metrics.explained_variance_score(y_test, y_pred),
            # metrics.explained_variance_score(
            #     y_test, y_pred, multioutput='raw_values'
            # ),
            # frame.groupby(truth).apply(
            #     lambda xs: grp_func(xs, metrics.explained_variance_score)
            # ),
            # '***', 'MAE:',
            # metrics.mean_absolute_error(y_test, y_pred),
            # metrics.mean_absolute_error(
            #     y_test, y_pred, multioutput='raw_values'
            # ),
            # frame.groupby(truth).apply(
            #     lambda xs: grp_func(xs, metrics.mean_absolute_error)
            # ),
            # '***', 'R2:',
            # metrics.r2_score(y_test, y_pred),
            # metrics.r2_score(
            #     y_test, y_pred, multioutput='raw_values'
            # ),
            # frame.groupby(truth).apply(
            #     lambda xs: grp_func(xs, metrics.r2_score)
            # ),
            '-----------------------',
            sep='\n'
        )

    def plot_confusion_matrix(self, y_test, y_pred, **kwargs):
        labels = range(len(self.labelEncoder.classes_))
        classes = self.labelEncoder.classes_
        # y_pred = self.enc_y.transform(y_pred)
        return super().plot_confusion_matrix(
            y_test, y_pred, labels, classes, **kwargs
        )

    def gridsearch(self, **kwargs):
        if self.type == 'classification':
            scores = ['precision', 'recall']
            ss = []
            for score in scores:
                s, p = self.report_gridsearch(score=score, **kwargs)
                ss.append((s, p))
            return max(ss)
        elif self.type == 'regression':
            return self.report_gridsearch(**kwargs)

    @property
    def gridsearch_params(self):
        return []

    def report_gridsearch(self, score=None, **kwargs):
        if self.type == 'classification':
            print("# Tuning hyper-parameters for %s" % score)
            scoring = '%s_macro' % score
        elif self.type == 'regression':
            print("# Tuning hyper-parameters")
            scoring = None

        X, y = self._preprocess(self.X, self.y)
        if len(self.gridsearch_params) == 0:
            trained_model = self.trained_model
            trained_model.fit(X, y)
        else:
            trained_model = model_selection.GridSearchCV(
                self.trained_model, self.gridsearch_params,
                scoring=scoring, n_jobs=-1, iid=True, cv=5
                # self.verbose
            )
            trained_model.fit(X, y)

            print("Best parameters set found on development set:")
            print(trained_model.best_params_)
            print()

            # print("Grid scores on development set:")
            # means = trained_model.cv_results_['mean_test_score']
            # stds = trained_model.cv_results_['std_test_score']
            # zipped = zip(means, stds, trained_model.cv_results_['params'])
            # for mean, std, params in zipped:
            #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            # print()

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_pred = trained_model.predict(self.X_test)
        if len(self.target_fields) == 1:
            y_pred = y_pred.reshape((-1, 1))

        if self.type == 'classification':
            self.print_classification_report(
                self.y_test, y_pred, score=score, **kwargs
            )
        elif self.type == 'regression':
            self.print_regression_report(self.y_test, y_pred, **kwargs)
        
        print()
        score = 0
        
        if hasattr(trained_model, 'best_score_'):
            score = trained_model.best_score_
        return score, trained_model.best_params_
