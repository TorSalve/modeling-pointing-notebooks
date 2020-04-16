from pointing_model import plotting, utils, PointingModelBase
import pandas as pd


class Plotting(PointingModelBase):

    def __init__(self):
        super().__init__()
        self.plotConfig = plotting.PlotConfig(self.config)

    def plot(self, **kwargs):
        self.reset()
        participant = kwargs.get('participant', -1)
        collection = kwargs.get('collection', -1)
        plt_type = kwargs.get('plt_type', 'projection')
        feature_key = kwargs.get('feature_key', '-')
        data = kwargs.get('data', 'normalized')

        if data == 'normalized':
            participants, featureX, calibrations, y = self.normalized
        elif data == 'normalized_all_snapshots':
            participants, featureX, calibrations, y = self.normalize_data(
                self.participants, self.X, self.calibrations, self.y,
                take_only_final=False
            )
        elif data == 'normalized_time':
            participants, featureX, calibrations, y = self.normalized_time
        else:
            participants = kwargs.get('participants', self.participants)
            featureX = kwargs.get('X', self.X)
            calibrations = kwargs.get('calibrations', self.calibrations)
            y = kwargs.get('y', self.y)

        if participant > 0 and collection > 0:
            Xplt = featureX[
                (featureX.pid == participant) &
                (featureX.cid == collection)
            ]
            yplt = y[y.pid == participant].loc[collection, :]

        if plt_type == 'projection':
            # print(feature_key, data, participant, collection)
            return plotting.plot_projection(Xplt, yplt, **kwargs)
        elif plt_type == 'timeline':
            return plotting.plot_timeline(Xplt, **kwargs)

        featureX = self.attach_target(featureX, y)
        featureX = self.attach_feature(
            participants, featureX, calibrations, y, feature_key
        )

        if 'X' in kwargs:
            del kwargs['X']
        if(plt_type == 'feature_boxplot'):
            return plotting.boxplot_feature(featureX, feature_key, **kwargs)
        elif(plt_type == 'feature_count_hist_plot'):
            return plotting.plot_count_hist(featureX, feature_key, **kwargs)
        elif(plt_type == 'feature_kdeplot'):
            return plotting.kdeplot_feature(featureX, feature_key, **kwargs)
        else:
            return plotting.plot_feature(featureX, feature_key, **kwargs)

    def plot_pca(self, **kwargs):
        self.reset()
        force = kwargs.get('force', False)
        if self.pca is None or force:
            self.compute_pca(**kwargs)
        kwargs = {**self.plotConfig.getConfig('pca'), **kwargs}
        plotting.plot_pca(self.pca, **kwargs)

    def plot_correlation_matrix(self, **kwargs):
        force = kwargs.get('force', False)
        if self.correlation_matrix is None or force:
            self.compute_correlation_matrix(**kwargs)
        kwargs = {**self.plotConfig.getConfig('correlation_matrix'), **kwargs}
        plotting.plot_correlation_matrix(self.correlation_matrix, **kwargs)

    def plot_pairplot(self, **kwargs):
        self.reset()
        force = kwargs.get('force', False)
        if self.pairplot_data is None or force:
            self.compute_pairplot(**kwargs)
        kwargs = {**self.plotConfig.getConfig('pairplot'), **kwargs}
        plotting.plot_pairplot(self.pairplot_data, **kwargs)

    def plot_pca_coefficients(self, **kwargs):
        self.reset()
        kwargs['base_fields'] = kwargs.get(
            'base_fields',
            utils.all_body_fields() + utils.all_body_orientation_fields() +
            utils.all_features()
        )
        kwargs['n_components'] = kwargs.get('n_components', .99)
        pca, fs = self.compute_pca(**kwargs)
        components = pd.DataFrame(pca.components_, columns=fs)
        kwargs = {**self.plotConfig.getConfig('pca_coefficients'), **kwargs}
        plotting.plot_pca_coefficients(components, **kwargs)

    def plot_selectKBest_chi2(self, **kwargs):
        fs = utils.all_body_fields() +\
            utils.all_body_orientation_fields() +\
            utils.all_features()
        fields = kwargs.get('fields', fs)
        kwargs['fields'] = fields
        exclude = kwargs.get('exclude_features', [])
        p, X, c, y = self.preprocess(features=fields, exclude_features=exclude)
        kbest = self.find_best_features(
            p, X, c, y, **kwargs
        )
        if(len(kbest) == 0):
            return
        kwargs = {**self.plotConfig.getConfig('selectKBest'), **kwargs}
        plotting.plot_selectKBest_chi2(kbest, **kwargs)

    def plot_selectKBest_mutual_information(self, **kwargs):
        fs = utils.all_body_fields() +\
            utils.all_body_orientation_fields() +\
            utils.all_features()
        fields = kwargs.get('fields', fs)
        kwargs['fields'] = fields
        exclude = kwargs.get('exclude_features', [])
        p, X, c, y = self.preprocess(features=fields, exclude_features=exclude)
        kbest = self.find_best_features(
            p, X, c, y, regression=True, **kwargs
        )
        if(len(kbest) == 0):
            return
        kwargs = {**self.plotConfig.getConfig('selectKBest_mi'), **kwargs}
        plotting.plot_selectKBest_mutual_information(kbest, **kwargs)

    def plot_extratrees(self, **kwargs):
        fs = utils.all_body_fields() +\
            utils.all_body_orientation_fields() +\
            utils.all_features()
        fields = kwargs.get('fields', fs)
        kwargs['fields'] = fields
        exclude = kwargs.get('exclude_features', [])
        p, X, c, y = self.preprocess(features=fields, exclude_features=exclude)
        kbest = self.find_best_features_extratree(
            p, X, c, y, **kwargs
        )
        kwargs = {**self.plotConfig.getConfig('extratrees'), **kwargs}
        plotting.plot_extratrees(kbest, **kwargs)

    def plot_config(self, plot_key, **kwargs):
        self.reset()
        # print(plot_key)
        self.plot(**{**self.plotConfig.getConfig(plot_key), **kwargs})

    def plot_all_loaded(self, **kwargs):
        for particiant in self.get_loaded_participant_ids():
            collections = self.y[self.y.pid == particiant].index
            for collection in collections:
                self.plot(particiant, collection, **kwargs)

    def plot_projection(self, participant, collection, **kwargs):
        self.plot_config(
            'projection', participant=participant, collection=collection,
            **kwargs
        )
