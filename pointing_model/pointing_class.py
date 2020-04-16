import pointing_model.utils as utils
import pointing_model.loading as loading
import pointing_model.features as features
import pointing_model.learning as learning
import pointing_model.plotting as plotting
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import pandas as pd
import time


class PointingModel(
    learning.Learning, plotting.Plotting, features.DynamicFeatures
):
    default_config = {
        'load': {},
        'normalize': {},
        'ml': {},
        'dynamic': {'load': False}
    }

    def __init__(self, config={}):
        self.config = dict(utils.merge(self.default_config, config))
        super().__init__()
        self.config = dict(utils.merge(self.default_config, config))
        self.read = False
        self.__read()
        self.__normalize()

    def __read(self):
        if self.read:
            return
        if self.config.get('load_fast', False):
            self.participants, self.X, self.y, self.calibrations =\
                loading.log_fast(**self.config['load'])
        elif self.config.get('load_default_fast', False):
            self.participants, self.X, self.y, self.calibrations =\
                loading.log_default_fast(**self.config['load'])
        else:
            self.participants, self.X, self.y, self.calibrations = loading.log(
                **self.config['load']
            )
        self.read = True

    def __normalize(self):
        participants, X, calibrations, y = self.normalize_data(
            self.participants, self.X, self.calibrations, self.y,
            **self.config['normalize']
        )
        self.normalized_participants = participants
        self.normalized_X = X
        self.normalized_calibrations = calibrations
        self.normalized_y = y

    def normalize_data(self, participants, X, calibrations, y, **kwargs):
        normalize_by_xz_start = kwargs.get('normalize_by_xz_start', True)
        normalize_by_height = kwargs.get('normalize_by_height', True)
        normalize_by_time = kwargs.get('normalize_by_time', False)
        take_only_final = kwargs.get('take_only_final', True)
        force = kwargs.get('force', False)

        if take_only_final:
            X = self.take_only_final(X)
        if normalize_by_xz_start:
            X = self.normalize_X_by_xz_start(X, force=force)
        if normalize_by_height:
            participants, X, calibrations = self.normalize_X_by_height(
                participants, X, calibrations, force=force
            )
        if normalize_by_time:
            X = self.normalize_X_by_time(X)
        return participants, X, calibrations, y

    @property
    def use_dynamic_features(self):
        dconfig = self.config.get('dynamic', {})
        return dconfig.get('load', False)

    @property
    def norm_p(self):
        return copy.deepcopy(self.normalized_participants)

    @property
    def norm_X(self):
        return copy.deepcopy(self.normalized_X)

    @property
    def norm_c(self):
        return copy.deepcopy(self.normalized_calibrations)

    @property
    def norm_y(self):
        return copy.deepcopy(self.normalized_y)

    @property
    def normalized(self):
        return self.norm_p, self.norm_X, self.norm_c, self.norm_y

    @property
    def raw_data(self):
        return self.participants, self.X, self.calibrations, self.y

    @property
    def normalized_time(self):
        X = copy.deepcopy(self.X_normalized_by_time)
        if X is None:
            X = copy.deepcopy(self.normalize_X_by_time(self.norm_X))
        return self.norm_p, X, self.norm_c, self.norm_y

    def take_only_final(self, X, override=False):
        if self.X_only_final is not None and not override:
            return self.X_only_final
        self.X_only_final = copy.deepcopy(X).groupby(['pid', 'cid']).tail(1)
        return self.X_only_final

    # should only be used for plotting
    def normalize_X_by_time(self, X, force=False):
        if self.X_normalized_by_time is not None and not force:
            return self.X_normalized_by_time

        X_normalized_by_time = copy.deepcopy(X)
        time = X_normalized_by_time['time']
        time_min = X_normalized_by_time\
            .groupby(['pid', 'cid'])['time']\
            .transform('min')
        time_max = X_normalized_by_time\
            .groupby(['pid', 'cid'])['time']\
            .transform('max')
        X_normalized_by_time['time'] =\
            (time - time_min) / (time_max - time_min)

        if not force:
            self.X_normalized_by_time = copy.deepcopy(X_normalized_by_time)

        return X_normalized_by_time

    def normalize_X_by_height(
        self, participants, X, calibrations, force=False
    ):
        if self.X_normalized_by_height is not None\
                and self.p_normalized_by_height is not None\
                and self.c_normalized_by_height is not None\
                and not force:
            return\
                self.p_normalized_by_height,\
                self.X_normalized_by_height,\
                self.c_normalized_by_height

        participants = copy.deepcopy(participants)
        X = copy.deepcopy(X)
        calibrations = copy.deepcopy(calibrations)

        def normalize_height(xs, fields=[]):
            fields = fields if len(fields) > 0 else utils.all_body_fields()
            pid = xs.name
            height = self.participants.loc[pid]['height'] #/ 100
            xs[fields] = xs[fields] / height
            return xs

        X = X.groupby(['pid'])\
            .apply(lambda xs: normalize_height(xs))

        calibrations = calibrations.groupby(['pid'])\
            .apply(lambda xs: normalize_height(xs))

        participants = participants.groupby(['id'])\
            .apply(lambda xs: normalize_height(
                xs, utils.participant_fields(False)
            ))

        if not force:
            self.p_normalized_by_height = copy.deepcopy(participants)
            self.X_normalized_by_height = copy.deepcopy(X)
            self.c_normalized_by_height = copy.deepcopy(calibrations)

        return participants, X, calibrations

    def normalize_X_by_xz_start(self, X, force=False):
        if self.X_normalized_xz is not None and not force:
            return self.X_normalized_xz
        X = copy.deepcopy(X)

        def normalize_xz(xs):
            pid, cid = xs.name
            _x = self.X[(self.X['pid'] == pid) & (self.X['cid'] == cid)]
            hmd_x = _x['hmd.X'].iloc[0]
            xs[utils.x_fields()] = xs[utils.x_fields()].sub(hmd_x)
            hmd_z = _x['hmd.Z'].iloc[0]
            xs[utils.z_fields()] = xs[utils.z_fields()].sub(hmd_z)
            # print(hmd_x, hmd_z, xs[['hmd.X', 'hmd.Z']].head()), exit()
            return xs

        groupby = ['pid', 'cid']
        fields = groupby + utils.x_fields() + utils.z_fields()
        X[fields] = X[fields]\
            .reset_index()\
            .groupby(groupby)\
            .apply(lambda xs: normalize_xz(xs))\
            .set_index('index')

        if not force:
            self.X_normalized_xz = copy.deepcopy(X)

        return X
