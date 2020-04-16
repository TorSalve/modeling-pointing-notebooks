import pandas as pd
import numpy as np
from pointing_model import features, utils
import copy


def velocity(X, field='indexfinger', key='velocity'):
    return full_data_function(X, __velocity, field, key)


def acceleration(X, field='indexfinger', key='acceleration'):
    return full_data_function(X, __acceleration, field, key)


def full_data_function(X, function, field, key):
    values = function(X, field, key)
    return values\
        .reset_index()\
        .rename(
            columns={'index': 'sid'}
        )[['pid', 'cid', 'sid', key]]


def fit_polynomial(X, field, key, deg=3):
    def fit_poly(xs):
        x = xs['time'].values
        y = xs[field].values
        coefficients = np.polyfit(x, y, deg)
        for i, c in enumerate(np.flip(coefficients)):
            xs["%s-%s" % (key, i)] = c
        return xs

    poly = X.reset_index()[['pid', 'cid', 'time', field]]\
        .groupby(['pid', 'cid'])\
        .apply(fit_poly)\
        .groupby(['pid', 'cid'])\
        .tail(1)\
        .drop(columns=['time', field])
    return ddataframe(X, poly)


def window_function(X, function, window, field, key, rel):
    values = function(X, field, key)
    if rel:
        w = get_relative_window(values, *window)
    else:
        w = get_absolute_window(values, *window)
    mean = w[['pid', 'cid', key]]\
        .groupby(['pid', 'cid'])\
        .mean()\
        .reset_index()
    return ddataframe(X, mean)


def velocity_abs_window_mean(
    p, X, c, y, window=(0, 1),
    field='indexfinger', key='velocity_indexfinger_window_mean'
):
    return window_function(X, __velocity, window, field, key, False)


def velocity_rel_window_mean(
    p, X, c, y, window=(0, 1),
    field='indexfinger', key='velocity_indexfinger_rel_window_mean'
):
    return window_function(X, __velocity, window, field, key, True)


def acceleration_abs_window_mean(
    p, X, c, y, window=(0, 1),
    field='indexfinger', key='acceleration_indexfinger_window_mean'
):
    return window_function(X, __acceleration, window, field, key, False)


def acceleration_rel_window_mean(
    p, X, c, y, window=(0, 1),
    field='indexfinger', key='acceleration_indexfinger_rel_window_mean'
):
    return window_function(X, __acceleration, window, field, key, True)


def __velocity(X, field='indexfinger', key='velocity'):
    def delta(X, field, d=True):
        s = X[field].values
        t = np.roll(s, 1, axis=0)
        t[0] = s[0]
        return utils.distance(t, s) if d else s - t

    def velocity(xs, key):
        delta_x = delta(xs, utils.body_field(field))
        delta_t = delta(xs, 'time', False)
        xs[key] = np.nan_to_num(np.divide(delta_x, delta_t))
        return xs

    return X\
        .reset_index()\
        .groupby(['pid', 'cid'])\
        .apply(velocity, key=key)\
        .set_index('index')


def __acceleration(X, field='indexfinger', key='acceleration'):
    def delta(X, field, d=True):
        s = X[field].values
        t = np.roll(s, 1, axis=0)
        t[0] = s[0]
        return utils.distance(t, s) if d else s - t

    def acceleration(xs, key, vel_key):
        delta_v = delta(xs, vel_key, False)
        delta_t = delta(xs, 'time', False)
        xs[key] = np.nan_to_num(np.divide(delta_v, delta_t))
        return xs

    velocity_key = 'velocity'
    return __velocity(X, field, velocity_key)\
        .reset_index()\
        .groupby(['pid', 'cid'])\
        .apply(acceleration, key=key, vel_key=velocity_key)\
        .set_index('index')


def get_absolute_window(X, i, j):
    def window(xs):
        _i, _j = copy.deepcopy(i), copy.deepcopy(j)
        if i < 0:
            _i += xs['time'].max()
            if j == 0:
                _j = xs['time'].max()
        if j < 0:
            _j += xs['time'].max()
        if i >= j:
            raise IndexError("the first index must be smaller than the second")
        return xs[(xs['time'] >= _i) & (xs['time'] <= _j)]
    return X.groupby(['pid', 'cid']).apply(window).reset_index(drop=True)


def get_relative_window(X, i, j):
    _X = utils.normalize_time(X)
    return X[(_X['time'] > i) & (_X['time'] <= j)]


def ddataframe(X, feature):
    return X\
        .groupby(['pid', 'cid'])\
        .tail(1)[['pid', 'cid']]\
        .reset_index()\
        .rename(columns={'index': 'sid'})\
        .merge(feature, on=['pid', 'cid'])
