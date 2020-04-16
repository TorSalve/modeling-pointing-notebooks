from pointing_model import features, utils
import pandas as pd
import numpy as np


class DynamicFeatures(features.Features):

    def __init__(self):
        dconfig = self.config.get('dynamic', {})
        super().__init__()
        if dconfig.get('load', False):
            self.__dfeatures()

    def __dfeatures(self):
        w_start, w_mid, w_end = (0, .3), (.5, .75), (-0.3, 0)
        w_rel_start, w_rel_mid, w_rel_end = (0, .2), (0.4, 0.6), (0.8, 1)
        functions = {
            'velocity':
            lambda p, X, c, y: features.velocity(
                X, field='indexfinger', key='velocity'
            ),
            'acceleration':
            lambda p, X, c, y: features.acceleration(
                X, field='indexfinger', key='acceleration'
            ),

            'velocity_indexfinger_window_start':
            lambda p, X, c, y: features.velocity_abs_window_mean(
                p, X, c, y, window=w_start,
                key='velocity_indexfinger_window_start'
            ),
            'velocity_indexfinger_window_mid':
            lambda p, X, c, y: features.velocity_abs_window_mean(
                p, X, c, y, window=w_mid,
                key='velocity_indexfinger_window_mid'
            ),
            'velocity_indexfinger_window_end':
            lambda p, X, c, y: features.velocity_abs_window_mean(
                p, X, c, y, window=w_end,
                key='velocity_indexfinger_window_end'
            ),
            'velocity_indexfinger_rel_window_start':
            lambda p, X, c, y: features.velocity_rel_window_mean(
                p, X, c, y, window=w_rel_start,
                key='velocity_indexfinger_rel_window_start'
            ),
            'velocity_indexfinger_rel_window_mid':
            lambda p, X, c, y: features.velocity_rel_window_mean(
                p, X, c, y, window=w_rel_mid,
                key='velocity_indexfinger_rel_window_mid'
            ),
            'velocity_indexfinger_rel_window_end':
            lambda p, X, c, y: features.velocity_rel_window_mean(
                p, X, c, y, window=w_rel_end,
                key='velocity_indexfinger_rel_window_end'
            ),

            'acceleration_indexfinger_window_start':
            lambda p, X, c, y: features.acceleration_abs_window_mean(
                p, X, c, y, window=w_start,
                key='acceleration_indexfinger_window_start'
            ),
            'acceleration_indexfinger_window_mid':
            lambda p, X, c, y: features.acceleration_abs_window_mean(
                p, X, c, y, window=w_mid,
                key='acceleration_indexfinger_window_mid'
            ),
            'acceleration_indexfinger_window_end':
            lambda p, X, c, y: features.acceleration_abs_window_mean(
                p, X, c, y, window=w_end,
                key='acceleration_indexfinger_window_end'
            ),
            'acceleration_indexfinger_rel_window_start':
            lambda p, X, c, y: features.acceleration_rel_window_mean(
                p, X, c, y, window=w_rel_start,
                key='acceleration_indexfinger_rel_window_start'
            ),
            'acceleration_indexfinger_rel_window_mid':
            lambda p, X, c, y: features.acceleration_rel_window_mean(
                p, X, c, y, window=w_rel_mid,
                key='acceleration_indexfinger_rel_window_mid'
            ),
            'acceleration_indexfinger_rel_window_end':
            lambda p, X, c, y: features.acceleration_rel_window_mean(
                p, X, c, y, window=w_rel_end,
                key='acceleration_indexfinger_rel_window_end'
            ),

            'indexfinger_horizontal_polynomial':
            lambda p, X, c, y: features.fit_polynomial(
                X, 'indexfinger.X', 'indexfinger_horizontal_polynomial', 3
            ),
            'indexfinger_vertical_polynomial':
            lambda p, X, c, y: features.fit_polynomial(
                X, 'indexfinger.Y', 'indexfinger_vertical_polynomial', 3
            ),
            'indexfinger_depth_polynomial':
            lambda p, X, c, y: features.fit_polynomial(
                X, 'indexfinger.Z', 'indexfinger_depth_polynomial', 3
            )
        }

        for key in functions:
            self.add_feature_function(key, functions[key])
