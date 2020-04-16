import pointing_model.utils as utils


class PlotConfig():

    plotConfig = {}
    learningConfig = {}

    def __init__(self, config):
        self.learningConfig = config
        self.init_boxplots()
        self.init_distance_boxplots()
        self.init_count_hist_plot()
        self.init_plots()
        self.init_relative_time()
        self.init_others()
        self.init_distances()
        self.init_orientations()
        self.init_kde_plots()

    def init_kde_plots(self):
        defaults = {
            "plt_type": 'feature_kdeplot',
        }
        config = {
            'kde_completion_time': {
                "feature_key": 'completion_time',
                "title": "Completion time",
                'xlabel': 'Time', "ylim": (0, 1.5), 
                "xlim": (0, 15),
                **defaults,
            }
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_count_hist_plot(self):
        defaults = {
            "plt_type": 'feature_count_hist_plot',
            "plot_mean": False,
            "label_feature": False,
            "label_mean": False,
            "legend": True,
            "legend_kws": {},
            "ylabel": 'bins [count]',
            "xlabel": 'target',
            "data": "normalized"
        }
        config = {
            'count_hist_plot_above_head': {
                "feature_key": 'above_head',
                "title": "Index finger, above head",
                **defaults
            },
            'count_hist_plot_above_hand': {
                "feature_key": 'above_hand',
                "title": "Index finger, above hand",
                **defaults
            },
            'count_hist_plot_indexfinger_body_position_x': {
                "feature_key": 'indexfinger_body_position_x',
                "title": "Index finger horizontal position relative to body",
                **defaults
            },
            'count_hist_plot_indexfinger_body_position_y': {
                "feature_key": 'indexfinger_body_position_y',
                "title": "Index finger vertical position relative to body",
                **defaults
            },
            'count_hist_plot_indexfinger_body_position_z': {
                "feature_key": 'indexfinger_body_position_z',
                "title": "Index finger depth relative to body",
                **defaults
            },
            'count_hist_plot_indexfinger_body_position_z2': {
                "feature_key": 'indexfinger_body_position_z2',
                "title": "Index finger depth relative to body",
                **defaults
            },
            'count_hist_plot_handO_y_relative': {
                "feature_key": 'handO_y_relative',
                "title": "",
                **defaults
            },
            'count_hist_plot_indexfingerO_y_relative': {
                "feature_key": 'indexfingerO_y_relative',
                "title": "",
                **defaults
            }
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_distance_boxplots(self):
        defaults = {
            "plt_type": 'feature_boxplot',
            "plot_mean": False,
            "label_feature": False,
            "label_mean": False,
            "legend": False,
            "data": "normalized"
        }
        config = {}
        for key in utils.get_all_distance_keys():
            config['boxplot_%s' % key] = {
                'feature_key': key,
                "ylim": (0, 1),
                "ylabel": "distance [% of height]",
                "title": utils.distance_key_to_readable(key),
                **defaults,
            }
        self.plotConfig = {**self.plotConfig, **config}

    def init_boxplots(self):
        defaults = {
            "plt_type": 'feature_boxplot',
            "plot_mean": False,
            "label_feature": False,
            "label_mean": False,
            "legend": False,
            "data": "normalized"
        }
        config = {
            'boxplot_indexfinger_vertical': {
                "feature_key": 'indexfinger_vertical',
                # "ylim": {'bottom': 0},
                "ylabel": "height over the floor [% of height]",
                "title": "Distance from floor to index finger",
                **defaults,
            },
            'boxplot_indexfinger_horizontal': {
                "title": "Horizontal position of the index finger",
                # "ylim": (-0.55, 0.55),
                "feature_key": 'indexfinger_horizontal',
                "ylabel": "position [% of height]",
                **defaults,
            },
            'boxplot_indexfinger_depth': {
                "title": "Depth of the index finger position",
                # "ylim": (0, 0.5),
                "feature_key": 'indexfinger_depth',
                "ylabel": "position [% of height]",
                **defaults,
            },
            'boxplot_hmd_vertical': {
                "feature_key": 'hmd_vertical',
                # "ylim": {'bottom': 0},
                "ylabel": "height over the floor [% of height]",
                "title": "Distance from floor to HMD",
                **defaults,
            },
            'boxplot_hmd_horizontal': {
                "title": "Horizontal position of the HMD",
                # "ylim": (-0.55, 0.55),
                "feature_key": 'hmd_horizontal',
                "ylabel": "position [% of height]",
                **defaults,
            },
            'boxplot_hmd_depth': {
                "title": "Depth of the HMD position",
                # "ylim": (0, 0.5),
                "feature_key": 'hmd_depth',
                "ylabel": "position [% of height]",
                **defaults,
            },
            'boxplot_elbow_flexion': {
                "feature_key": 'elbow_flexion',
                "ylim": (0, 180),
                "ylabel": "elbow flexion [degrees]",
                "title": "Flexion of the elbow",
                **defaults,
            },
            'boxplot_shoulder_abduction': {
                "feature_key": 'shoulder_abduction',
                "ylim": (0, 180),
                "ylabel": "shoulder abduction [degrees]",
                "title": "Abduction of the shoulder",
                **defaults,
            },
            'boxplot_shoulder_horizontal': {
                "feature_key": 'shoulder_horizontal',
                "ylim": (0, 180),
                "ylabel": "shoulder horizontal angle [degrees]",
                "title": "Horizontal angle of the shoulder",
                **defaults,
            },
            'boxplot_shoulder_xz_slope': {
                "feature_key": 'shoulder_xz_slope',
                "ylabel": "slope", "ylim": (-10, 10),
                "title": (
                    "Slope of the line between left and right shoulder markers"
                ),
                **defaults,
            },
            'boxplot_completion_time': {
                "feature_key": 'completion_time',
                "ylabel": "time",
                "title": "Completion time",
                **defaults,
            }
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_plots(self):
        config = {
            'projection': {
                'plt_type': 'projection',
            },
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_relative_time(self):
        defaults = {
            "data": 'normalized_time',
            'plt_type': 'feature_boxplot',
            'groupby': ['time'],
            "label_feature": True,
            "legend": True,
            'ylim': (0, 2.1),
            'plot_feature': False,
            'plot_time_mean': True
        }
        config = {
            'indexfinger_velocity_time_normalized': {
                "feature_key": 'indexfinger_velocity',
                "ylabel": "indexfinger velocity [m/s]",
                "xlabel": "(normalized) time",
                "title": "Mean velocity of the index finger pr. participant",
                **defaults,
            },
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_others(self):
        default = {
            "data": "normalized"
        }
        config = {
            'correlation_matrix': {
                "ylabel": "",
                "xlabel": "",
                "title": "Correlation matrix of body values",
                "figsize": (20, 10),
                **default
            },
            'pairplot': {
                "ylabel": "",
                "xlabel": "",
                "title": "",
                **default
            },
            'pca': {
                'ylim': (0, 1.05),
                'xlim': {'left': -0.95},
                'legend_kws': {'loc': 'lower center'},
                "ylabel": "cumulative sum of principal components",
                "xlabel": "principal component",
                **default
            },
            'count_hist': {
                "ylabel": 'bins [count]',
                "xlabel": 'target',
                **default
            },
            'selectKBest': {
                "ylabel": '$p$-value',
                "xlabel": 'feature',
                # 'ylim': (0, 0.2),
                'xlim': {'left': -0.5},
                'title': '$\\chi^2$ $p$-values for all features',
                'legend': True,
                'legend_kws': {},
                'xticks_rot': 90,
                **default
            },
            'selectKBest_mi': {
                "ylabel": 'summed mutual information',
                "xlabel": 'feature',
                # 'ylim': (0, 0.2),
                # 'xlim': {'left': -0.5},
                'title': (
                    'Mutual information for all features, ' +
                    'split by target position'
                ),
                'legend': True,
                'legend_kws': {},
                **default
            },
            'pca_coefficients': {
                'legend': True,
                'legend_kws': {},
                'xlim': {'left': -0.5},
                'ylim': {'bottom': 0},
                'title': 'Summed absolute PC coefficients for all features',
                'ylabel': "absolute sum of coefficients",
                "xlabel": 'feature',
                **default
            },
            'extratrees': {
                "xlabel": 'feature',
                "ylabel": 'score',
                # 'ylim': (0, 0.2),
                # 'xlim': {'left': -0.5},
                'title': 'Feature importance',
                'legend': False,
                **default
            },
        }
        self.plotConfig = {**self.plotConfig, **config}

    def init_distances(self):
        defaults = {
            "plt_type": 'feature_boxplot',
            "plot_mean": False,
            "label_feature": False,
            "label_mean": False,
            "legend": False,
        }
        config = {}
        for key in utils.get_all_distance_keys():
            a, b = map(utils.body_field_to_readable, key.split('_')[:2])
            config["boxplot_" + key] = {
                'title': 'Distance between %s and %s' % (a, b),
                'ylabel': 'distance [% of height]',
                'feature_key': key,
                **defaults
            }

        self.plotConfig = {**self.plotConfig, **config}

    def init_orientations(self):
        defaults = {
            "plot_mean": False,
            "label_feature": False,
            "label_mean": False,
            "legend": False,
        }
        specifics = {
            "hmdO_X": {"ylim": (0, 11.5), "xlim": (-.65, .65)},
            "hmdO_Y": {"ylim": (0, 10.5), "xlim": (-.65, .65)},
            "hmdO_Z": {"ylim": (0, 15), "xlim": (-.25, .25)},
            "upperarmO_X": {"ylim": (0, 7.2), "xlim": (-.75, .75)},
            "upperarmO_Y": {"ylim": (0, 6.4), "xlim": (-.75, .75)},
            "upperarmO_Z": {"ylim": (0, 3.6), "xlim": (-1.2, 1.2)},
            "handO_X": {"ylim": (0, 1.5), "xlim": (-1.2, 1.2)},
            "handO_Y": {"ylim": (0, 1.3), "xlim": (-2.1, 2.1)},
            "handO_Z": {"ylim": (0, 2), "xlim": (-1.2, 1.2)},
            "indexfingerO_X": {"ylim": (0, 3.1), "xlim": (-1.2, 1.2)},
            "indexfingerO_Y": {"ylim": (0, 1.1), "xlim": (-2.1, 2.1)},
            "indexfingerO_Z": {"ylim": (0, 4.1), "xlim": (-1.2, 1.2)},
            "forearmO_X": {"ylim": (0, 8.5), "xlim": (-.65, .65)},
            "forearmO_Y": {"ylim": (0, 8.5), "xlim": (-1.2, 1.2)},
            "forearmO_Z": {"ylim": (0, 3.6), "xlim": (-1.2, 1.2)},
            "leftShoulderO_X": {"ylim": (0, 2.5), "xlim": (-1.2, 1.2)},
            "leftShoulderO_Y": {"ylim": (0, 1.2), "xlim": (-1.2, 1.2)},
            "leftShoulderO_Z": {"ylim": (0, 3.6), "xlim": (-1.2, 1.2)},
            "rightShoulderO_X": {"ylim": (0, 3.2), "xlim": (-.65, .65)},
            "rightShoulderO_Y": {"ylim": (0, 3.6), "xlim": (-1.2, 1.2)},
            "rightShoulderO_Z": {"ylim": (0, 5.1), "xlim": (-.65, .65)},
        }
        config = {}
        for key in utils.get_all_orientation_keys():
            a, p = key.split('_')[:2]
            field = utils.body_field_to_readable(a.replace('O', ''))
            axis = utils.orientation_axis_to_readable(p.lower())
            config["kde_" + key] = {
                'title': 'Orientation of the %s (%s)' % (field, axis),
                'ylabel': 'Density', 'feature_key': key,
                "plt_type": 'feature_kdeplot',
                **defaults, **specifics.get(key, {})
            }

            config["boxplot_" + key] = {
                'title': 'Orientation of the %s (%s)' % (field, axis),
                'ylabel': 'orientation', 'feature_key': key,
                'xlabel': "%s (%s) orientation" % (field, axis),
                "plt_type": 'feature_boxplot',
                **defaults,
            }

        self.plotConfig = {**self.plotConfig, **config}

    def getConfig(self, plot_key):
        return self.plotConfig.get(plot_key, {})


def learning_plots():
    return {
        'confusion_matrix': {
            'title': 'Confusion matrix of an ML algorithm',
            'ylabel': "True target",
            "xlabel": 'Predicted target',
        }
    }


def dynamic_feature_config():
    return {

        'velocity_indexfinger_window_start': {
            'title': 'Index finger velocity absolute start window',
            'ylabel': 'velocity'
        },
        'velocity_indexfinger_window_mid': {
            'title': 'Index finger velocity absolute middle window',
            'ylabel': 'velocity'
        },
        'velocity_indexfinger_window_end': {
            'title': 'Index finger velocity absolute end window',
            'ylabel': 'velocity'
        },

        'velocity_indexfinger_rel_window_start': {
            'title': 'Index finger velocity relative start window',
            'ylabel': 'velocity'
        },
        'velocity_indexfinger_rel_window_mid': {
            'title': 'Index finger velocity relative middle window',
            'ylabel': 'velocity'
        },
        'velocity_indexfinger_rel_window_end': {
            'title': 'Index finger velocity relative end window',
            'ylabel': 'velocity'
        },

        'acceleration_indexfinger_window_start': {
            'title': 'Index finger acceleration absolute start window',
            'ylabel': 'acceleration'
        },
        'acceleration_indexfinger_window_mid': {
            'title': 'Index finger acceleration absolute middle window',
            'ylabel': 'acceleration'
        },
        'acceleration_indexfinger_window_end': {
            'title': 'Index finger acceleration absolute end window',
            'ylabel': 'acceleration'
        },

        'acceleration_indexfinger_rel_window_start': {
            'title': 'Index finger acceleration relative start window',
            'ylabel': 'acceleration'
        },
        'acceleration_indexfinger_rel_window_mid': {
            'title': 'Index finger acceleration relative middle window',
            'ylabel': 'acceleration'
        },
        'acceleration_indexfinger_rel_window_end': {
            'title': 'Index finger acceleration relative end window',
            'ylabel': 'acceleration'
        },

    }
