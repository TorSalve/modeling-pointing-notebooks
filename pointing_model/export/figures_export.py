import pointing_model.utils as utils
import pointing_model
import os
import shutil
import pointing_model.plotting as plotting


def export(point_model, path="./export"):
    figure_base_path = '%s/content' % path
    path_end = "/figures/generated"

    def path_gen(folder):
        path = figure_base_path + folder + path_end
        utils.ensure_dir_exists(path)
        return path

    pointing_movement_path = path_gen("/pointing_movement")
    data_collection_path = path_gen("/data_collection_study")
    application_path = path_gen("/application")

    p, X, c, y = point_model.normalized
    X = point_model.attach_target(X, y)
    proj_defaults = {
        'save': True, 'view_elev': 20, 'view_azim': 50,
        'highlight_endpoint': False
    }
    X.groupby(utils.target_fields())\
        .apply(
            lambda xs: plotting.plot_projection(
                xs, xs.name,
                save_path=(
                    "%s/plot_projection_grouped_%s_%s_%s.png"
                    % ((data_collection_path, ) + xs.name)
                ),
                **proj_defaults
            )
        )
    X.groupby(utils.target_fields())\
        .apply(
            lambda xs: plotting.plot_projection(
                xs, xs.name, show_marker_labels=True,
                set_plot_limit=False, show_true_target=False,
                save_path=(
                    "%s/plot_projection_grouped_%s_%s_%s_no_target.png"
                    % ((data_collection_path, ) + xs.name)
                ),
                **proj_defaults
            )
        )

    X[utils.x_fields()] = X[utils.x_fields()].sub(X['hmd.X'], axis=0)
    X[utils.y_fields()] = X[utils.y_fields()].sub(X['hmd.Y'], axis=0)
    X[utils.z_fields()] = X[utils.z_fields()].sub(X['hmd.Z'], axis=0)

    X.groupby(utils.target_fields())\
        .apply(
            lambda xs: plotting.plot_projection(
                xs, xs.name,
                proj_fields=(
                    utils.x_fields(exclude=['leftShoulder']),
                    utils.y_fields(exclude=['leftShoulder']),
                    utils.z_fields(exclude=['leftShoulder'])
                ),
                save_path=(
                    "%s/plot_projection_grouped_hmd_normalized_%s_%s_%s_no_target.png"
                    % ((data_collection_path, ) + xs.name)
                ),
                show_marker_labels=True,
                set_plot_limit=False, show_true_target=False,
                **proj_defaults
            )
        )

    X.groupby(utils.target_fields())\
        .apply(
            lambda xs: plotting.plot_projection(
                xs, xs.name,
                proj_fields=(
                    utils.x_fields(exclude=['leftShoulder']),
                    utils.y_fields(exclude=['leftShoulder']),
                    utils.z_fields(exclude=['leftShoulder'])
                ),
                save_path=(
                    "%s/plot_projection_grouped_hmd_normalized_%s_%s_%s.png"
                    % ((data_collection_path, ) + xs.name)
                ),
                **proj_defaults
            )
        )

    projections = []
    collection = 70
    _projections = [
        (1, collection), (2, collection), (3, collection), (4, collection),
        (5, collection), (6, collection), (7, collection), (8, collection),
        (9, collection), (10, collection), (11, collection), (12, collection),
        (13, collection)
    ]
    for p, c in _projections:
        defaults = {
            'plot_key': 'projection',
            'save': True, "participant": p, "collection": c,
            'data': 'normalized_all_snapshots'
        }
        plot_target = {
            'fn': 'plot_config',
            'fn_args': {
                'save_path': (
                    "%s/plot_projection_%s_%s.png"
                    % (data_collection_path, p, c)
                ),
                **defaults
            }
        }
        plot_no_target = {
            'fn': 'plot_config',
            'fn_args': {
                'save_path': (
                    "%s/plot_projection_%s_%s_no_target.png"
                    % (data_collection_path, p, c)
                ),
                "show_marker_labels": True, "show_true_target": False,
                "set_plot_limit": False, **defaults
            }
        }
        projections.append(plot_target)
        projections.append(plot_no_target)

    count_hist = []
    _count_hist = [
        'above_head',
        'above_hand',
        'indexfinger_body_position_x',
        'indexfinger_body_position_y',
        'indexfinger_body_position_z',
    ]
    for k in _count_hist:
        plot = {
            'fn': 'plot_config',
            'fn_args': {
                'save': True, 'plot_key': "count_hist_plot_%s" % k,
                'save_path': (
                    "%s/plot_count_hist_%s.png" % (pointing_movement_path, k)
                )
            }
        }
        count_hist.append(plot)

    boxplots = []
    _boxplots = [k for k in utils.all_features() if k not in _count_hist]
    for k in _boxplots:
        plot = {
            'fn': 'plot_config',
            'fn_args': {
                'save': True, 'plot_key': "boxplot_%s" % k,
                'save_path': "%s/boxplot_%s.png" % (pointing_movement_path, k)
            }
        }
        boxplots.append(plot)

    kde = []
    _kde = list(map(
        utils.field_to_orientation_key,
        utils.flatten(
            list(map(utils.body_orientation_field, ['upperarm', 'hmd']))
        )
    ))
    for k in _kde:
        plot = {
            'fn': 'plot_config',
            'fn_args': {
                'save': True, 'plot_key': "kde_%s" % k,
                'save_path': pointing_movement_path + "/kde_%s.png" % k
            }
        }
        kde.append(plot)

        plot = {
            'fn': 'plot_config',
            'fn_args': {
                'save': True, 'plot_key': "boxplot_%s" % k,
                'save_path': pointing_movement_path + "/boxplot_%s.png" % k
            }
        }
        boxplots.append(plot)

    features = utils.all_features()

    configs = [
        {
            'plotting': [
                # plot PCA
                {
                    'fn': 'plot_pca',
                    'fn_args': {
                        'save': True, 'force': True, 'load_features': False,
                        'save_path': (
                            "%s/plot_pca_base.png" % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_pca',
                    'fn_args': {
                        'save': True, 'force': True,
                        'include_features': features,
                        'save_path': (
                            "%s/plot_pca_all.png" % pointing_movement_path
                        )
                    }
                },
                # {
                #     'fn': 'plot_pca',
                #     'fn_args': {
                #         'save': True, 'force': True,
                #         'base_fields': (
                #             list(point_model.feature_functions.keys())
                #             + utils.target_fields()
                #         ),
                #         'save_path': (
                #             "%s/plot_pca_features.png"
                #             % pointing_movement_path
                #         )
                #     }
                # },
                {
                    'fn': 'plot_pca',
                    'fn_args': {
                        'save': True, 'force': True,
                        'base_fields': features,
                        'save_path': (
                            "%s/plot_pca_all_features.png"
                            % pointing_movement_path
                        )
                    }
                },
                # plot correlation matrix
                {
                    'fn': 'plot_correlation_matrix',
                    'fn_args': {
                        'save': True, 'force': True,
                        'title': 'Correlation matrix for raw data and targets',
                        'additional_fields': utils.target_fields(),
                        'save_path': (
                            "%s/correlation_matrix_base.png"
                            % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_correlation_matrix',
                    'fn_args': {
                        'save': True, 'force': True,
                        'title': (
                            'Correlation matrix for '
                            + 'raw data, features and targets'
                        ),
                        'additional_fields': features + utils.target_fields(),
                        'save_path': (
                            "%s/correlation_matrix_all.png"
                            % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_correlation_matrix',
                    'fn_args': {
                        'save': True, 'force': True,
                        'title': (
                            'Correlation matrix for '
                            + 'raw orientation data and targets'
                        ),
                        'base_fields': (
                            utils.all_body_orientation_fields()
                            + utils.target_fields()
                        ),
                        'save_path': (
                            "%s/correlation_matrix_orientations.png"
                            % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_correlation_matrix',
                    'fn_args': {
                        'save': True, 'force': True,
                        'title': (
                            'Correlation matrix for '
                            + 'all computed features and the targets'
                        ),
                        'base_fields': (
                            list(point_model.feature_functions.keys())
                            + utils.target_fields()
                        ),
                        'include': list(point_model.feature_functions.keys()),
                        'save_path': (
                            "%s/correlation_matrix_all_features.png"
                            % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_correlation_matrix',
                    'fn_args': {
                        'save': True, 'force': True,
                        'title': 'Correlation matrix for features and targets',
                        'base_fields': features + utils.target_fields(),
                        'save_path': (
                            "%s/correlation_matrix_features.png"
                            % pointing_movement_path
                        )
                    }
                },
                {
                    'fn': 'plot_extratrees',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_extratrees.png"
                            % pointing_movement_path
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_extratrees',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_extratrees_all_features.png"
                            % pointing_movement_path
                        ),
                        'fields': (
                            list(point_model.feature_functions.keys())
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_selectKBest_chi2',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_selectKBest_chi2.png"
                            % pointing_movement_path
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_selectKBest_chi2',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_selectKBest_chi2_all_features.png"
                            % pointing_movement_path
                        ),
                        'fields': (
                            list(point_model.feature_functions.keys())
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_selectKBest_mutual_information',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_selectKBest_mi.png"
                            % pointing_movement_path
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_selectKBest_mutual_information',
                    'fn_args': {
                        'save': True,
                        'save_path': (
                            "%s/plot_selectKBest_mi_all_features.png"
                            % pointing_movement_path
                        ),
                        'fields': (
                            list(point_model.feature_functions.keys())
                        ),
                        'xticks_rot': 90
                    }
                },
                {
                    'fn': 'plot_pca_coefficients',
                    'fn_args': {
                        'save': True, 'base_fields': features,
                        'save_path': (
                            "%s/plot_pca_coefficients_features.png"
                            % pointing_movement_path
                        ),
                        'legend_kws': {'ncol': 6}
                    }
                }
            ] +\
            kde +\
            boxplots +\
            projections +\
            count_hist
        },
    ]

    for config in configs:
        for plot in config['plotting']:
            fn = getattr(point_model, plot['fn'])
            fn(**plot['fn_args'])

    plotting.plot_target_grid(
        save=True, save_path=application_path + "/targets.png"
    )
