import pointing_model.utils as utils
import math
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pandas as pd
import copy
from scipy import ndimage
from math import atan2, degrees
from sklearn.preprocessing import MinMaxScaler
import random
from collections import OrderedDict


# style_label = 'seaborn-paper'
# sns.set_context("paper")
style_label = 'seaborn'
# style_label = 'seaborn-whitegrid'
# style_label = 'ggplot'
offset = .05
markers = utils.marker_cycle()
colors = utils.color_cycle()
dpi = 100
g_figsize = (12, 8)


def get_fig_ax(**kwargs):
    projection = kwargs.get('projection', None)
    figsize = kwargs.get('figsize', g_figsize)

    if projection != '3d':
        with plt.style.context(style_label):
            fig = plt.figure(dpi=dpi, figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
    else:
        fig = plt.figure(dpi=dpi, figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)

    plt.grid(True)
    return fig, ax


def plot_projection(X, y, **kwargs):
    highlight_endpoint = kwargs.get('highlight_endpoint', True)
    show_marker_labels = kwargs.get('show_marker_labels', False)
    show_true_target = kwargs.get('show_true_target', True)
    set_plot_limit = kwargs.get('set_plot_limit', True)
    view_elev = kwargs.get('view_elev', 10)
    view_azim = kwargs.get('view_azim', 10)
    color_by_target = kwargs.get('color_by_target', False)
    xfs, yfs, zfs = kwargs.get(
        'proj_fields',
        (utils.x_fields(), utils.y_fields(), utils.z_fields())
    )
    xlim = kwargs.get('xlim', (-1.1, 1.1))
    ylim = kwargs.get('ylim', (-0.75, 3.6))
    zlim = kwargs.get('zlim', (0, 2.6))
    mark = kwargs.get('mark', None)

    # https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
    # with plt.xkcd():
    kwargs['projection'] = '3d'
    fig, ax = get_fig_ax(**kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    if set_plot_limit:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    if view_elev is not None:
        ax.elev = view_elev
    if view_azim is not None:
        ax.azim = view_azim
    fields = utils.body_fields()

    def o(xs):
        return xs + offset

    _target_colors = [
        '#333333', '#ff0000', '#ff4000', '#ff8000', '#ffab00', '#ffbf00',
        '#ffff00', '#bfff00', '#80ff00', '#40ff00', '#00ff00',
        '#00ff80', '#00ffbf', '#00ffff', '#00bfff', '#0080ff',
        '#0040ff', '#0000ff', '#4000ff', '#8000ff', '#bf00ff',
        '#ff00ff', '#ff00bf', '#ff0080', '#ff0040', '#ff0000',
        '#000000',
    ]
    # random.shuffle(_target_colors)
    target_colors = {
        str(tuple(map(float, t.values()))): _target_colors[i]
        for i, t
        in enumerate(utils.get_targets())
    }
    last = X.iloc[-1]
    for _, r in X.iterrows():
        marker = next(markers)
        line_defaults = {
            'marker': marker, 'markersize': 5, 'alpha': 0.7
        }
        Xs, Ys, Zs = r[xfs], r[zfs], r[yfs]
        if(highlight_endpoint and r['time'] == last['time']):
            ax.plot(
                Xs, Ys, Zs, linewidth=3, color='r', **line_defaults
            )
            if show_marker_labels:
                for i in range(len(Xs)):
                    ax.text(Xs[i], Ys[i], Zs[i], fields[i], fontsize=10)
            continue
        if color_by_target:
            i = r['cid']
            target = str(tuple(r[utils.target_fields()].to_numpy()))
            c = target_colors.get(target)
            ax.plot(
                Xs, Ys, Zs, linewidth=1, c=c, label=target,
                alpha=.4, marker='o', markersize=3
            )
        else:
            ax.plot(
                Xs, Ys, Zs, linewidth=1, **line_defaults
            )

    if show_true_target:
        plot_truetarget(y, ax)

    if isinstance(mark, tuple) and len(mark) == 3:
        m, n, j = mark
        ax.scatter(m, n, j, c='r', s=50, edgecolors='k')
        ax.text(m-.03, n, j, "HMD", fontsize=12)

    if isinstance(y, tuple):
        plt.title('Participants pointing at target (%s, %s, %s)' % y)
    elif color_by_target:
        plt.title('Participants pointing color coded by targets')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    else:
        participant, collection = X.iloc[0].pid, X.iloc[0].cid
        plt.title(
            'Participant %d, Collection %d' % (participant, collection)
        )

    plt_show_save(fig, **kwargs)


def plot_truetarget(y, ax):
    if isinstance(y, tuple):
        _x, _z, _y = y
    else:
        _x, _z, _y = y[utils.target_fields()].values
    ax.scatter(_x, _y, _z, marker='x', c='r', s=20)
    ax.text(
        _x+offset, _y+offset, _z+offset,
        '%.2f, %.2f, %.2f' % (_x, _z, _y), fontsize=10
    )


def plot_timeline(X, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    axis = kwargs.get('axis', 'x')
    # with plt.xkcd():
    with plt.style.context(style_label):
        if(axis == 'x'):
            fields = utils.x_fields()
        elif(axis == 'y'):
            fields = utils.y_fields()
        else:
            fields = utils.z_fields()
        fields = ['time'] + fields
        ax = X[fields].plot(
            x='time', kind='line', style='.-',
            title='timeline', figsize=figsize
        )
        ax.set_ylabel('%s-value' % axis)
        plt_title_label_lim_legend(**kwargs)
        plt_show_save(None, **kwargs)


def attach_feat(feature, key, ax, **kwargs):
    line_type = kwargs.get('line_type', 'line')
    label_feature = kwargs.get('label_feature', False)
    scatter_size = kwargs.get('scatter_size', 50)

    targets = feature[utils.target_fields()]
    pid, cid = feature.iloc[0].pid, feature.iloc[0].cid
    x, y, z = feature.iloc[0][utils.target_fields()]
    label = ""
    if label_feature:
        label = "p:%d, c:%d, t:(%s,%s,%s)" % (pid, cid, x, y, z)
    marker = next(markers)
    if feature.shape[0] == 1 or line_type == 'scatter':
        ax.scatter(
            feature['time'], feature[key],
            label=label, marker=marker,
            s=scatter_size
        )
    else:
        ax.plot(
            feature['time'], feature[key],
            label=label, marker=marker, markersize=4
        )


def attach_text(feature, key, ax):
    offset = .5
    x, y, z = feature[utils.target_fields()]
    ax.text(
        feature['time'], feature[key],
        '%s, %s, %s' % (x, y, z),
        fontsize=10
    )


def attach_mean(feature, key, ax, label_mean):
    label = ""
    if label_mean:
        x, y, z = feature.iloc[0][utils.target_fields()]
        pid = feature.iloc[0].pid
        label = "$\\overline{x}$,p:%d,t:(%s,%s,%s)" % (pid, x, y, z)
    mean = feature[key].mean()
    ax.axhline(mean, ls=':', label=label, c=next(colors))
    return mean


def attach_feature_mean(feature, key, ax):
    feature = feature.reset_index('pid').reset_index('time')
    marker = next(markers)
    ax.plot(
        feature.time, feature[key],
        label="p:%s" % (feature.pid.iloc[0]),
        marker=marker, markersize=4
    )


def boxplot_feature(features, key, **kwargs):
    kwargs['xlabel'] = kwargs.get('xlabel', 'target')
    groupby = kwargs.get('groupby', utils.target_fields())
    rename = kwargs.get('rename', True)

    if rename:
        features = rename_fields(features, sort=False)
        key = utils.body_fields_short(key)
        groupby = list(map(utils.body_fields_short, groupby))

    fig, ax = get_fig_ax(**kwargs)
    features.boxplot(
        by=groupby, column=key,
        ax=ax, rot=90)
    plt.suptitle("")
    plt_title_label_lim_legend(**kwargs)
    plt_show_save(fig, **kwargs)


def plot_feature(features, key, **kwargs):
    label_mean = kwargs.get('label_mean', False)
    plot_feature = kwargs.get('plot_feature', True)
    plot_target = kwargs.get('plot_target', False)
    plot_mean = kwargs.get('plot_mean', False)
    plot_time_mean = kwargs.get('plot_time_mean', False)

    features = rename_fields(features)
    key = utils.body_fields_short(key)

    fig, ax = get_fig_ax(**kwargs)

    with plt.style.context(style_label):
        group = features[
                ['pid', 'cid', key, 'time']
                + utils.target_fields()
            ]\
            .groupby(['pid', 'cid'])
        if(plot_feature):
            group.apply(lambda f: attach_feat(f, key, ax, **kwargs))
        if(plot_target):
            group.tail(1).apply(lambda f: attach_text(f, key, ax), axis=1)
        if(plot_mean):
            features.groupby(['pid'] + utils.target_fields())\
                .apply(lambda f: attach_mean(f, key, ax, label_mean))
        if(plot_time_mean):
            decimals = 2
            features['time'] = features['time']\
                .apply(lambda x: round(x, decimals))
            group = features[['pid', key, 'time']]\
                .groupby(['time', 'pid'])\
                .mean()\
                .groupby(['pid'])\
                .apply(lambda f: attach_feature_mean(f, key, ax))
        plt_title_label_lim_legend(**kwargs)
        plt_show_save(fig, **kwargs)


def plt_title_label_lim_legend(**kwargs):
    xlabel = kwargs.get('xlabel', 'time')
    ylabel = kwargs.get('ylabel', 'value')
    xlim = kwargs.get('xlim', 'auto')
    ylim = kwargs.get('ylim', 'auto')
    title = kwargs.get('title', 'Value change over time')
    legend = kwargs.get('legend', False)
    legend_kws = kwargs.get('legend_kws', {'loc': 'lower center', 'ncol': 4})
    xticks_rot = kwargs.get('xticks_rot', None)
    yticks_rot = kwargs.get('yticks_rot', None)
    default_fontsize = {'title': 18, 'label': 15, 'tick': 12}
    fontsize = kwargs.get('fontsize', default_fontsize)
    fontsize = dict(utils.merge(fontsize, default_fontsize))
    remove_legend_duplicates = kwargs.get('remove_legend_duplicates', True)

    plt.title(title, fontsize=fontsize['title'])
    plt.xlabel(xlabel, fontsize=fontsize['label'])
    plt.ylabel(ylabel, fontsize=fontsize['label'])
    if ylim != 'auto':
        if isinstance(ylim, dict):
            plt.ylim(**ylim)
        else:
            plt.ylim(ylim)
    if xlim != 'auto':
        if isinstance(xlim, dict):
            plt.xlim(**xlim)
        else:
            plt.xlim(xlim)
    plt.xticks(fontsize=fontsize['tick'])
    if xticks_rot is not None:
        plt.xticks(rotation=xticks_rot, fontsize=fontsize['tick'])
    plt.yticks(fontsize=fontsize['tick'])
    if yticks_rot is not None:
        plt.yticks(rotation=yticks_rot, fontsize=fontsize['tick'])
    if legend:
        if remove_legend_duplicates:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            lgd = plt.legend(
                by_label.values(), by_label.keys(),
                frameon=True, **legend_kws
            )
        else:
            lgd = plt.legend(frameon=True, **legend_kws)

        frame = lgd.get_frame()
        frame.set_facecolor('#EAEAF2')
        frame.set_edgecolor('#9c9cb8')


def plt_show_save(fig=None, **kwargs):
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', 'figures/figure.png')
    tight_layout = kwargs.get('tight_layout', True)
    grid = kwargs.get('grid', True)

    plt.grid(grid)
    if tight_layout:
        plt.tight_layout()
    if save:
        filename = save_path
        utils.ensure_dir_exists(filename, is_file=True)
        plt.savefig(filename, dpi="figure")
    else:
        plt.show()
        pass
    if fig is not None:
        plt.close(fig)
    else:
        plt.clf()


def sns_show_save(fig, **kwargs):
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', 'figures/figure.png')

    if save:
        filename = save_path
        utils.ensure_dir_exists(filename, is_file=True)
        fig.savefig(filename)
    else:
        plt.show()
    plt.clf()


def plot_pca(pca, **kwargs):
    _type = kwargs.get('type', 'cumsum')
    fig, ax = get_fig_ax(**kwargs)
    if(_type == 'cumsum'):
        plot_pca_cumsum(pca, ax, **kwargs)
    plt_show_save(fig, **kwargs)


def plot_pca_cumsum(pca, ax, **kwargs):
    n_components = kwargs.get('n_components', 'mle')
    plot_nth_component = kwargs.get('plot_nth_component', .99)

    with plt.style.context(style_label):
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = n_components if utils.is_number(n_components) else 1
        plt.plot(cumsum, label="cumulative explained variance")
        title = "%d components describe %d%% of the data"\
            % (len(cumsum), n_components * 100)
        if(plot_nth_component > 0):
            components = np.where(cumsum >= plot_nth_component)[0]
            ci = components[0]
            plt.scatter(
                ci, cumsum[ci], c='r', zorder=10,
                label="%d components describe %.2f%% of the data"
                % (ci, cumsum[ci] * 100)
            )
            kwargs['legend'] = True
        plt_title_label_lim_legend(title=title, **kwargs)


def plot_correlation_matrix(correlation_matrix, **kwargs):
    fig, ax = get_fig_ax(**kwargs)
    correlation_matrix = rename_fields(correlation_matrix, axis='both')
    with plt.style.context(style_label):
        sns.heatmap(
            correlation_matrix,
            vmax=1.0, vmin=-1.0,
            square=True, ax=ax, cmap='coolwarm',
            xticklabels=True, yticklabels=True
        )
        # kwargs['xticks_rot'] = 45
        # kwargs['yticks_rot'] = -45
        kwargs['fontsize'] = {'label': 'small', 'tick': 'small'}
        plt_title_label_lim_legend(**kwargs)
        plt_show_save(fig, **kwargs)


def plot_pairplot(X, **kwargs):
    diag_kind = kwargs.get('diag_kind', 'kde')
    kind = kwargs.get('kind', 'scatter')
    hue = kwargs.get('hue', 'target')
    plot_kws = {}
    if kind == 'scatter':
        plot_kws = {'alpha': 0.6, 's': 50, 'edgecolor': 'k'}
    pairplot_type = kwargs.get('pairplot_type', 'auto')

    if pairplot_type == 'kdeplot':
        fig = sns.PairGrid(X)
        fig.hue_vals = X[hue]
        fig.hue_names = X[hue].unique()
        fig.palette = sns.color_palette("Paired", len(fig.hue_names))
        fig = fig.map_upper(plt.scatter, **plot_kws)
        fig = fig.map_diag(sns.kdeplot)
        fig = fig.map_lower(sns.kdeplot)
    else:
        fig = sns.pairplot(
            X,
            hue=hue, diag_kind=diag_kind,
            plot_kws=plot_kws, kind=kind
        )
    sns_show_save(fig, **kwargs)


def plot_selectKBest_chi2(X, **kwargs):
    regression = kwargs.get('regression', False)
    fig, ax = get_fig_ax(**kwargs)
    X = rename_fields(
        X.set_index(['feature']), axis=0,
        targets=[], sort=False
    )
    width = .5
    ticks = range(len(X['p-value']))
    rs = ax.scatter(
        ticks, X['p-value'].values,
        label='p-value'
    )
    if not regression:
        ax.axhline(y=.05, label="$p = 0.05$", c='r')

    ax.set_xticks(ticks)
    ax.set_xticklabels(X.index.values)
    plt_title_label_lim_legend(**kwargs)
    plt_show_save(**kwargs)


def plot_selectKBest_mutual_information(X, **kwargs):
    fig, ax = get_fig_ax(**kwargs)

    X = rename_fields(X, axis='both', sort=False)
    idx = X.sum(axis=1).sort_values().index
    X.loc[idx].plot.bar(stacked=True, ax=ax)

    plt_title_label_lim_legend(**kwargs)
    plt_show_save(**kwargs)


def plot_extratrees(X, **kwargs):
    fig, ax = get_fig_ax(**kwargs)

    rename_fields(X, axis=0, targets=[])\
        .plot(kind='bar', ax=ax, width=0.75)

    plt_title_label_lim_legend(**kwargs)
    plt_show_save(**kwargs)


def plot_pca_coefficients(components, **kwargs):
    with plt.style.context(style_label):
        fig, ax = get_fig_ax(**kwargs)

        features = components.columns.values
        components = rename_fields(components)\
            .abs()\
            .rename(mapper=lambda x: 'PC-%s' % x, axis=0)

        components.plot.bar(ax=ax, stacked=True)

        plt_title_label_lim_legend(**kwargs)
        plt_show_save(fig, **kwargs)


def rename_fields(df, axis=1, targets=None, sort=False):
    if targets is None:
        targets = utils.target_fields() if 'TrueTarget.X' in df.columns else []
    if axis == 'both':
        axis = [0, 1]
    else:
        axis = [axis]
    for ax in axis:
        if isinstance(df, pd.Series):
            df = df.rename(utils.body_fields_short, axis=ax)
        else:
            df = df.rename(mapper=utils.body_fields_short, axis=ax)
        c = df.columns if ax else df.index
        columns = [i for i in c if i not in targets]
        columns = sorted(columns) if sort else columns
        if isinstance(df, pd.DataFrame):
            df.columns = columns + targets if ax == 1 else df.columns
        df.index = columns + targets if ax == 0 else df.index
    return df


def autolabel(rs, ax):
    for r in rs:
        height = int(r.get_height())
        ax.annotate(
            '{}'.format(height), textcoords="offset points",
            xy=(r.get_x() + r.get_width() / 2, height),
            ha='center', va='bottom',
            xytext=(0, 3),  # 3 points vertical offset
        )


def plot_count_hist(X, key, **kwargs):
    with plt.style.context(style_label):
        figsize = kwargs.get('figsize', g_figsize)
        fig, ax = get_fig_ax(**kwargs)

        def vl_cnt(xs, key):
            vl_cnts = xs[key].value_counts()
            # print(vl_cnts, xs.name, sep='\n')
            return vl_cnts

        X['target'] = X[utils.target_fields()]\
            .apply(lambda xs: str(tuple(xs)), axis=1)
        X[[key, 'target']]\
            .groupby(['target'])\
            .apply(lambda xs: vl_cnt(xs, key))\
            .unstack(level=-1)\
            .fillna(0)\
            .plot.bar(ax=ax, width=.8, stacked=True)

        plt_title_label_lim_legend(**kwargs)
        plt_show_save(fig, **kwargs)


def plot_learning_confusion_matrix(matrix, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    sns.heatmap(
        matrix, vmin=0,
        annot=True, ax=ax, cmap='coolwarm'
    )

    plt_title_label_lim_legend(**kwargs)
    plt_show_save(fig, **kwargs)


def plot_target_grid(**kwargs):
    fscores = kwargs.get('fscores', {})
    title = kwargs.get('title', 'Target grid')
    figsize = kwargs.get('figsize', g_figsize)
    cmap = kwargs.get('cmap', 'YlGn')
    label_target = kwargs.get('label_target', True)
    vmax = kwargs.get('vmax', 1)
    vmin = kwargs.get('vmin', None)
    cast_target_to_float = kwargs.get('cast_target_to_float', False)
    cbar_label = kwargs.get('cbar_label', '')
    kwargs['projection'] = '3d'
    target_fields = kwargs.get('target_fields', utils.target_fields())
    fig, ax = get_fig_ax(**kwargs)

    # x = y = np.arange(-.2, .2, .1)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros(X.shape)
    # ax.plot_surface(X, Y, Z, alpha=1, color='r')
    ax.scatter(
        0, 0,
        alpha=0.75, depthshade=False,
        edgecolors="k", c='r', s=100,
        marker='x'
    )

    if len(fscores) > 0:
        x_index = utils.list_get_or_default(target_fields, 'trueTarget.X', None)
        y_index = utils.list_get_or_default(target_fields, 'trueTarget.Y', None)
        z_index = utils.list_get_or_default(target_fields, 'trueTarget.Z', None)
        cs = []
        targets = []
        for k, v in fscores.items():
            k = utils.parse_tuple(k)
            _x = k[x_index] if x_index is not None else 0
            _y = k[y_index] if y_index is not None else 1.49
            _z = k[z_index] if z_index is not None else 2.5
            _is = [
                c for c, v
                in zip([_x, _y, _z], [x_index, y_index, z_index])
                if v is not None
            ]
            if cast_target_to_float:
                i = str(tuple(map(float, _is)))
            else:
                i = str(tuple(_is))
                # i = str((_x, _y, _z))
            cs.append(float(fscores.get(i, 0)))
            targets.append((_x, _y, _z))
        x, y, z = zip(*targets)
        ss = ax.scatter(
            x, z, y, s=150, c=cs, depthshade=False,
            vmax=vmax, cmap=cmap, edgecolors="k",
            vmin=vmin
        )
        cbar = fig.colorbar(ss)
        cbar.ax.set_ylabel(cbar_label, rotation=270)
    else:
        targets = [
            (x, y, z)
            for x in utils.get_horizontal_targets()
            for y in utils.get_vertical_targets()
            for z in utils.get_depth_targets()
        ]
        x, y, z = zip(*targets)
        ax.scatter(x, z, y, s=150, c='g', edgecolors="k")

    if label_target:
        for i in range(len(x)):
            offset_x, offset_z, offset_y = -.65, 0, .05
            label = '(%s, %s, %s)' % (x[i], y[i], z[i])
            ax.text(
                x[i]+offset_x, z[i]+offset_z, y[i]+offset_y, label, fontsize=12
            )

    ax.set_xlim(-1.8, 1.8), ax.set_ylim(-.1, 3.6), ax.set_zlim(0, 2.99)
    ax.set_xlabel('x'), ax.set_ylabel('z'), ax.set_zlabel('y')
    ax.elev = 15
    ax.azim = -80
    plt.title(title)
    plt_show_save(fig, **kwargs)


def plot_polar(X, key, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    x = X[key].values
    y = np.ones(X.shape)

    plt.polar(x, y, 'ro')

    plt_show_save(fig, **kwargs)


def plot_violinplot(X, feature_key, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    X['target'] = X[utils.target_fields()]\
        .apply(lambda xs: str(tuple(xs)), axis=1)

    sns.violinplot(x=X['target'], y=X[feature_key], ax=ax)

    kwargs['xticks_rot'] = 90
    plt_title_label_lim_legend(**kwargs)
    sns_show_save(fig, **kwargs)


def plot_means(X, feature_key, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    X['target'] = X[utils.target_fields()]\
        .apply(lambda xs: str(tuple(xs)), axis=1)

    g = X.groupby('target')[feature_key].median()

    plt.plot(g.index, g.values, marker='.', label='median')

    kwargs['xticks_rot'] = 90
    plt_title_label_lim_legend(**kwargs)
    plt_show_save(fig, **kwargs)


def kdeplot_feature(X, key, **kwargs):
    label_targets = kwargs.get('label_targets', False)
    figsize = kwargs.get('figsize', g_figsize)
    # fig, ax = get_fig_ax(**kwargs)
    ylim = kwargs.get('ylim', None)
    xlim = kwargs.get('xlim', None)

    fig, axs = plt.subplots(
        3, 9, figsize=figsize,
        sharex='col', sharey='row',
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )

    def plot_kde(xs):
        tx, ty, tz = xs.name
        itx = utils.get_horizontal_targets().index(tx)
        ity = utils.get_vertical_targets().index(ty)
        itz = utils.get_depth_targets().index(tz)
        _ax = axs[itx, ity*3 + itz]
        g = sns.kdeplot(xs[key], ax=_ax, shade=True)
        g.legend_.remove()

        if ylim is not None and xlim is not None:
            _ax.text(
                xlim[0]-(xlim[0]*.05),
                ylim[1]-(ylim[1]*.2),
                str(xs.name)
            )
            _ax.set_ylim(ylim)
            _ax.set_xlim(xlim)

    X[[key] + utils.target_fields()]\
        .groupby(utils.target_fields())\
        .apply(lambda xs: plot_kde(xs))

    if label_targets:
        labelLines(plt.gca().get_lines(), fontsize=10, zorder=2.5)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    plt.title(kwargs.get('title', 'kde plot of %s' % key))
    plt.ylabel(kwargs.get('ylabel', 'Density'))
    plt.xlabel(kwargs.get('xlabel', utils.orientation_to_readable(key)))
    plt_show_save(fig, **kwargs)


# https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1]\
        + (ydata[ip]-ydata[ip-1])\
        * (x-xdata[ip-1])\
        / (xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        pct = 0.3
        _xmin, _xmax = xmin + (xmax - xmin) * pct, xmax - (xmax - xmin) * pct
        xvals = np.linspace(_xmin, _xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)


def plot_parallel_coordinates(X, keys, target, **kwargs):
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    X = rename_fields(X, axis=1)
    keys = list(map(utils.body_fields_short, keys))
    target = utils.body_fields_short(target)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(X[keys])
    features = pd.DataFrame(
        features, index=X[keys].index,
        columns=X[keys].columns
    )
    features[target] = X[target]
    features['target'] = X[target]

    pd.plotting.parallel_coordinates(features, target, ax=ax)

    kwargs['xticks_rot'] = 90
    plt_title_label_lim_legend(**kwargs)
    plt_show_save(**kwargs)


def plot_grouped_line(
    X, feature, **kwargs
):
    groupby_time = kwargs.get('groupby_time', False)
    plot_mean_line = kwargs.get('plot_mean_line', True)
    plot_all_lines = kwargs.get('plot_all_lines', False)
    y = kwargs.get('y', 'time')
    figsize = kwargs.get('figsize', g_figsize)
    fig, ax = get_fig_ax(**kwargs)

    for name, g in X.groupby(utils.target_fields()):
        if plot_all_lines:
            if groupby_time:
                g_ = g.round({'time': 2}).groupby(['time'])
            else:
                g_ = g.groupby(['pid', 'cid'])
            g_.plot.line(
                y=feature, x=y, ax=ax, label=name,
                color="C%d" % get_target_color(name, return_index=True),
                alpha=.1
            )
        if plot_mean_line:
            g.round({'time': 2}).groupby(['time'])\
                .mean().plot.line(
                    y=feature, ax=ax, label='%s' % str(name),
                    color="C%d" % get_target_color(name, return_index=True),
                    alpha=.6, #cmap='rainbow'
                )

    kwargs['remove_legend_duplicates'] = True
    kwargs['legend'] = True
    kwargs['legend_kws'] = {'ncol': 3}
    plt_title_label_lim_legend(**kwargs)
    plt_show_save(fig, **kwargs)


def clean_target(target):
    if isinstance(target, tuple):
        target = list(target)
    if isinstance(target, list) or isinstance(target, np.ndarray):
        target = str(tuple(map(float, target)))
    return target


def get_target_color(target, shuffle=False, return_index=False):
    target = clean_target(target)

    _target_colors = [
        '#333333', '#ff0000', '#ff4000', '#ff8000', '#ffab00', '#ffbf00',
        '#ffff00', '#bfff00', '#80ff00', '#40ff00', '#00ff00',
        '#00ff80', '#00ffbf', '#00ffff', '#00bfff', '#0080ff',
        '#0040ff', '#0000ff', '#4000ff', '#8000ff', '#bf00ff',
        '#ff00ff', '#ff00bf', '#ff0080', '#ff0040', '#ff0000',
        '#000000',
    ]
    if shuffle:
        random.shuffle(_target_colors)

    targets = utils.get_targets()
    clean = [clean_target(list(ts.values())) for ts in targets]
    i = clean.index(target)

    return _target_colors[i] if not return_index else i


def analyze_distances(ds, **kwargs):
    model_name = kwargs.get('model_name', '')
    save = kwargs.get('save', True)
    scores = kwargs.get('scores', {})
    def compute_distance(xs):
        xs[('dist', 'real_mean')] = utils.distance(
            np.array([list(xs.name)]),
            np.array([list(xs['pred'])])
        )[0]
        return xs

    g = ds.groupby([('true', 'x'), ('true', 'y'), ('true', 'z')])\
        .mean()\
        .apply(lambda xs: compute_distance(xs), axis=1)

    print(
        '', 'Description of distances:',
        g[('dist', 'real_mean')].describe(), '',
        sep='\n'
    )

    ks = g.reset_index()[[('true', 'x'), ('true', 'y'), ('true', 'z')]]\
        .apply(lambda xs: str(tuple(xs)), axis=1)
    vs = g[('dist', 'real_mean')]
    d = dict(zip(ks.values, vs.values))
    # plotting.boxplot_feature(
    #     ds, 'dist', rename=False,
    #     groupby=[('true', 'x'), ('true', 'y'), ('true', 'z')],
    # )
    plot_target_grid(
        title=(
            'Targets colored according to their distance' +
            ' to the predicted target (using %s).'
            % model_name
        ),
        fscores=d, vmin=None, vmax=None,
        cmap='rainbow_r', label_target=False,
        cast_target_to_float=True,
        save_path="./plot_distance_targets_%s.png" % model_name,
        save=save
    )
    scores['mean_distance'] = g[('dist', 'real_mean')].mean()
    return g