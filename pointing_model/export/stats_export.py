import pointing_model.utils as utils
import pointing_model
import os
import shutil
import pandas as pd
import numpy as np


def export(point_model, path="./export/"):
    utils.ensure_dir_exists(path)

    # export_pca_components(point_model, path)
    export_participants(point_model, path)
    export_calibrations(point_model, path)
    export_calibrations(point_model, path, True)
    export_SelectKBest_chi2(point_model, path)


def export_pca_components(point_model, path):
    path += 'content/pointing_movement/import/'
    point_model.reset()
    fs = utils.all_body_fields() +\
        utils.all_body_orientation_fields() +\
        utils.all_features()
    fs = point_model.compute_pca(base_fields=fs)

    components = pd.DataFrame(point_model.pca.components_, columns=fs).round(2)
    components = components.abs().sum().sort_values(ascending=False)
    path += 'table_pca_components.tex'
    table = components.to_latex(escape=False)
    latex = pack_table(
        table,
        'Overview of PCA Coefficients',
        'tab:data_collection:participants'
    )
    with open(path, 'w') as f:
        f.write(latex)
        print('exported PCA components latex table.')


def export_SelectKBest_chi2(point_model, path):
    path += 'content/pointing_movement/import/'
    p, X, c, y = point_model.normalized
    fs = utils.all_body_fields() +\
        utils.all_body_orientation_fields() +\
        utils.all_features()
    k = 10
    kbest = point_model.find_best_features(
        p, X, c, y, fields=fs, load_features=True
    ).head(k)
    table = kbest.to_latex(escape=False, index=False)
    latex = pack_table(
        table,
        '$\\chi^2$ scores of the top %d features' % k,
        'tab:pointing_movement:chi2'
    )
    path += 'table_chi2.tex'
    with open(path, 'w') as f:
        f.write(latex)
        print('exported SelectKBest chi2 latex table.')


def export_participants(point_model, path):
    path += 'content/data_collection_study/import/'
    utils.ensure_dir_exists(path)
    participants = point_model.participants
    stats = participants.describe(percentiles=[])\
        .rename(columns=utils.participant_fields_to_readable())\
        .drop(columns=['age'])\
        .T\
        .drop(columns=['count', '50%'])\
        .rename(columns={
            'mean': '\\thead{mean}',
            'std': '\\thead{std}',
            'min': '\\thead{min}',
            'max': '\\thead{max}'
        })\
        .round(2)
    path += 'table_participants.tex'
    with open(path, 'w') as f:
        table = stats.to_latex(escape=False)
        latex = pack_table(
            table,
            'Overview of the participants [cm]',
            'tab:data_collection:participants'
        )
        f.write(latex)
        print('exported participant latex table.')


def export_calibrations(point_model, path, big=False):
    section = 'appendix'
    take = utils.all_body_fields()
    note = ''
    drop = ['count', '50%', 'min', 'max']
    if not big:
        section = 'pointing_movement'
        take = utils.flatten([
            utils.body_field('rightShoulder'),
            utils.body_field('leftShoulder'),
            utils.body_field('hmd')
        ])
        note = '. See \\cref{} for all datapoints.'
        # drop = ['count', '50%']
    path += 'content/%s/import/' % section
    utils.ensure_dir_exists(path)
    participants = point_model.participants
    calibrations = point_model.calibrations

    def get_stats(take, only_total=False):
        aggs = ['mean', 'std']
        total = calibrations[take]\
            .rename(columns=utils.body_fields_to_readable())\
            .agg(aggs)
        idx = pd.MultiIndex.from_product([take, aggs])
        total = pd.DataFrame(np.ravel(total.values, order='F'), idx, [''])\
            .T\
            .rename(columns=utils.body_fields_to_readable())\
            .rename(mapper=lambda s: '\\textbf{%s}' % s, axis=1)

        if not only_total:
            stats = calibrations[['pid'] + take]\
                .rename(columns=utils.body_fields_to_readable())\
                .groupby(['pid'])\
                .describe(percentiles=[])\
                .drop(drop, axis=1, level=1)\
                .rename(mapper=lambda s: '\\textbf{%s}' % s, axis=1)
            stats = stats.append(total, sort=False)
        else:
            stats = total.T
        return stats * 100

    def write_stats(path, stats):
        with open(path, 'w') as f:
            file_name = os.path.basename(f.name)
            column_format = 'l' * (len(stats.columns)+1)
            table = stats.to_latex(
                escape=False, column_format=column_format,
                float_format="{:0.2f}".format
            )
            latex = pack_table(
                table,
                'Overview of the calibration [cm]%s' % note,
                'tab:%s:%s' % (section, file_name.replace('.tex', '')),
            )
            f.write(latex)
            print('exported %s calibration latex table.' % section)

    if not big:
        stats = get_stats(take)
        spath = path + 'table_calibration.tex'
        write_stats(spath, stats)

        stats = get_stats(take, True)
        spath = path + 'table_calibration_total.tex'
        write_stats(spath, stats)
    else:
        # num_chunks = 3
        # chunk_len = round(len(take)/num_chunks)
        chunk_len = 6
        for i, cs in enumerate(utils.chunks(take, chunk_len)):
            stats = get_stats(cs)
            log_path = path + 'table_calibration_%s.tex' % i
            write_stats(log_path, stats)


def pack_table(table, caption='', label='tab:tab', landscape=False):
    magic_comment = '% !TeX root = ../../../thesis.tex'
    table = """
\\begin{{table}}
    \\caption{{{caption}}}
    \\label{{{label}}}
        {table}
\\end{{table}}
        """\
        .format(table=table, caption=caption, label=label)\
        .replace('\\\\', '\\\\ \\hdashline')\
        .replace(
            'rightShoulderMarkerDist.X',
            (
                '\\makecell[l]{Horizontal distance from right shoulder '
                + '\\\\markerset to right shoulder \\smallskip}'
            )
        )\
        .replace(
            'rightShoulderMarkerDist.Y',
            (
                '\\makecell[l]{Vertical distance from right shoulder '
                + '\\\\markerset to right shoulder \\smallskip}'
            )
        )
    if landscape:
        table = """
\\afterpage{{%
    \\clearpage
    \\thispagestyle{{empty}}
    \\begin{{landscape}}
        \\centering
        {table}
    \\end{{landscape}}
    \\clearpage
}}"""\
        .format(table=table)

    return """{magic_comment}
{table}""".format(magic_comment=magic_comment, table=table)
