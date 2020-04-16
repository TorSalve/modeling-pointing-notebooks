import pointing_model.utils as utils
import os
import progressbar
import pandas as pd
from datetime import date


def log_path(**kwargs):
    default = "../PointingThere/Assets/Resources/Log"
    path = kwargs.get('loading_path', default)
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        raise Exception("Log path does not exist or is empty: %s" % path)
    return path


def log_fast(**kwargs):
    default = "./export/data/ATHCC"
    path = kwargs.get('loading_path', default)
    dataset = kwargs.get('dataset', 'full')
    paths = [
        '%s/%s/participants.csv' % (path, dataset),
        '%s/%s/collections.csv' % (path, dataset),
        '%s/%s/targets.csv' % (path, dataset),
        '%s/%s/calibrations.csv' % (path, dataset),
    ]
    data = tuple(map(pd.read_csv, paths))
    return utils.rename_new_to_old(*data)


def log_default_fast(**kwargs):
    path = kwargs.get('loading_path', './data_export')
    paths = [
        '%s/participants.csv' % path,
        '%s/collections.csv' % path,
        '%s/targets.csv' % path,
        '%s/calibrations.csv' % path,
    ]
    p, X, y, c = tuple(map(pd.read_csv, paths))
    p = p.set_index('id')
    y = y.set_index('id')
    X = X.set_index('index')
    return p, X, y, c

def log(**kwargs):
    op = kwargs.get('only_participants', False)
    participants = read_participants(**kwargs)
    X, y, calibration = read_all_collections_snaphots_and_calibrations(
        participants.index, **kwargs
    )
    X["cid"] = pd.to_numeric(X["cid"])
    y["pid"] = pd.to_numeric(y["pid"])
    return participants, X, y, calibration


def read_participants(**kwargs):
    olp = kwargs.get('only_load_participants', [])
    path = "%s/participants.csv" % (log_path(**kwargs))
    utils.check_file(path)
    participants = pd.read_csv(path).drop_duplicates(keep='last')
    participants = utils.filter_df(participants, 'id', olp)
    participants = participants.set_index('id')\
        .rename(columns={
            'rightShoulderMarkerDistX': 'rightShoulderMarkerDist.X',
            'rightShoulderMarkerDistY': 'rightShoulderMarkerDist.Y'
        })
    return participants


def read_all_collections_snaphots_and_calibrations(participantIds, **kwargs):
    collections = pd.DataFrame(columns=utils.all_target_fields())
    snapshots = pd.DataFrame(columns=utils.all_fields())
    calibrations = pd.DataFrame(columns=utils.calibration_fields())
    with progressbar.ProgressBar(max_value=len(participantIds)) as bar:
        i = 0
        for participantId in participantIds:
            collection = read_collections(participantId, **kwargs)
            collections = collections.append(collection, sort=False)
            calibration = read_calibrations(participantId, **kwargs)
            calibrations = calibrations.append(calibration, sort=False)
            snapshot = read_all_snapshots(
                participantId, collection.index, **kwargs
            )
            snapshots = snapshots.append(snapshot, sort=False)
            i += 1
            bar.update(i)
    snapshots['pid'] = pd.to_numeric(snapshots['pid'])
    calibrations['pid'] = pd.to_numeric(calibrations['pid'])
    collections = collections\
        .drop(columns=['id'])\
        .reset_index()\
        .rename(columns={'index': 'id'})\
        .set_index('id')
    return snapshots, collections, calibrations


def read_collections(participantId, **kwargs):
    path = "%s/%d/collections.csv" % (log_path(**kwargs), participantId)
    utils.check_file(path)
    olc = kwargs.get('only_load_collections', [])
    olt = kwargs.get('only_load_targets', [])
    dfc = kwargs.get('drop_first_collections', -1)
    exclude = kwargs.get('exclude_collections', {})
    collections = pd.read_csv(path)
    collections = utils.filter_df(collections, 'id', olc)
    collections = utils.add_column(collections, 'pid', int(participantId))
    if dfc > 0:
        collections = collections.iloc[dfc:]
    y = collections
    if len(olt) > 0:
        y = pd.DataFrame(columns=utils.all_target_fields())
        for t in olt:
            df = collections[
                (collections['trueTarget.X'] == t['x']) &
                (collections['trueTarget.Y'] == t['y']) &
                (collections['trueTarget.Z'] == t['z'])
            ]
            y = y.append(df, sort=False)
    for pid in exclude:
        cids = exclude[pid]
        y = y[(y.pid != pid) & (~y.id.isin(cids))]
    y.set_index('id', inplace=True)
    return y


def read_calibrations(participantId, **kwargs):
    lpath = log_path(**kwargs)
    path = "%s/%d/calibration.csv" % (lpath, participantId)
    utils.check_file(path)
    snapshots = pd.read_csv(path)
    snapshots = utils.add_column(snapshots, 'pid', int(participantId))
    snapshots['time'] = snapshots['time'].values - snapshots['time'].values[0]
    return snapshots.sort_values(by='time')


def read_all_snapshots(participantId, collectionIds, **kwargs):
    snapshots = pd.DataFrame(columns=utils.all_fields())
    exclude = kwargs.get('exclude_collections', {})
    exclude = exclude.get(str(participantId), [])
    # print("loading participant %d" % participantId)
    for collectionId in collectionIds:
        if collectionId not in exclude:
            snapshot = read_snapshots(
                participantId, collectionId, **kwargs
            )
            snapshots = snapshots.append(snapshot, sort=False)
    return snapshots


def read_snapshots(participantId, collectionId, **kwargs):
    lpath = log_path(**kwargs)
    path = "%s/%d/collection_%d.csv" % (lpath, participantId, collectionId)
    utils.check_file(path)
    snapshots = pd.read_csv(path)
    snapshot_id = "%d_%d" % (participantId, collectionId)
    snapshots = utils.add_column(snapshots, 'id', snapshot_id)
    snapshots = utils.add_column(snapshots, 'pid', int(participantId))
    snapshots = utils.add_column(snapshots, 'cid', int(collectionId))
    snapshots['time'] = snapshots['time'].values - snapshots['time'].values[0]
    otfp = kwargs.get('only_take_final_position', False)
    if otfp:
        snapshots = snapshots.iloc[-1:]
    return snapshots.sort_values(by='time')
