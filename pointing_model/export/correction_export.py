import pandas as pd
import pointing_model.utils as utils
import os
from datetime import datetime


round_to = 7


def log_path(**kwargs):
    dt = datetime.today()
    seconds = dt.timestamp()
    default = './export/corrected_data/%.0f' % seconds
    path = kwargs.get('export_path', default)
    if os.path.exists(path):
        q = "The log path already exists. Overwrite?"
        if not utils.query_yes_no(q, 'no'):
            exit()
    else:
        os.makedirs(path)
    return path


def export(X, y, participants, calibrations, **kwargs):
    Xc = correct_X(X)
    yc = correct_y(y)
    # participantsc = correct_participants(participants)
    calibrationsc = correct_calibrations(calibrations)
    path = log_path(**kwargs)
    export_participants(participants, path)
    Xc.groupby(['pid', 'cid'])\
        .apply(lambda X: export_X(X, path))
    yc.groupby(['pid'])\
        .apply(lambda y: export_y(y, path))
    calibrationsc.groupby(['pid'])\
        .apply(lambda c: export_calibrations(c, path))


def export_X(X, path):
    pid, cid = X.name
    p = "%s/%s" % (path, pid)
    if not os.path.exists(p):
        os.makedirs(p)
    p = "%s/collection_%s.csv" % (p, cid)
    X[utils.all_body_orientation_fields()] *= -1
    X.round(round_to)\
        .drop(columns=['id', 'pid', 'cid'])\
        .to_csv(p, index=False)


def export_y(y, path):
    pid = y.name
    p = "%s/%s" % (path, pid)
    if not os.path.exists(p):
        os.makedirs(p)
    p = "%s/collections.csv" % p
    y.reset_index()\
        .round(round_to)\
        .drop(columns=['pid'])\
        .to_csv(p, index=False)


def export_calibrations(X, path):
    pid = X.name
    p = "%s/%s" % (path, pid)
    if not os.path.exists(p):
        os.makedirs(p)
    p = "%s/calibration.csv" % p
    X.round(round_to)\
        .drop(columns=['pid'])\
        .to_csv(p, index=False)


def export_participants(participants, path):
    p = "%s/participants.csv" % path
    participants.reset_index()\
        .round(round_to)\
        .to_csv(p, index=False)


def correct_X(X):
    correction_vector = {
        'x': 0.0154 / 2,
        'y': 0.0787 / 2,
        'z': 0.1032 / 2
    }
    X[utils.x_fields()] = X[utils.x_fields()] + correction_vector['x']
    X[utils.y_fields()] = X[utils.y_fields()] + correction_vector['y']
    X[utils.z_fields()] = X[utils.z_fields()] + correction_vector['z']
    return X


def correct_y(y):
    return y


def correct_participants(participants):
    cm_fields = utils.participant_fields()
    cm_fields.remove('handedness')
    cm_fields.remove('gender')
    cm_fields.remove('age')
    participants[cm_fields] = participants[cm_fields] / 100
    return participants


def correct_calibrations(calibration):
    return correct_X(calibration)
