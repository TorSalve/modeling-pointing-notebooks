import numpy as np
import pointing_model.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import copy


def compute(participants, X, calibrations, y, function, key, **kwargs):
    values = pd.DataFrame()
    for _, p in participants.iterrows():
        calibration = calibrations[calibrations.pid == p.name]
        Xs = X[X.pid == p.name]
        values = values.append(
            function(p, Xs, calibration, y, key, **kwargs),
            sort=False
        )
    return values


def compute_elbow_flexion(
    participant, Xs, calibration, y, key='elbow_flexion'
):
    Bs = Xs[utils.body_field('forearm')].values
    Cs = Xs[utils.body_field('upperarm')].values
    b = participant.upperarmMarkerDist
    c = participant.forearmMarkerDist
    return compute_angle(Xs, Bs, Cs, b, c, key)


# def compute_shoulder_abduction(
#     participant, Xs, calibration, y, key='shoulder_abduction'
# ):
#     Bs = Xs[['upperarm.X', 'upperarm.Y']].values
#     Cs = Xs[['rightShoulder.X', 'rightShoulder.Y']].values
#     b = participant.upperarmLength - participant.upperarmMarkerDist
#     c = participant['rightShoulderMarkerDist.Y']
#     return compute_angle(Xs, Bs, Cs, b, c, key)


def compute_shoulder_abduction(
    participant, Xs, calibration, y, key='shoulder_abduction'
):
    Bs = Xs[['upperarm.X', 'upperarm.Y']].values
    anker = calibration.iloc[0][['upperarm.X', 'upperarm.Y']].values
    Cs = np.tile(anker, (Bs.shape[0], 1))
    origin = Xs[['rightShoulder.X', 'rightShoulder.Y']]
    # origin['rightShoulder.X'] -= participant['rightShoulderMarkerDist.X']
    # origin['rightShoulder.Y'] += participant['rightShoulderMarkerDist.Y']
    b = utils.distance(origin.values, Cs)
    c = participant.upperarmLength - participant.upperarmMarkerDist
    return compute_angle(Xs, Bs, Cs, b, c, key)


def compute_shoulder_horizontal(
    participant, Xs, calibration, y, key='shoulder_horizontal'
):
    Bs = Xs[['upperarm.X', 'upperarm.Z']].values
    anker = calibration.iloc[2][['upperarm.X', 'upperarm.Z']].values
    Cs = np.tile(anker, (Bs.shape[0], 1))
    origin = Xs[['rightShoulder.X', 'rightShoulder.Z']].values
    b = utils.distance(origin, Cs)
    c = participant.upperarmLength - participant.upperarmMarkerDist
    return compute_angle(Xs, Bs, Cs, b, c, key)


def compute_angle(Xs, Bs, Cs, b, c, feature_key):
    a = utils.distance(Bs, Cs)
    if utils.is_number(b):
        b = np.repeat(b, a.shape[0])
    if utils.is_number(c):
        c = np.repeat(c, a.shape[0])
    angles = angle(a, b, c)
    return dataframe(Xs, feature_key, angles)


def compute_distance(Xs, key, field_A, field_B):
    As = Xs[utils.body_field(field_A)].values
    Bs = Xs[utils.body_field(field_B)].values
    distances = utils.distance(As, Bs)
    return dataframe(Xs, key, distances)


def compute_field_value(Xs, key, field):
    value = Xs[field].values
    return dataframe(Xs, key, value)


def compute_velocity(Xs, field='indexfinger', key='indexfinger_velocity'):
    As = Xs[utils.body_field(field)].values
    As = np.vstack((As, [0, 0, 0]))
    Bs = np.roll(As, 1, axis=0)
    # remove first because it is 0 anyways, last because it got appended
    distances = utils.distance(As, Bs)[1:-1]
    As = Xs['time'].values
    As = np.hstack((As, [0]))
    Bs = np.roll(As, 1, axis=0)
    # remove first because it is 0 anyways, last because it got appended
    times = (As - Bs)[1:-1]
    velocity = np.divide(distances, times)
    # add a zero to begin with
    velocity = np.hstack(([0], velocity))
    return dataframe(Xs, key, velocity)


def compute_above_head(Xs, key='above_head'):
    one_hot = np.where(Xs['indexfinger.Y'].values >= 1, 1, 0)
    return dataframe(Xs, key, one_hot)


def compute_above_hand(Xs, key='above_hand'):
    one_hot = np.where(Xs['indexfinger.Y'].values >= Xs['hand.Y'].values, 1, 0)
    return dataframe(Xs, key, one_hot)


def compute_orientation_relative(
    Xs, field, thresholds=(.65, .75, 1), key='handO_y_relative'
):
    threshold_sm, threshold_md, threshold_lg = thresholds

    values = np.abs(copy.deepcopy(Xs[field].values))
    values[(values >= threshold_lg)] = 3
    values[(values >= threshold_md) & (values < threshold_lg)] = 2
    values[(values >= threshold_sm) & (values < threshold_md)] = 1
    values[(values < threshold_sm)] = 0
    # values[(values >= elbow_height) & (values < shoulder_height)] = 1

    return dataframe(Xs, key, values)


def compute_indexfinger_body_position_y(
    participant, Xs, calibration, y, key='indexfinger_body_position_y'
):
    height = participant['height'] / 100
    # print(Xs['indexfinger.Y'].head(), height, sep='\n'), exit()
    shoulder_height =\
        participant['rightShoulderMarkerDist.Y'] / 100 +\
        calibration['rightShoulder.Y'].mean()

    values = copy.deepcopy(Xs['indexfinger.Y'].values)
    _values = copy.deepcopy(values)
    _values[(values >= height)] = 2
    _values[(values >= shoulder_height) & (values < height)] = 1
    _values[(values < shoulder_height)] = 0

    return dataframe(Xs, key, _values)


def compute_indexfinger_body_position_x(
    participant, Xs, calibration, y, key='indexfinger_body_position_x'
):
    shoulder_right = Xs['rightShoulder.X']
    shoulder_left = Xs['leftShoulder.X']

    values = copy.deepcopy(Xs['indexfinger.X'].values)
    values = np.where(
        shoulder_right < values, 2,
        np.where(shoulder_left > values, 0, 1)
    )
    return dataframe(Xs, key, values)


def compute_indexfinger_body_position_z(
    participant, Xs, calibration, y,
    key='indexfinger_body_position_z', **kwargs
):
    og_particiants = kwargs.get('og_p')

    shoulder = Xs['rightShoulder.Z']
    elbow = shoulder + participant['upperarmLength']
    hand = elbow + participant['forearmLength']
    upperarm = (shoulder + elbow) / 2
    forearm = (elbow + hand) / 2

    og_participant = og_particiants.loc[participant.name]
    cm = lambda c: c / (og_participant['height'] * 100)
    handplus1 = hand + cm(1)
    handplus2 = hand + cm(2)
    handplus3 = hand + cm(3)
    handplus4 = hand + cm(4)
    handplus5 = hand + cm(5)
    handplus6 = hand + cm(6)
    handplus7 = hand + cm(7)
    handplus8 = hand + cm(8)
    handplus9 = hand + cm(9)
    handplus10 = hand + cm(10)
    handplus11 = hand + cm(11)
    handplus12 = hand + cm(12)

    values = copy.deepcopy(Xs['indexfinger.Z'].values)
    _values = copy.deepcopy(values)

    # larger than hand + 10
    _values[(values >= handplus12)] = 12
    _values[(values >= handplus11) & (values < handplus12)] = 11
    _values[(values >= handplus9) & (values < handplus11)] = 10
    _values[(values >= handplus8) & (values < handplus9)] = 9
    _values[(values >= handplus7) & (values < handplus8)] = 8
    _values[(values >= handplus6) & (values < handplus7)] = 7
    _values[(values >= handplus5) & (values < handplus6)] = 6
    _values[(values >= handplus4) & (values < handplus5)] = 5
    _values[(values >= handplus3) & (values < handplus4)] = 4
    _values[(values >= handplus2) & (values < handplus3)] = 3
    _values[(values >= handplus1) & (values < handplus2)] = 2
    _values[(values >= hand) & (values < handplus1)] = 1
    # smaller than hand
    _values[(values < hand)] = 0


    # # larger than hand + 10
    # _values[(values >= handplus2)] = 1
    # # smaller than hand
    # _values[(values < handplus2)] = 0
    return dataframe(Xs, key, _values)


def compute_indexfinger_body_position_z2(
    participant, Xs, calibration, y,
    key='indexfinger_body_position_z', **kwargs
):
    cal = calibration.iloc[2]
    upperarm = cal['upperarm.Z']
    forearm = cal['forearm.Z']
    hand = cal['hand.Z']
    indexfinger = cal['indexfinger.Z']+0.2

    values = copy.deepcopy(Xs['indexfinger.Z'].values)
    _values = copy.deepcopy(values)

    _values[(values >= indexfinger)] = 2
    _values[(values >= hand) & (values < indexfinger)] = 1
    _values[(values < hand)] = 0
    return dataframe(Xs, key, _values)


def compute_shoulder_xz_slope(
    participant, Xs, calibration, y, key='shoulder_xz_slope'
):
    x1, y1 = Xs['rightShoulder.X'], Xs['rightShoulder.Z']
    x2, y2 = Xs['leftShoulder.X'], Xs['leftShoulder.Z']
    return compute_slope_2d(Xs, x1, y1, x2, y2, key)


def compute_slope_2d(Xs, x1, y1, x2, y2, key):
    values = (y2 - y1) / (x2 - x1)
    return dataframe(Xs, key, values)


def dataframe(Xs, feature_key, feature):
    return pd.DataFrame({
        'sid': Xs.index,
        'pid': Xs.pid.values,
        'cid': Xs.cid.values,
        feature_key: feature
    }, dtype=float)


# Triangle:         A
#                   .
#                c ... b
#                 .....
#               B   a   C
# computes elbow_angle of A
def angle(a, b, c):
    cosine = (b*b + c*c - a*a) / (2*b*c)
    # cosine = np.where(cosine < -1, cosine, cosine - 2)
    # cosine = np.where(cosine > 1, cosine, cosine + 2)
    return np.degrees(np.arccos(cosine))
