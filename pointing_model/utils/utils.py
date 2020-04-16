import pandas as pd
import numpy as np
import itertools
import os
import warnings
import sys
import copy
import math


def body_fields():
    return [
        'indexfinger',
        'hand',
        'forearm',
        'upperarm',
        'rightShoulder',
        'hmd',
        'leftShoulder',
    ]


def participant_fields(withMeta=True):
    fields = []
    if withMeta:
        fields += [
            'handedness',
            'gender',
            'age',
        ]

    return fields + [
        'forearmLength',
        'forearmMarkerDist',
        'fingerLength',
        'upperarmLength',
        'upperarmMarkerDist',
        'height',
        'rightShoulderMarkerDist.X',
        'rightShoulderMarkerDist.Y'
    ]


def continuous_features():
    return [
        'indexfinger_vertical',
        'indexfinger_horizontal',
        'indexfinger_depth',
        'hmd_vertical',
        'hmd_horizontal',
        'hmd_depth',

        # 'completion_time', #
        # 'rightShoulder_vertical', #
        # 'rightShoulder_horizontal', #
        # 'rightShoulder_depth', #

        # 'elbow_flexion', #
        'shoulder_abduction',
        'shoulder_horizontal',

        # 'upperarmO_X', #
        # 'upperarmO_Y', #
        # 'upperarmO_Z', #
        # 'hmdO_X', #
        # 'hmdO_Y', #
        # 'hmdO_Z', #
        # 'shoulder_xz_slope', #
        
        distance_key('indexfinger', 'upperarm'),
        distance_key('indexfinger', 'rightShoulder'),
        distance_key('indexfinger', 'leftShoulder'),
        distance_key('leftShoulder', 'upperarm'),
        # distance_key('leftShoulder', 'rightShoulder'), #
        distance_key('rightShoulder', 'upperarm'),
        # distance_key('hand', 'indexfinger'), #
        distance_key('hand', 'upperarm'),
        distance_key('hand', 'leftShoulder'),
        distance_key('hand', 'rightShoulder'),
        distance_key('hand', 'hmd'),
        distance_key('forearm', 'leftShoulder'),
        distance_key('forearm', 'rightShoulder'),
        distance_key('forearm', 'hmd'),
        # distance_key('forearm', 'hand'), #
        distance_key('forearm', 'indexfinger'),
        distance_key('forearm', 'upperarm'),
        # distance_key('hmd', 'leftShoulder'), #
        distance_key('hmd', 'rightShoulder'),
        distance_key('hmd', 'indexfinger'),
        distance_key('hmd', 'upperarm'),
    ]


def categorical_features():
    return [
        # 'above_head',
        'above_hand',
        'indexfinger_body_position_x',
        'indexfinger_body_position_y',
        # 'indexfinger_body_position_z',
    ]


def all_features():
    return continuous_features() + categorical_features()


def selected_features():
    return all_features()


def spatial_features():
    return selected_features()


def temporal_features():
    return dynamic_features()


def restricted_spatial_features():
    return restricted_features()


def exclude_temporal_features(additional=[]):
    return get_exclude(
        [
            'velocity_indexfinger_window_mid',
            'velocity_indexfinger_rel_window_start',
            'velocity_indexfinger_rel_window_mid',
            'velocity_indexfinger_rel_window_end',
            'acceleration_indexfinger_window_mid',
            'acceleration_indexfinger_rel_window_mid',
            'indexfinger_horizontal_polynomial-3',
            'indexfinger_horizontal_polynomial-2',
            'indexfinger_horizontal_polynomial-1',
            'indexfinger_horizontal_polynomial-0',
            'indexfinger_vertical_polynomial-3',
            'indexfinger_vertical_polynomial-2',
            'indexfinger_vertical_polynomial-1',
            'indexfinger_vertical_polynomial-0',
            'indexfinger_depth_polynomial-3',
            'indexfinger_depth_polynomial-2',
            'indexfinger_depth_polynomial-1',
            'indexfinger_depth_polynomial-0'
        ] + additional
    )


def restricted_features():
    return [
        'indexfinger_vertical',
        'indexfinger_horizontal',
        'indexfinger_depth',
        'hmd_vertical',
        'hmd_horizontal',
        'hmd_depth',

        # 'hmdO_X',
        # 'hmdO_Y',
        # 'hmdO_Z',

        'above_head',
        'above_hand',
        'indexfinger_body_position_x',
        'indexfinger_body_position_y',
        'indexfinger_body_position_z',

        distance_key('hmd', 'indexfinger'),
        distance_key('forearm', 'hmd'),
        # distance_key('forearm', 'hand'),
        # distance_key('forearm', 'indexfinger'),
        distance_key('hand', 'hmd'),
        # distance_key('hand', 'indexfinger'),
    ]


def restricted_features():
    return [
        'indexfinger_vertical',
        'indexfinger_horizontal',
        'indexfinger_depth',
        
        'hmd_vertical',
        'hmd_horizontal',
        # 'hmd_depth',
        
        'hmdO_X',
        'hmdO_Y',
        'hmdO_Z',
        
        'above_head',
        'above_hand',
        
        'indexfinger_body_position_x',
        'indexfinger_body_position_y',
        # 'indexfinger_body_position_z',
        
        'hmd_indexfinger_distance',
        'forearm_hmd_distance',
        # 'forearm_hand_distance',
        # 'forearm_indexfinger_distance',
        'hand_hmd_distance',
        # 'hand_indexfinger_distance',
    ]


def dynamic_features():
    return [
        'velocity_indexfinger_window_start',
        'velocity_indexfinger_window_mid',
        'velocity_indexfinger_window_end',

        'velocity_indexfinger_rel_window_start',
        'velocity_indexfinger_rel_window_mid',
        'velocity_indexfinger_rel_window_end',

        'acceleration_indexfinger_window_start',
        'acceleration_indexfinger_window_mid',
        'acceleration_indexfinger_window_end',

        'acceleration_indexfinger_rel_window_start',
        'acceleration_indexfinger_rel_window_mid',
        'acceleration_indexfinger_rel_window_end',

        'indexfinger_horizontal_polynomial',
        'indexfinger_vertical_polynomial',
        'indexfinger_depth_polynomial',
    ]


def spatial_temporal_features():
    return temporal_features() + [
        'indexfinger_vertical',
        'indexfinger_horizontal',
        'indexfinger_depth',
        'hmd_vertical',
        'hmd_horizontal',
        'hmd_depth',

        'above_hand',
        'indexfinger_body_position_x',
        'indexfinger_body_position_y',

        distance_key('indexfinger', 'upperarm'),
        distance_key('indexfinger', 'rightShoulder'),
        distance_key('indexfinger', 'leftShoulder'),
        distance_key('leftShoulder', 'upperarm'),
        distance_key('rightShoulder', 'upperarm'),
        distance_key('hand', 'upperarm'),
        distance_key('hand', 'leftShoulder'),
        distance_key('hand', 'rightShoulder'),
        distance_key('hand', 'hmd'),
        distance_key('forearm', 'leftShoulder'),
        distance_key('forearm', 'rightShoulder'),
        distance_key('forearm', 'hmd'),
        distance_key('forearm', 'indexfinger'),
        distance_key('forearm', 'upperarm'),
        distance_key('hmd', 'rightShoulder'),
        distance_key('hmd', 'indexfinger'),
        distance_key('hmd', 'upperarm'),
    ]


def body_fields_short(field):
    translation = {
        'indexfinger': 'idf',
        'hand': 'hnd',
        'forearm': 'foa',
        'upperarm': 'upa',

        'rightShoulder': 'rsh',
        'leftShoulder': 'lsh',
        'hmd': 'hmd',
        'shoulder': 'sh',
        'elbow': 'elb',

        'trueTarget': 'target',
        'distance': 'dist',
        'position': 'pos',
        'body': 'bdy',

        'velocity': 'vel',
        'acceleration': 'acc',
        'window': 'win',

        'polynomial': 'poly',
        'horizontal': 'hori',
        'vertical': 'vert',
        'depth': 'depth'
    }

    for k in translation:
        if type(field) != str:
            continue
        field = field.replace(k, translation[k])
    return field


def participant_fields_to_readable():
    return {
        'forearmLength': 'Forearm length',
        'forearmMarkerDist': 'Forearm markerset distance to the elbow',
        'fingerLength': 'Index finger length',
        'upperarmLength': 'Upperarm length',
        'upperarmMarkerDist': 'Forearm markerset distance to the elbow',
        'height': 'Height'
    }


def automatic_selected_features():
    return [
        'indexfinger_horizontal_polynomial-3',
        'indexfinger_horizontal_polynomial-2',

        'shoulder_abduction',

        'indexfinger_leftShoulder_distance',
        'hand_hmd_distance',
        'forearm_hmd_distance',
        'hmd_indexfinger_distance',
        'hmd_upperarm_distance',

        'above_hand',

        'indexfinger_body_position_x',
        'indexfinger_body_position_y',

        'indexfinger.X',
        'indexfinger.Y',
        'indexfinger.Z',
        'hand.X',
        'hand.Y',
        'hand.Z',
        'forearm.X',
        'forearm.Y',
        'forearm.Z',
        'upperarm.X',
        'upperarm.Y',

        'indexfingerO.Y',
        'handO.X',
        'handO.Y',
        'handO.Z',
        'forearmO.X',
        'forearmO.Y',
        'upperarmO.X',
        'upperarmO.Y',
        'hmdO.Y'
    ]


def get_exclude(selected_features):
    _all_features =\
        dynamic_features() +\
        all_features() +\
        all_body_fields() +\
        all_body_orientation_fields() +\
        [
            'indexfinger_horizontal_polynomial-3',
            'indexfinger_horizontal_polynomial-2',
            'indexfinger_horizontal_polynomial-1',
            'indexfinger_horizontal_polynomial-0',
            'indexfinger_vertical_polynomial-3',
            'indexfinger_vertical_polynomial-2',
            'indexfinger_vertical_polynomial-1',
            'indexfinger_vertical_polynomial-0',
            'indexfinger_depth_polynomial-3',
            'indexfinger_depth_polynomial-2',
            'indexfinger_depth_polynomial-1',
            'indexfinger_depth_polynomial-0'
        ]

    return list(set(_all_features) - set(selected_features))


def participant_field_to_readable(field):
    return participant_fields_to_readable().get(field, field)


def body_fields_to_readable_translation():
    return {
        'indexfinger': 'index finger',
        'hand': 'hand',
        'forearm': 'forearm',
        'upperarm': 'upper arm',
        'rightShoulder': 'right shoulder',
        'hmd': 'HMD',
        'leftShoulder': 'left shoulder',
    }


def body_fields_to_readable():
    fields = body_fields_to_readable_translation()
    res = {}
    for f in fields:
        field = fields[f]
        _fields = ['%s (x)' % field, '%s (y)' % field, '%s (z)' % field]
        body_fields = {f: _fields[i] for i, f in enumerate(body_field(f))}
        res = {**res, **body_fields}
    return res


def body_field_to_readable(field):
    return body_fields_to_readable_translation().get(field, field)


def positional_axis_translation():
    return {
        'x': 'horizontal',
        'y': 'vertical',
        'z': 'depth'
    }


def positional_axis_to_readable(axis):
    return positional_axis_translation().get(axis, axis)


def positional_to_readable(field, only_axis=False):
    field, axis = field.split('.')
    field = body_field_to_readable(field)
    axis = positional_axis_to_readable(axis.lower())
    return\
        "%s %s position" % (field, axis)\
        if not only_axis\
        else axis


def orientation_axis_translation():
    return {
        'x': 'pitch',
        'y': 'yaw',
        'z': 'roll'
    }


def orientation_axis_to_readable(axis):
    return orientation_axis_translation().get(axis, axis)


def orientation_to_readable(field):
    field, axis = field.split('_')
    field = field.replace('O', '')
    field = body_field_to_readable(field)
    axis = orientation_axis_to_readable(axis.lower())
    return "%s (%s) orientation" % (field, axis)


def target_fields():
    target_fields = ['trueTarget']
    targets = ['%s.X' % f for f in target_fields]
    targets += ['%s.Y' % f for f in target_fields]
    targets += ['%s.Z' % f for f in target_fields]
    return targets


def all_target_fields():
    targets = ['id', 'pid'] + target_fields()
    return targets


def all_participant_fields():
    targets = ['id'] + participant_fields()
    return targets


def x_fields(exclude=[]):
    return ['%s.X' % f for f in body_fields() if f not in exclude]


def y_fields(exclude=[]):
    return ['%s.Y' % f for f in body_fields() if f not in exclude]


def z_fields(exclude=[]):
    return ['%s.Z' % f for f in body_fields() if f not in exclude]


def meta_fields():
    return ['pid', 'cid', 'time']


def body_field(f):
    return ['%s.X' % f, '%s.Y' % f, '%s.Z' % f]


def all_body_fields():
    return flatten([
        body_field(f) for f in body_fields()
    ])


def body_orientation_field(f):
    return ['%sO.X' % f, '%sO.Y' % f, '%sO.Z' % f]


def all_body_orientation_fields():
    return flatten([
        ['%sO.X' % f, '%sO.Y' % f, '%sO.Z' % f] for f in body_fields()
    ])


def all_fields():
    return meta_fields() + all_body_fields() + all_body_orientation_fields()


def calibration_fields():
    return ['pid', 'time'] + all_body_fields() + all_body_orientation_fields()


def flatten(l):
    return [item for sublist in l for item in sublist]


def dataframe_to_series(df, index, value):
    indecies = list(df[index])
    values = list(df[value])
    return pd.Series(values, index=indecies)


# https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
# "merges b into a"
def merge(a, b):
    for k in set(a.keys()).union(b.keys()):
        if k in a and k in b:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                yield (k, dict(merge(a[k], b[k])))
            else:
                yield (k, b[k])
        elif k in a:
            yield (k, a[k])
        else:
            yield (k, b[k])


def get_or_default(field, default, **kwargs):
    if(field in kwargs):
        return kwargs[field]
    return default


def filter_df(df, field, filter_by):
    if len(filter_by) == 0:
        return df
    df = df[df[field].isin(filter_by)]
    df.reset_index(drop=True, inplace=True)
    return df


def add_column(df, col, val):
    vals = [val] * len(df.index)
    insert = pd.DataFrame({col: vals})
    df = insert.join(df)
    return df


def marker_cycle():
    return itertools.cycle([
        '^', 'o', 's', 'p', '*', 'X', 'D', '8', 'P'
    ])


def color_cycle():
    return itertools.cycle([
        '#AFB42B', '#D32F2F', '#536DFE', '#388E3C', '#FF9800',
        '#E64A19', '#7C4DFF', '#5D4037', '#0288D1', '#009688'
    ])


def check_file(path):
    if not os.path.exists(path):
        print("%s does not exist" % path)


def check_dimensions(X, Xs, additional_rows=0, additional_columns=0, name=''):
    Xsx, Xsy = Xs.shape
    Xx, Xy = X.shape
    str_name = " (%s)" % name if name else name
    if Xsx != Xx+additional_rows:
        raise Exception(
            "too many rows in result, %d != %d%s" % (Xsx, Xx, str_name)
        )
    if Xsy != Xy+additional_columns:
        raise Exception(
            "too many columns in result, %d != %d%s" % (Xsy, Xy+1, str_name)
        )
    if(name in Xs and Xs[[name]].isnull().values.any()):
        # warnings.warn("result contains null values%s" % str_name)
        pass
    if not (X[all_fields()].equals(Xs[all_fields()])):
        warnings.warn("basic fields are not the equal%s" % str_name)


def get_targets(xs=[], ys=[], zs=[], exclude=[]):
    xs = xs if len(xs) > 0 else get_horizontal_targets()
    ys = ys if len(ys) > 0 else get_vertical_targets()
    zs = zs if len(zs) > 0 else get_depth_targets()

    def d(x, y, z):
        return {'x': x, 'y': y, 'z': z}

    return [
        d(x, y, z)
        for x in xs
        for y in ys
        for z in zs
        if d(x, y, z) not in exclude
    ]


def get_horizontal_targets():
    return [-1, 0, 1]


def get_vertical_targets():
    return [.49, 1.49, 2.49]


def get_depth_targets():
    return [1.5, 2.5, 3.5]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def ensure_dir_exists(path, is_file=False, drop_file=True):
    if is_file:
        dirname = os.path.dirname(path)
    else:
        dirname = path
    if dirname:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if drop_file and os.path.isfile(path):
            os.remove(path)


def distance_permutations():
    fields = difference(
        body_fields(),
        []
    )
    distance_perm = itertools.permutations(fields, 2)
    return set([tuple(sorted(dp)) for dp in distance_perm])


def distance_key(a, b):
    return "%s_%s_distance" % (a, b)


def get_all_distance_keys():
    return [distance_key(a, b) for a, b in distance_permutations()]


def distance_key_to_readable(k):
    a, b, _ = k.split('_')
    a, b = body_field_to_readable(a), body_field_to_readable(b)
    return "Distance between %s and %s" % (a, b)


def get_all_orientation_keys():
    return list(map(field_to_orientation_key, all_body_orientation_fields()))


def field_to_orientation_key(field):
    return field.replace('.', '_')


def chunks(lst, n):
    lst = sorted(lst)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def difference(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def unique_cominations(elems):
    for i in range(len(elems)+1):
        yield itertools.combinations(elems, i)


def unique_permutations(elems):
    for i in range(len(elems)+1):
        perm = itertools.permutations(elems, i)
        yield set([tuple(sorted(p)) for p in perm])


def get_all_new_fields():
    return get_all_new_body_orientation_fields() + get_all_new_body_fields()


def get_all_new_body_orientation_fields():
    return flatten([
        body_field("%sO" % f) for f in new_body_fields()
    ])


def get_all_new_body_fields():
    return flatten([
        body_field(f) for f in new_body_fields()
    ])


def new_body_fields():
    return [
        'indexFinger',
        'hand',
        'forearm',
        'upperArm',
        'rightShoulder',
        'hmd',
        'leftShoulder',
    ]


def new_x_fields(exclude=[]):
    return ['%s.X' % f for f in new_body_fields() if f not in exclude]


def new_y_fields(exclude=[]):
    return ['%s.Y' % f for f in new_body_fields() if f not in exclude]


def new_z_fields(exclude=[]):
    return ['%s.Z' % f for f in new_body_fields() if f not in exclude]


def new_targets():
    return body_field('target')


def rename_new_to_old(p, X, y, c):
    new = get_all_new_fields()
    old = all_body_fields() + all_body_orientation_fields()
    d = dict(zip(new, old))
    t_new = ['cid'] + new_targets()
    t_old = ['id'] + target_fields()
    t = dict(zip(t_new, t_old))
    ps = {'pid': 'id', 'indexFingerLength': 'fingerLength'}

    p = p.rename(columns=ps)
    p['index'] = p['id'].values
    p = p.set_index('index')
    fs = participant_fields(withMeta=False)
    p[fs] = p[fs] * 100
    X = X.rename(columns=d)
    X['id'] = X["pid"].astype(str) + "_" + X["cid"].astype(str)
    y = y.rename(columns=t)
    c = c.rename(columns=d).set_index('calid')

    return p, X, y, c


def parse_tuple(string):
    import ast
    s = ast.literal_eval(str(string))
    if type(s) == tuple:
        return s
    elif type(s) == float:
        return (s, )


def list_get_or_default(ls, f, d=None):
    try:
        return ls.index(f)
    except ValueError:
        return d


def get_positions_from_targets(target_fields):
    return [t.replace('trueTarget.', '').lower() for t in target_fields]


def extend_targets_from_target_fields(
    xs, target_fs, replace=[0, 1.49, 2.5]
):
    xs = np.array(xs).reshape((-1, len(target_fs)))
    sj = [
        target_fs.index(tf) if tf in target_fs else None
        for tf in target_fields()
    ]
    ys = np.zeros((xs.shape[0], 3))
    for i, m in enumerate(sj):
        ys[:, i] = xs[:, m] if m is not None else replace[i]
    return ys


def float_or_list_to_tuple(fl):
    if type(fl) == float:
        fl = (fl, )
    return tuple(fl)


def only_endpoints(X):
    return X.groupby(['pid', 'cid']).tail(1).reset_index(drop=True)


def normalize_time(X):
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
    return X_normalized_by_time


def ordinal(n):
    return "%d%s"\
        % (
            n,
            "tsnrhtdd"[(math.floor(n/10) % 10 != 1)*(n % 10 < 4) * n % 10::4]
        )


def poly_to_readable(field, short=False):
    f, exp = field.split('-')
    exp = int(exp)
    f = ' '.join(f.split('_'))
    x = ''
    if exp == 1:
        x = "$(x)$ "
    elif exp > 1:
        x = "$(x^%s)$ " % exp
    exp = ordinal(exp)
    return "%s %scoefficient of the %s" % (exp, x, f)


def save_distances(dist, **kwargs):
    csv_path = kwargs.get('csv_path', './')
    ensure_dir_exists(csv_path, is_file=True)
    dist.to_csv(csv_path)

def dataframe_from_nested_dict(dct):
    return pd.DataFrame.from_dict(
        {
            (i,j): dct[i][j] 
            for i in dct.keys() 
            for j in dct[i].keys()
        }, orient='index'
    )