from pointing_model.loading import log
from pointing_model import utils


def export(filename="export", **kwargs):
    participants, X, y = log(**kwargs)
    X.groupby(['pid', 'cid'])\
        .tail(1)\
        .reset_index()\
        .set_index("pid")\
        .join(participants, how='outer')\
        .reset_index()\
        .rename(columns={'level_0': 'participantId'})\
        .merge(
            y.reset_index(),
            left_on=['participantId', 'cid'], right_on=['pid', 'id']
        )\
        .drop(columns=[
            'id_x', 'id_y', 'pid'
        ])\
        .rename(columns={'cid': 'collectionId'})\
        .set_index('collectionId')\
        .sort_values(by=utils.target_fields())[
            ['participantId'] + utils.target_fields() +
            utils.participant_fields() + utils.all_body_fields() +
            utils.all_body_orientation_fields()
        ]\
        .to_csv('%s.csv' % filename)
    print("exported to %s.csv" % filename)
