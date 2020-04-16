import pandas as pd
from . import utils


class PointingModelBase():

    def __init__(self):
        self.config = {}
        self.participants = pd.DataFrame(
            columns=utils.all_participant_fields()
        )
        self.X = pd.DataFrame(columns=utils.all_fields())
        self.y = pd.DataFrame(columns=utils.all_target_fields())
        self.X_only_final = None
        self.X_normalized_by_time = None
        self.p_normalized_by_height = None
        self.X_normalized_by_height = None
        self.c_normalized_by_height = None
        self.X_normalized_xz = None

    def get_loaded_participant_ids(self):
        return self.participants.index

    def get_loaded_collection_ids(self):
        return self.y.index

    def reset(self):
        self.X_only_final = None
        self.X_normalized_by_time = None
        self.X_normalized_by_height = None
        self.X_normalized_xz = None
