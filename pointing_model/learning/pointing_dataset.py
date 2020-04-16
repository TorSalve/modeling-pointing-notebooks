from torch.utils.data import Dataset
import numpy as np


class PointingDataset(Dataset):

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __getitem__(self, index):
        sX = self.X.iloc[index].values.astype(np.float)
        sy = self.y.iloc[index].values.astype(np.float)
        return sX, sy

    def __len__(self):
        return self.y.shape[0]
