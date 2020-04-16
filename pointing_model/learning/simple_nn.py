import torch
import numpy as np
from .model_interface import PointingMlModel


class SimpleNN(PointingMlModel):

    def __init__(self, config={}):
        super().__init__()
        self.config = config

    def model(self, in_dimension):
        hidden_dimension = 100
        out_dimension = 3
        return torch.nn.Sequential(
            torch.nn.Linear(in_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, out_dimension),
        ).to(self.torch_device)

    def loss(self):
        return torch.nn.MSELoss(reduction='sum')

    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    def train(self, X, y):
        model = self.model(X.shape[1])
        loss_fn = self.loss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        X = torch.tensor(
            X.values.astype(np.float), dtype=self.torch_dtype
        ).to(self.torch_device)
        y = torch.tensor(
            y.values.astype(np.float), dtype=self.torch_dtype
        ).to(self.torch_device)
        t = 0
        closs = 1000
        while closs > 20:
            y_pred = model(X)
            # print(y_pred, y, y_pred.shape, y.shape, sep="\n")
            loss = loss_fn(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())
                print(y_pred.data.cpu().numpy()[0], y.data.cpu().numpy()[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            closs = loss.item()
            t += 1
