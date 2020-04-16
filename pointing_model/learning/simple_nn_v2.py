import torch
import numpy as np
from .model_interface import PointingMlModel
from .pointing_dataset import PointingDataset
import time


class SimpleNNv2(PointingMlModel):

    def __init__(self):
        super().__init__()
        self.trained_model = None

    def model(self, X):
        in_dimension = X.shape[1]
        hidden_dimension = 100
        out_dimension = 3
        return torch.nn.Sequential(
            torch.nn.Linear(in_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, out_dimension),
        ).to(self.torch_device)

    def loss(self):
        return torch.nn.MSELoss(reduction='sum')

    def optimizer(self, model, learning_rate):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def samplers(self, **kwargs):
        n_train = kwargs.get('n_train_samples', 20000)
        n_validation = kwargs.get('n_validation_samples', round(n_train/4, 0))

        train_sampler = torch.utils.data.SubsetRandomSampler(
            np.arange(n_train, dtype=np.int64)
        )
        validation_sampler = torch.utils.data.SubsetRandomSampler(
            np.arange(n_train, n_train + n_validation, dtype=np.int64)
        )
        return train_sampler, validation_sampler

    def loader(self, X, sampler, batch_size=64):
        loader = torch.utils.data.DataLoader(
            X, batch_size=batch_size,
            sampler=sampler, num_workers=2
        )
        return loader

    # https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch
    def train(self, X, y, **kwargs):
        self.X = X
        model = self.model(X)
        dataset = PointingDataset(X, y)

        learning_rate = kwargs.get('lr', 1e-4)
        n_epochs = kwargs.get('n_epochs', 10)
        batch_size = kwargs.get('batch_size', 32)

        loss = self.loss()
        optimizer = self.optimizer(model, learning_rate)

        validation_size = kwargs.get('validation_pct', 0.2)
        n_validation_samples = round(len(dataset)*validation_size, 0)

        train_sampler, validation_sampler = self.samplers(
            n_train_samples=len(dataset)-n_validation_samples,
            n_validation_samples=n_validation_samples
        )
        train_loader = self.loader(
            dataset, train_sampler, batch_size=batch_size
        )
        val_loader = self.loader(
            dataset, validation_sampler, batch_size=128
        )

        n_batches = len(train_loader)
        training_start_time = time.time()

        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = n_batches // 5
            if print_every <= 0:
                print_every = 1
            start_time = time.time()
            total_train_loss = 0

            for i, data in enumerate(train_loader, 0):
                # Get inputs
                tinputs, tlabels = data

                # Wrap them in a Variable object
                inputs = torch.tensor(tinputs, dtype=self.torch_dtype)\
                    .to(self.torch_device)
                labels = torch.tensor(tlabels, dtype=self.torch_dtype)\
                    .to(self.torch_device)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = model(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()

                # Print every Xth batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print(
                        "Epoch {}, {:d}%\t train_loss: {:.2f} took: {:.2f}s"
                        .format(
                            epoch+1, int(100 * (i+1) / n_batches),
                            running_loss / print_every,
                            time.time() - start_time
                        )
                    )
                    # Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            total_val_loss = 0
            for vinputs, vlabels in val_loader:

                # Wrap tensors in Variables
                vinputs = torch.tensor(vinputs, dtype=self.torch_dtype)\
                    .to(self.torch_device)
                vlabels = torch.tensor(vlabels, dtype=self.torch_dtype)\
                    .to(self.torch_device)

                # Forward pass
                val_outputs = model(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

            print(
                "Validation loss = {:.2f}"
                .format(total_val_loss / len(val_loader))
            )

        print(
            "Training finished, took {:.2f}s"
            .format(time.time() - training_start_time)
        )
        self.trained_model = model

    def predict(self, X):
        if self.trained_model is None:
            raise Exception("ml model not trained")
        print(self.X.shape, X.shape)
        X = torch.tensor(
                X.values.astype(np.float), dtype=self.torch_dtype
            ).to(self.torch_device)
        return self.trained_model(X)
