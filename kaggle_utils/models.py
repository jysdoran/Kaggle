from typing import List
import torch
from torch import nn

# Get cpu or gpu device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Numpy dataset
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], activation=nn.ReLU()):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least 2 layers are required")

        layers = []
        for size_from, size_to in zip(layer_sizes, layer_sizes[1:-1]):
            layers.append(nn.Linear(size_from, size_to))
            layers.append(activation)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits.squeeze(1)


class Trainer:
    def __init__(self, model) -> None:
        self.model = model.to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        self.batch_size = 64

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if batch % 100 == 0:
            #     loss, current = loss.item(), (batch + 1) * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def eval(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax() == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return correct, test_loss
        # print(
        #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        # )

    def train_loop(self, dataset_train, dataset_dev, epochs=5):
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size)
        dataloader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=self.batch_size)

        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            self.train(dataloader_train)
            yield self.eval(dataloader_dev)
        print("Done!")
