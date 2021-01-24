import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from hyperactive import Hyperactive


"""
derived from optuna example:
https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py
"""
DEVICE = torch.device("cpu")
BATCHSIZE = 256
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


# Get the MNIST dataset.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=BATCHSIZE,
    shuffle=True,
)
valid_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
    batch_size=BATCHSIZE,
    shuffle=True,
)


def pytorch_cnn(params):
    linear0 = params["linear.0"]
    linear1 = params["linear.1"]

    layers = []

    in_features = 28 * 28

    layers.append(nn.Linear(in_features, linear0))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))

    layers.append(nn.Linear(linear0, linear1))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))

    layers.append(nn.Linear(linear1, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    model = nn.Sequential(*layers)

    # model = create_model(params).to(DEVICE)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=0.01)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

    return accuracy


search_space = {
    "linear.0": list(range(10, 200, 10)),
    "linear.1": list(range(10, 200, 10)),
}


hyper = Hyperactive()
hyper.add_search(pytorch_cnn, search_space, n_iter=5)
hyper.run()
