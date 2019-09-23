# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import cross_val_score
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def test_sklearn():
    from sklearn.tree import DecisionTreeClassifier

    def model(para, X_train, y_train):
        model = DecisionTreeClassifier(
            criterion=para["criterion"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 21),
            "min_samples_split": range(2, 21),
            "min_samples_leaf": range(1, 21),
        }
    }

    opt = Hyperactive(search_config)
    opt.fit(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_xgboost():
    from xgboost import XGBClassifier

    def model(para, X_train, y_train):
        model = XGBClassifier(
            n_estimators=para["n_estimators"], max_depth=para["max_depth"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {model: {"n_estimators": range(2, 20), "max_depth": range(1, 11)}}

    opt = Hyperactive(search_config)
    opt.fit(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_lightgbm():
    from lightgbm import LGBMClassifier

    def model(para, X_train, y_train):
        model = LGBMClassifier(
            num_leaves=para["num_leaves"], learning_rate=para["learning_rate"]
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "num_leaves": range(2, 20),
            "learning_rate": [0.001, 0.005, 00.01, 0.05, 0.1, 0.5, 1],
        }
    }

    opt = Hyperactive(search_config)
    opt.fit(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_catboost():
    from catboost import CatBoostClassifier

    def model(para, X_train, y_train):
        model = CatBoostClassifier(
            iterations=para["iterations"],
            depth=para["depth"],
            learning_rate=para["learning_rate"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "iterations": [1],
            "depth": range(2, 10),
            "learning_rate": [0.001, 0.005, 00.01, 0.05, 0.1, 0.5, 1],
        }
    }

    opt = Hyperactive(search_config)
    opt.fit(X, y)
    # opt.predict(X)
    # opt.score(X, y)


def test_keras():
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
    from keras.datasets import cifar10
    from keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train[0:1000]
    y_train = y_train[0:1000]

    X_test = X_train[0:1000]
    y_test = y_train[0:1000]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    def cnn(para, X_train, y_train):
        model = Sequential()

        model.add(
            Conv2D(
                filters=para["filters.0"],
                kernel_size=para["kernel_size.0"],
                activation="relu",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(X_train, y_train, epochs=1)

        loss, score = model.evaluate(x=X_test, y=y_test)

        return score

    search_config = {cnn: {"filters.0": [32, 64], "kernel_size.0": [3, 4]}}

    opt = Hyperactive(search_config)
    opt.fit(X_train, y_train)
    # opt.predict(X)
    # opt.score(X, y)


"""
def test_pytorch():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    def cnn(para, X_train, y_train):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )

        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

        return running_loss

    search_config = {cnn: {"filters.0": [32, 64], "kernel_size.0": [3, 4]}}

    opt = Hyperactive(search_config)
    opt.fit(None, None)
    # opt.predict(X)
    # opt.score(X, y)



def test_chainer():
    def cnn(para, X_train, y_train):
        pass

    search_config = {cnn: {"filters.0": [32, 64], "kernel_size.0": [3, 4]}}
    opt = Hyperactive(search_config)
    opt.fit(None, None)
    # opt.predict(X)
    # opt.score(X, y)
"""
