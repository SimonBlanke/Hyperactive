import numpy as np


def create_search_space(size, dim):
    _dict_ = {}
    for i in range(dim):
        _dict_[str(i)] = size

    return _dict_


def sphere_function(para, X_train, y_train):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue
        loss.append(para[key] * para[key])

    return -np.array(loss).sum()


def rastrigin_function(para, X_train, y_train):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue

        loss_1d = 1 + para[key] * para[key] - np.cos(2 * np.pi * para[key])
        loss.append(loss_1d)

    return -(np.array(loss).sum())


sphere_function_search_config = {
    sphere_function: create_search_space(range(-10, 10), 5)
}


rastrigin_function_search_config = {
    rastrigin_function: create_search_space(range(-10, 10), 5)
}
