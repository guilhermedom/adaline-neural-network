import math

import numpy as np


def create_input(n, noise):
    dataset_matrix = []
    y_matrix = []
    dataset = []

    for _ in range(0, n):
        #a = np.full((5, 5), -1, dtype=int)
        y = [[1, -1, -1, -1, 1],
             [-1, 1, -1, 1, -1],
             [-1, -1, 1, -1, -1],
             [-1, -1, 1, -1, -1],
             [-1, -1, 1, -1, -1]]
        #Y_mat = np.array(Y).reshape(-1, 5)

        for _ in range(noise):
            pos_x = np.random.randint(low=5, size=None)
            pos_y = np.random.randint(low=5, size=None)
            y[pos_x][pos_y] = -1

        dataset_matrix.append(y)
        y_matrix.append(1)

        flat_list = [item for sublist in y for item in sublist]
        flat_list.append(1)
        dataset.append(flat_list)

        y_inv = [[-1, -1, 1, -1, -1],
                 [-1, -1, 1, -1, -1],
                 [-1, -1, 1, -1, -1],
                 [-1, 1, -1, 1, -1],
                 [1, -1, -1, -1, 1]]

        for _ in range(noise):
            pos_x = np.random.randint(low=5, size=None)
            pos_y = np.random.randint(low=5, size=None)
            y_inv[pos_x][pos_y] = -1

        dataset_matrix.append(y)
        y_matrix.append(-1)

        flat_list = [item for sublist in y_inv for item in sublist]
        flat_list.append(-1)
        dataset.append(flat_list)

    np_dataset = np.array(dataset)
    np.random.shuffle(np_dataset)
    np.savetxt("input.csv", np_dataset, delimiter=",", fmt='%d')

    return np_dataset

def train_test_split(dataset, percentage):
    np.random.shuffle(dataset)

    index_train_x = math.floor(percentage * dataset.shape[0])

    train = dataset[:index_train_x, :]
    y_train = train[:, -1]
    X_train = train[:, :-1]

    test = dataset[index_train_x:, :]
    y_test = test[:, -1]
    X_test = test[:, :-1]

    return (X_train, y_train, X_test, y_test)

def accuracy_score(y_pred, y_test):
    acc_counter = 0
    print(y_pred.shape[0])

    for i in range(0, y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            acc_counter += 1
    return 1 / (y_pred.shape[0]) * acc_counter