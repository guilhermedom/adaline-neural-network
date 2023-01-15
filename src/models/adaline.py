# -*- coding: utf-8 -*-

import math

import numpy as np
import matplotlib.pyplot as plt


class Adaline():
    def __init__(self, iterations=15, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost_list = []

        for _ in range(self.iterations):
            predicted_value = self.linear_activation(X)

            errors = (y - predicted_value)

            self.weights[1:] += self.learning_rate * np.dot(X.T, errors)
            self.weights[0] += self.learning_rate * errors.sum()

            cost = (errors ** 2).sum() / 2
            self.cost_list.append(cost)

        return self

    def linear_activation(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.linear_activation(X) >= 0.0, 1, -1)

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

        for _ in range(0, noise):
            pos_x = np.random.randint(0, 5)
            pos_y = np.random.randint(0, 5)
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

        for _ in range(0, noise):
            pos_x = np.random.randint(0, 5)
            pos_y = np.random.randint(0, 5)
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

dataset = create_input(n = 50, noise = 5)

train_x, train_y, test_x, test_y = train_test_split(dataset, 0.7)

model = Adaline(iterations = 15, learning_rate = 0.001)

model.fit(train_x, train_y)

plt.plot(range(1, len(model.cost_list) + 1), model.cost_list, marker = 'o', color = 'blue')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()

print(accuracy_score(model.predict(test_x), test_y))
