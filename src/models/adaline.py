# -*- coding: utf-8 -*-

import numpy as np


class Adaline():
    def __init__(self, iterations=1000, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.loss_list = []

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])

        for _ in range(self.iterations):
            predicted_value = self.linear_activation(X)

            errors = (y - predicted_value)

            self.weights[1:] += self.learning_rate * np.dot(X.T, errors)
            self.weights[0] += self.learning_rate * errors.sum()

            loss = (errors ** 2).sum() / 2
            self.loss_list.append(loss)

        return self

    def linear_activation(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.linear_activation(X) >= 0.0, 1, -1)
