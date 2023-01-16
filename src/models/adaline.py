# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from ..utils import create_input, accuracy_score, train_test_split


class Adaline():
    def __init__(self, iterations=1000, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.loss_list = []

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

sample_dataset = create_input(n = 50, noise = 5)

train_x, train_y, test_x, test_y = train_test_split(sample_dataset, 0.7)

model = Adaline(iterations = 15, learning_rate = 0.001)

model.fit(train_x, train_y)

plt.plot(range(1, len(model.loss_list) + 1), model.loss_list, marker = 'o', color = 'blue')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()

print(accuracy_score(model.predict(test_x), test_y))
