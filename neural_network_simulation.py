# Author: Matthew Alderman
# Date: 06/28/2024
# Professor Ghoraani
# Quote of the day: "done is always better than perfect" - unknown

import numpy as np
import matplotlib.pyplot as plt

# Input data
inputs = np.array([
    [1, 1], [0, 1], [1, 0], [-1, 0.5], [3, 0.5],
    [2, 0.7], [0, -1], [1, -1], [0, 2], [0, 0]
])
labels = np.array([[0], [0], [1], [1], [0], [0], [1], [1], [0], [1]])

# Adding bias term to inputs
inputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs))


class NeuralNetwork:
    def __init__(self, learning_r):
        self.weights = np.random.randn(3, 1)
        self.learning_rate = learning_r
        self.history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

    def train(self, inputs_train, labels_train, num_train_iterations):
        m = inputs_train.shape[0]

        for epoch in range(num_train_iterations):
            outputs = self.forward_propagation(inputs_train)
            errors = outputs - labels_train
            cost = np.sum(errors ** 2) / (2 * m)

            gradient = np.dot(inputs_train.T, errors) / m
            self.weights -= self.learning_rate * gradient

            self.history.append((self.weights.copy(), cost))
            print(f'Epoch {epoch + 1}/{num_train_iterations}, Cost: {cost}')


learning_rates = [0.5, 0.1, 0.01]
num_epochs = 50

# Storing the results for each learning rate
results = []

for lr in learning_rates:
    nn = NeuralNetwork(learning_r=lr)
    nn.train(inputs, labels, num_train_iterations=num_epochs)
    results.append(nn)


# Plotting the data points
def plot_data(inputs, labels):
    plt.scatter(inputs[labels[:, 0] == 0][:, 1], inputs[labels[:, 0] == 0][:, 2], marker='o', label='Class 0')
    plt.scatter(inputs[labels[:, 0] == 1][:, 1], inputs[labels[:, 0] == 1][:, 2], marker='x', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

# Plotting the decision boundary
def plot_decision_boundary(nn, inputs):
    x_values = np.linspace(-2, 4, 100)
    y_values = -(nn.weights[0] + nn.weights[1] * x_values) / nn.weights[2]
    plt.plot(x_values, y_values, label=f'LR={nn.learning_rate}')

# Plotting learning curves
def plot_learning_curve(nn):
    costs = [entry[1] for entry in nn.history]
    plt.plot(costs, label=f'LR={nn.learning_rate}')

# Plotting the final classifier line for each learning rate
plt.figure(figsize=(12, 8))
plot_data(inputs, labels)
for nn in results:
    plot_decision_boundary(nn, inputs)
plt.title('Final Classifier Line')
plt.legend()
plt.show()

# Plotting the learning curves for each learning rate
plt.figure(figsize=(12, 8))
for nn in results:
    plot_learning_curve(nn)
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.show()
