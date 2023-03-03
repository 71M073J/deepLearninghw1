import numpy as np


class NN:
    def __init__(self, layers):
        self.weights = [np.random.randn(x[0], x[1]) for x in layers]
        self.biases = [np.random.randn(1, x[1]) for x in layers]
        self.activation_functions = [sigmoid for _ in range(len(layers)-1)] + [linear]
        self.activation_derivatives = [sigmoid_derivative for _ in range(len(layers)-1)] + [linear_derivative]

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation_functions[i](z)
            activations.append(a)
        return activations

    def backprop(self, x, y, learning_rate):
        activations = self.forward(x)
        delta = [None] * len(self.weights)
        delta[-1] = activations[-1] - y
        for i in reversed(range(len(delta)-1)):
            delta[i] = np.dot(delta[i+1], self.weights[i+1].T) * self.activation_derivatives[i](np.dot(activations[i], self.weights[i]) + self.biases[i])
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, delta[i])
            self.biases[i] -= learning_rate * np.sum(delta[i], axis=0, keepdims=True)
        return mse(y, activations[-1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

if __name__ == "__main__":
    net = NN([(3, 10), (10, 10), (10,150), (150,10), (10, 2)])
    x = np.random.randn(100, 3)
    y = np.concatenate((np.atleast_2d(x[:,0]).T, np.atleast_2d(x[:,1] + x[:,2]).T))
    #print(np.atleast_2d(x[0]).T, np.atleast_2d(x[1] + x[2]).T)
    #print(y)
    x_train = x
    y_train = y
    learning_rate = 0.001
    batch_size = 12
    num_batches = len(x_train) // batch_size
    losses = []
    for i in range(1000):
        loss_val = 0
        for j in range(num_batches):
            x_batch = x_train[j * batch_size:(j + 1) * batch_size]
            y_batch = y_train[j * batch_size:(j + 1) * batch_size]
            loss_val += net.backprop(x_batch, y_batch, learning_rate)
        losses.append(loss_val / num_batches)
        if i % 100 == 0:
            print(f"Iteration {i}: loss = {loss_val / num_batches}")
