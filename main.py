import numpy as np


def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class NN:
    def __init__(self, layers, learning_rate=0.1,dropout=0.0,adam=False, loss=cross_entropy):
        self.weights = [np.random.randn(x[0], x[1]) for x in layers]
        self.biases = [np.random.randn(1, x[1]) for x in layers]
        self.activation_functions = [sigmoid for _ in range(len(layers) - 1)] + [softmax]
        self.activation_derivatives = [sigmoid_derivative for _ in range(len(layers) - 1)] + [softmax_derivative]
        self.learning_rate = learning_rate
        self.loss = loss
        self.training = True
        self.dropout_prob = dropout
        self.adam = adam
        # ADAM parameters
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        self.v_w = [np.zeros(w.shape) for w in self.weights]
        self.m_b = [np.zeros(b.shape) for b in self.biases]
        self.v_b = [np.zeros(b.shape) for b in self.biases]

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, x):
        activations = [x]
        dropout_masks = [None]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation_functions[i](z)
            if i != len(self.weights) - 1:
                # Apply dropout
                if self.training:
                    mask = np.random.binomial(1, 1 - self.dropout_prob, size=a.shape) / (1 - self.dropout_prob)
                    dropout_masks.append(mask)
                    a *= mask
                else:
                    dropout_masks.append(None)
            activations.append(a)
        return activations, dropout_masks

    def backprop(self, x, y):
        activations, dropout_masks = self.forward(x)
        delta = [None] * len(self.weights)
        delta[-1] = activations[-1] - y
        for i in reversed(range(len(delta) - 1)):
            delta[i] = np.dot(delta[i + 1], self.weights[i + 1].T) * self.activation_derivatives[i](
                np.dot(activations[i], self.weights[i]) + self.biases[i])
            # Apply dropout
            delta[i] *= dropout_masks[i + 1]
        for i in range(len(self.weights)):
            grad_w = np.dot(activations[i].T, delta[i])
            grad_b = np.sum(delta[i], axis=0, keepdims=True)
            if self.adam:
                self.t += 1
                self.m_w[i] = self.beta_1 * self.m_w[i] + (1 - self.beta_1) * grad_w
                self.v_w[i] = self.beta_2 * self.v_w[i] + (1 - self.beta_2) * np.square(grad_w)
                self.m_b[i] = self.beta_1 * self.m_b[i] + (1 - self.beta_1) * grad_b
                self.v_b[i] = self.beta_2 * self.v_b[i] + (1 - self.beta_2) * np.square(grad_b)
                m_w_hat = self.m_w[i] / (1 - self.beta_1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta_2 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta_1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta_2 ** self.t)
                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            else:
                # print(np.dot(activations[i].T, delta[i]))
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta[i])
                self.biases[i] -= self.learning_rate * np.sum(delta[i], axis=0, keepdims=True)

        return self.loss(y, activations[-1])

    def train(self, x, y, epochs, batch_size):
        losses = []
        for i in range(epochs):
            loss = 0
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]
            for j in range(0, x.shape[0], batch_size):
                x_batch = x[j:j + batch_size]
                y_batch = y[j:j + batch_size]
                #activations, dropout_masks = self.forward(x_batch)
                loss += self.backprop(x_batch, y_batch)

            losses.append(loss / num_batches)
            if i < 100:
                print(f"Iteration {i}: loss = {losses[-1]}")

    def predict(self, x):
        self.training = False
        activations, _ = self.forward(x)
        self.training = True
        return activations[-1]
        #return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    net = NN([(3, 50), (50,50),(50,2)], learning_rate=0.01, dropout=0.5, adam=False)
    size = 10000
    x = np.random.randn(size, 3)
    y = np.concatenate((np.atleast_2d(x[:, 0]) > 0, np.atleast_2d(x[:, 0]) <= 0)).T
    print(y.shape)
    # print(np.atleast_2d(x[0]).T, np.atleast_2d(x[1] + x[2]).T)
    # print(y)
    x_train = x
    y_train = y
    batch_size = 100
    epochs = 50
    num_batches = len(x_train) // batch_size
    net.train(x_train, y_train, epochs, batch_size)

    preds = net.predict(x)

    print((size - np.count_nonzero(np.argmax(y, axis=1) - np.argmax(preds, axis=1)))/size)
    print(cross_entropy(y, preds))


    test_size = 1000
    x = np.random.randn(test_size, 3)
    y = np.concatenate((np.atleast_2d(x[:, 0]) > 1, np.atleast_2d(x[:, 0]) <= 1)).T
    preds = net.predict(x)
    #print(preds, argmaxs)
    print((test_size - np.count_nonzero(np.argmax(y, axis=1) - np.argmax(preds, axis=1)))/test_size)
    print(cross_entropy(y, preds))
