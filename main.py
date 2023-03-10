import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(y_true, y_pred, eps=1e-180):

    y_pred = np.clip(y_pred, eps, 1. - eps)

    N = y_true.shape[0]
    suma = np.sum(y_true * np.log(y_pred))
    return -suma/N

def sigmoid(x):
    x = np.clip(x, -512, 512)

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
                if self.training and self.dropout_prob > 0:
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
            if self.dropout_prob > 0:
                delta[i] *= dropout_masks[i + 1]

        self.t += 1
        for i in range(len(self.weights)):
            grad_w = np.dot(activations[i].T, delta[i])
            grad_b = np.sum(delta[i], axis=0, keepdims=True)#DELTA ZA BIASE POSEBEJ !!
            if self.adam:
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
                # DELTA ZA BIASE POSEBEJ !!
                self.biases[i] -= self.learning_rate * np.sum(delta[i], axis=0, keepdims=True)

        a = self.loss(y, activations[-1])
        return a
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
                curlos = self.backprop(x_batch, y_batch)
                loss += curlos
                print(curlos)
            losses.append(loss)
            if i < 1000:
                print(f"Iteration {i}: loss = {losses[-1]}")
        plt.plot(range(len(losses) - 2), losses[2:])
        plt.show()
    def predict(self, x):
        self.training = False
        activations, _ = self.forward(x)
        self.training = True
        return activations[-1]
        #return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    import pickle
    with open("./DL_HW1/data/train_data.pckl", "rb") as f:
        x = pickle.load(f)
    y = np.zeros((x["data"].shape[0],x["labels"].max() + 1))
    y[range(y.shape[0]),x["labels"]] = 1
    #net = NN([(3072,1024),(1024,256),(256,64),(64,10)], learning_rate=0.1, dropout=0.1, adam=True)
    net = NN([(3072,256),(256,10)], learning_rate=0.1, dropout=0.01, adam=False)
    net.train(x["data"], y, 10, 1000)

    with open("./DL_HW1/data/test_data.pckl", "rb") as ft:
        xt = pickle.load(ft)

    preds = net.predict(x["data"])
    print((x["data"].shape[0] - np.count_nonzero(x["labels"] - np.argmax(preds, axis=1))) / x["data"].shape[0])
    quit()
    input_size = 2
    net = NN([(input_size, 1),(1,2)], learning_rate=0.1, dropout=0.0, adam=False)
    size = 1000
    x = np.random.randn(size, input_size)
    y = (-x[:,0] + x[:,1]) > 0
    print(y.shape)
    # print(np.atleast_2d(x[0]).T, np.atleast_2d(x[1] + x[2]).T)
    # print(y)
    x_train = x
    y_train = np.atleast_2d(y.astype(np.float32)).T
    y_train = np.concatenate((y_train.T, 1 - y_train.T)).T

    batch_size = 100
    epochs = 100
    num_batches = len(x_train) // batch_size
    net.train(x_train, y_train, epochs, batch_size)

    preds = net.predict(x)

    print((size - np.count_nonzero(np.argmax(y_train, axis=1) - np.argmax(preds, axis=1)))/size)
    print(cross_entropy(y_train, preds))


    test_size = 100000
    x = np.random.randn(test_size, input_size)
    y = (-x[:,0] + x[:,1]) > 0
    y_train = np.atleast_2d(y.astype(np.float32)).T
    y = np.concatenate((y_train.T, 1 - y_train.T)).T
    preds = net.predict(x)
    #print(preds, argmaxs)
    print((test_size - (np.count_nonzero(np.argmax(y, axis=1) - np.argmax(preds, axis=1))))/test_size)
    print(cross_entropy(y, preds))
