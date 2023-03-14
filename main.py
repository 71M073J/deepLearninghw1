import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy
from scipy.special import expit as logistic_sigmoid
def cross_entropy_loss(y_true, y_pred):
    eps = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -xlogy(y_true, y_pred).sum() / y_pred.shape[0]

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
    def __init__(self, layers, learning_rate=0.1,dropout=0.0,adam=False, loss=cross_entropy_loss):
        self.weights = [np.random.randn(x[0], x[1]) for x in layers]
        self.biases = [np.random.randn(1, x[1]) for x in layers]
        self.activation_functions = [logistic_sigmoid for _ in range(len(layers) - 1)] + [softmax]
        self.activation_derivatives = [sigmoid_derivative for _ in range(len(layers) - 1)] + [softmax_derivative]
        self.learning_rate = learning_rate
        self.loss = loss
        self.training = True
        self.dropout_prob = dropout
        #L2
        self.L2 = True
        self.alpha = 0.0001

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
                if self.training and self.dropout_prob > 2:
                    mask = np.random.binomial(1, 1 - self.dropout_prob, size=self.biases[i].shape) / (1 - self.dropout_prob)
                    dropout_masks.append(mask)
                    a *= mask
                else:
                    dropout_masks.append(None)
            activations.append(a)
        return activations, dropout_masks

    def backprop(self, x, y):
        activations, dropout_masks = self.forward(x)
        loss = self.loss(y, activations[-1])

        #ADD L2 reg
        values = 0
        for s in self.weights:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / x.shape[0]

        ###
        delta = [None] * len(self.weights)
        delta[-1] = activations[-1] - y

        w_update = [np.zeros_like(w) for w in self.weights]
        b_update = [np.zeros_like(b) for b in self.biases]
        w_update[-1] = activations[-2].T.dot(delta[-1]) / x.shape[0]# * dropout_masks[-1]
        b_update[-1] = np.mean(delta[-1], 0)
        if self.L2:
            w_update[-1] += (self.alpha/x.shape[0]) * self.weights[-1]



        for i in reversed(range(len(delta) - 1)):
            #delta[i] = np.dot(delta[i + 1], self.weights[i + 1].T) * self.activation_derivatives[i](
            #    np.dot(activations[i], self.weights[i]) + self.biases[i])
            delta[i] = delta[i+1].dot(self.weights[i+1].T)

            # Apply dropout
            if self.dropout_prob > 0:
                delta[i] *= dropout_masks[i + 1]
            #TODO CHECK INDEX
            delta[i] *= self.activation_derivatives[i+1](activations[i+1])
            w_update[i] = activations[i].T.dot(delta[i]) / x.shape[0]
            b_update[i] = np.mean(delta[i], 0)
            if self.L2:
                w_update[i] += self.alpha * self.weights[i] / x.shape[0]

        #TODO ADAM
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * w_update[i]
            self.biases[i] -= self.learning_rate * b_update[i]

            #self.optimizer(w_update, b_update)
        return loss
    def train(self, x, y, epochs, batch_size):
        losses = []
        timeout = 0
        cnt = 0
        for i in range(epochs):
            timeout -= 1
            if cnt < 10 and timeout <= 0 and len(losses) > 2 and (losses[-2] - losses[-1]) < (losses[-1] + losses[-2])/20:
                print("Halving LR...", losses[-2:])
                cnt += 1
                timeout = int(i/5) + 4
                self.learning_rate /= 2
            #if len(losses) > 15 and np.mean(losses[-5:]) - losses[-1] < 1:
            #    break
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
                if j%10 == 1:
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
    net = NN([(3072,512), (512,256),(256,10)], learning_rate=0.1, dropout=0.0, adam=False)
    net.train(x["data"], y, 50, 1000)

    with open("./DL_HW1/data/test_data.pckl", "rb") as ft:
        xt = pickle.load(ft)

    preds = net.predict(x["data"])
    print((x["data"].shape[0] - np.count_nonzero(x["labels"] - np.argmax(preds, axis=1))) / x["data"].shape[0])
    quit()
    input_size = 2
    net = NN([(input_size, 3),(3,2)], learning_rate=0.1, dropout=0.0, adam=False)
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
