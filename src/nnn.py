import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

def sigmoid(x, derivate=False):
    sig = 1 / (1 + np.exp(-x))
    if derivate:
        return sig * (1 - sig)
    return sig

def relu(x, derivate=False):
    if derivate:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def mse(y, y_hat, derivate=False):
    if derivate:
        return (y_hat - y)
    return np.mean((y_hat - y) ** 2)

def initialize_parameters_deep(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L - 1):
        parameters['W' + str(l + 1)] = np.random.rand(layers_dims[l], layers_dims[l + 1]) * 2 - 1
        parameters['b' + str(l + 1)] = np.random.rand(1, layers_dims[l + 1]) * 2 - 1
    return parameters

def train(x_data, y_data, learning_rate, params, training=True):
    params['A0'] = x_data
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])
    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])
    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])
    output = params['A3']
    if training:
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['Z3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])
        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['Z2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])
        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['Z1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate
        params['b3'] -= np.mean(params['dZ3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dZ2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dZ1'], axis=0, keepdims=True) * learning_rate
    return output

if __name__ == "__main__":
    N = 1000
    X, Y = make_gaussian_quantiles(n_samples=N, n_features=2, n_classes=2)
    Y = Y[:, np.newaxis]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    layers_dims = [2, 6, 10, 1]
    params = initialize_parameters_deep(layers_dims)
    error = []
    for i in range(50000):
        output = train(X, Y, 0.001, params)
        if i % 50 == 0:
            err = mse(Y, output)
            print(err)
            error.append(err)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
