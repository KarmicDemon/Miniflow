### Shows an example of how to use miniflow

from miniflow import *
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

import numpy as np

data = load_boston()
X_ = data['data']
Y_ = data['target']

# Normalize
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

num_features = X_.shape[1]
num_hidden_layers = 10

#randomize weights/biases
W1_ = np.random.randn(num_features, num_hidden_layers)
b1_ = np.zeros(num_hidden_layers)
W2_ = np.random.randn(num_hidden_layers, 1)
b2_ = np.zeros(1)

# Miniflow ANN
X, y, W1, b1, W2, b2 = Input(), Input(), Input(), Input(), Input(), \
    Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)

cost = MSE(y, l2)

node_dict = {
    X: X_,
    y: Y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 1000

m = X_.shape[0]
batch_size = 10
steps_per_epoch = m // batch_size

graph = topological_sort(node_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

### Train
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, y_batch = resample(X_, Y_, n_samples=batch_size)

        X.value = X_batch
        y.value = y_batch

        round_trip(graph)

        stochastic_gradient_descent_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
