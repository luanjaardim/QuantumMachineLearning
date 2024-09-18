from circuit import QuantumCircuit, generate_params, generate_weights
import pennylane as qml
from pennylane import numpy as np
from utils import *
from functools import partial
np.random.seed(state)

# qml.AngleEmbedding
# qml.AmplitudeEmbedding

params = generate_params()

X_train, X_test, y_train, y_test = import_database(params)

c = QuantumCircuit(params)

drawer = qml.draw(c.circuit)

# 1 is the maximum any of our gates can receive as parameters, some others can receive more.
weights_init = generate_weights(params, num=10)
bias_init = np.zeros(10, requires_grad=True)
print(drawer(weights_init[0], X_train))

def cost(weights, bias, X, Y):
    predictions = [c.variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

opt = qml.NesterovMomentumOptimizer(0.01)

batch_size = 5

# train the variational classifier
weights = weights_init
bias = bias_init
for w, b in zip(weights, bias):
    for it in range(params['epochs']):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, train_size * len(X_train), (batch_size,))
        feats_train_batch = X_train[batch_index]
        Y_train_batch = y_train[batch_index]

        w, b, _, _ = opt.step(cost, w, b, feats_train_batch, Y_train_batch)

        # Compute predictions on train and validation set
        predictions_train = np.sign(c.variational_classifier(w, b, X_train))
        predictions_val = np.sign(c.variational_classifier(w, b, X_test))

        # Compute accuracy on train and validation set
        acc_train = accuracy(y_train, predictions_train)
        acc_val = accuracy(y_test, predictions_val)

        if (it + 1) % 2 == 0:
            _cost = cost(w, b, X_train, y_train)
            print(
                f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
                f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
            )
