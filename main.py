import torch
from circuit import QuantumCircuit, generate_params, generate_weights
import pennylane as qml
from pennylane import numpy as np
from utils import *
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(state)

# qml.AngleEmbedding
# qml.AmplitudeEmbedding

X_train, X_test, y_train, y_test = import_database()

params = generate_params()
c = QuantumCircuit(params)

drawer = qml.draw(c.circuit)

# From where in the fucking hell comes this 3
weights_init = 0.01 * np.random.randn(params['num_layers'], params['num_qubits'], 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

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
for it in range(60):
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, train_size * len(X_train), (batch_size,))
    feats_train_batch = X_train[batch_index]
    Y_train_batch = y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, feats_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = np.sign(c.variational_classifier(weights, bias, X_train))
    predictions_val = np.sign(c.variational_classifier(weights, bias, X_test))

    # Compute accuracy on train and validation set
    acc_train = accuracy(y_train, predictions_train)
    acc_val = accuracy(y_test, predictions_val)

    if (it + 1) % 2 == 0:
        _cost = cost(weights, bias, X_train, y_train)
        print(
            f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
            f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
        )
