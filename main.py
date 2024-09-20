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

weights_init = generate_weights(params, num=5)
bias_init = np.zeros(10, requires_grad=True)

circuits = []

for i in range(params['num_circuit']):

    c = QuantumCircuit(params)

    drawer = qml.draw(c.circuit)

    print(drawer(weights_init[0], X_train))
    # TODO: return average accuracy of the circuit
    avg_acc = c.train(weights_init, bias_init, X_train, X_test, y_train, y_test)
    print('Circuit\'s average best accuracy: ', avg_acc)

    circuits.append((c, avg_acc))

# TODO: rank of trained circuits
print(circuits)
