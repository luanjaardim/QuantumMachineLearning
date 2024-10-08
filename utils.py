import pennylane as qml
import random
import torch

# Make a reproducible experiment
state = 96
random.seed(state)
torch.random.manual_seed(state)

non_trivial_class = 8
num_qubits = {
        'angle' : 16, # The image was croped to qubits limitation, 64 qubits were needed
        'amplitude' : 6
}
train_size = 0.8
test_size = 0.2

# Possble gates to choose and the needed amount of wires to each of them
possible_gates = [ # Without the I
        (qml.RX, 1),
        (qml.RY, 1),
        (qml.RZ, 1),
        (qml.CRX, 2),
        (qml.CRY, 2),
        (qml.CRZ, 2),
        (qml.Hadamard, 1),
        (qml.CNOT, 2),
        (qml.X, 1),
]

def import_database(params):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import numpy as np
    dataset = datasets.load_digits()
    samples = dataset.data
    labels = np.array(list(map(lambda x: -1 if x != 8 else 1, dataset.target)))
    # Database decrease for improve learning of the non-trivial class
    if params['sample_elim']:
        limit_len = 400
        indexes = [ i for i in range(len(labels)) if labels[i] == 1 ]
        for i in range(len(labels)):
            if i not in indexes: indexes.append(i)
            if len(indexes) == limit_len: break

    # AngleEmbedding
    if params['embedding'] == "angle":
        samples = np.array(list(map(lambda m: m[2:6, 2:6].flatten(), dataset.images)))

    return train_test_split(samples[indexes], labels[indexes], test_size=test_size, train_size=train_size, random_state=state)

class Gate:
    def __init__(self, gate, wires) -> None:
        # Check if the port has the correct quantity of wires
        if gate in [ qml.CRX, qml.CRY, qml.CRZ, qml.CNOT ]: assert(len(wires) == 2)
        else: assert(len(wires) == 1)
        self.wires = wires
        if gate not in [qml.X, qml.CNOT, qml.Hadamard]: self.gate = lambda x: gate(x, wires=self.wires)
        else: self.gate = lambda _: gate(wires=self.wires) # X gate, Hadamard and CNOT do not receive parameters
