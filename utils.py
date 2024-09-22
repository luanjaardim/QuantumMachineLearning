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
    indexes = [ i for i in range(len(labels)) if labels[i] == 1 ]
    # Database decrease for improve learning of the non-trivial class
    if params['sample_elim']:
        limit_len = 400
        for i in range(len(labels)):
            if i not in indexes: indexes.append(i)
            if len(indexes) == limit_len: break
    elif not params['sample_elim']:
        to_mirror = dataset.images[indexes]
        v_mirrored = np.array(list(map(lambda m: m[::-1], to_mirror)))
        over_sampled = []
        for i in range(len(to_mirror)):
            over_sampled.extend([
                    (to_mirror[i].reshape((64,)), 1),       # Original samples
                    (v_mirrored[i].reshape((64,)), 1),       # Original samples mirrored vertically
                    (np.array(list(map(lambda x: x[::-1], to_mirror[i]))).reshape((64,)), 1), # Original samples mirrored horizontally
                    (np.array(list(map(lambda x: x[::-1], v_mirrored[i]))).reshape((64,)), 1), # Original samples mirrored both vertically and horizontally
                ])
        print(len(over_sampled))
        print(over_sampled[0][0].shape)
        limit_len = 700
        not_non_trivial_indexes = [ i for i in range(len(labels)) if labels[i] != 1 ]
        for i in range(len(not_non_trivial_indexes[:limit_len])):
            over_sampled.append((dataset.data[i], -1))
        samples, labels = map(lambda x: np.array(list(x)), list(zip(*over_sampled)))

    # AngleEmbedding
    if params['embedding'] == "angle":
        samples = np.array(list(map(lambda m: m[2:6, 2:6].flatten(), dataset.images)))

    return train_test_split(samples[indexes] if params['sample_elim'] else samples,
                            labels[indexes] if params['sample_elim'] else labels,
                            test_size=test_size,
                            train_size=train_size,
                            random_state=state)

def encode_gate(g):
    if g is None: return -1
    # The maximum value for the index is 8, so we need 4 bits to encode each value
    if len(g.wires) == 1: return (g.wires[0] << 4) | g.index
    else: return (g.wires[1] << 8) | (g.wires[0] << 4) | g.index

def decode_gate(v):
    v_int = int(v)
    if v_int == -1: return None # Identity gate

    index = (v_int & 0b1111) % len(possible_gates) # as a mutation can happen we need to ensure the index is valid
    gate, num_wires = possible_gates[index]
    start_ind = ((v_int >> 4) & 0b1111) % 6 # TODO: make this 6 generic
    wires = [start_ind]
    if num_wires == 2:
        wires.append(((v_int >> 8) & 0b1111) % 6) # TODO: make this 6 generic

    return Gate(gate, wires, index)

def model_to_encoded_matrix(model):
    m = []
    for line in model:
        m.append([])
        for gate in line:
            m[-1].append(encode_gate(gate))
    return m

def encoded_matrix_to_model(m):
    model = []
    for line in m:
        model.append([])
        for i in line:
            model[-1].append(decode_gate(i))
    return model

class Gate:
    def __init__(self, gate, wires, index) -> None:
        self.index = index # Gate position in possible_gates list
        # Check if the port has the correct quantity of wires
        if gate in [ qml.CRX, qml.CRY, qml.CRZ, qml.CNOT ]: assert(len(wires) == 2)
        else: assert(len(wires) == 1)
        self.wires = wires
        if gate not in [qml.X, qml.CNOT, qml.Hadamard]: self.gate = lambda x: gate(x, wires=self.wires)
        else: self.gate = lambda _: gate(wires=self.wires) # X gate, Hadamard and CNOT do not receive parameters
    def __str__(self) -> str:
        return f"[ ${self.gate.__name__} : {self.wires}]"
