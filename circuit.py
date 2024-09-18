import pennylane as qml
from pennylane import numpy as np

from utils import *

def generate_params(num_layers = 2, embedding_type = 'amplitude', prob_put_gate=0.6):
    return {
            'embedding'     : embedding_type, # angle or amplitude
            'prob_put_gate' : prob_put_gate, # the probability to choose some gate to put at the position
            'num_layers'    : num_layers,
            'num_qubits'    : num_qubits[embedding_type],
            'max_num_gates' : num_layers * prob_put_gate * num_qubits[embedding_type],
    }

def generate_weights(params):
    return np.random.rand(params['num_qubits'], params['num_layers'])

def create_model(params):
    import random
    model = []
    numbers = list(range(params['num_qubits']))
    for q in range(params['num_qubits']):
        model.append([])
        for _ in range(params['num_layers']):
            if random.random() > params['prob_put_gate']:
                gate, num_wires = random.choice(possible_gates)
                if num_wires == 1:
                    wires = [q]
                else:
                    to_choose = numbers.copy()
                    to_choose.remove(q) # removing the current qubit from the possibilities
                    wires = [q, random.choice(to_choose)]
                model[-1].append(Gate(gate, wires))
            else:
                model[-1].append(None)
    return model

class QuantumCircuit:
    # TODO: Change params to vary the circuit to be executed
    def __init__(self, params) -> None:
        self.params = params
        # AngleEmbedding encodes N features into n qubits
        # AmplitudeEmbedding encodes 2^n features into n qubits
        # for our dataset the image has 64 pixels, our features,
        # so 64 qubits for 'angle' and 6 for 'amplitude'
        dev = qml.device("default.qubit", wires=self.params['num_qubits'])
        self.model = create_model(self.params)
        self.circuit = qml.qnode(dev, interface="autograd")(self.__circuit)

    def layer(self, w, layer):
        for gate in range(self.params['num_layers']):
            if self.model[layer][gate] is not None:
                self.model[layer][gate].gate(w[layer][gate])

    def __circuit(self, weights, f):
        qml.AmplitudeEmbedding(features=f, wires=range(self.params['num_qubits']), normalize=True)

        for i, layer_weights in enumerate(weights):
            self.layer(layer_weights, i)

        return qml.expval(qml.PauliZ(0))

    def variational_classifier(self, weights, bias, x):
        return self.circuit(weights, x) + bias
