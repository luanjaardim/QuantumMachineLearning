import pennylane as qml
from pennylane import numpy as np

from utils import *

def generate_params(num_layers = 6, embedding_type = 'amplitude', prob_put_gate=0.6, sample_elim=True, num_generations=4):
    return {
            'embedding'      : embedding_type, # angle or amplitude
            'prob_put_gate'  : prob_put_gate, # the probability to choose some gate to put at the position
            'num_layers'     : num_layers,
            'num_qubits'     : num_qubits[embedding_type],
            'num_parents'    : 2,
            'num_generations': num_generations,
            'max_num_gates'  : num_layers * prob_put_gate * num_qubits[embedding_type],
            'sample_elim'    : sample_elim,
            'batch_size'     : 20,
            'patience'       : 10,
            'epochs'         : 60,
    }

def generate_weights(params, num=1):
    # 1 is the maximum any of our gates can receive as parameters, some others can receive more.
    return np.random.randn(num, params['num_layers'], params['num_qubits'], requires_grad=True)

def create_model(params):
    import random
    model = []
    numbers = list(range(params['num_qubits']))
    for _ in range(params['num_layers']):
        model.append([])
        for q in range(params['num_qubits']):
            if random.random() < params['prob_put_gate']:
                pos = random.choice(range(len(possible_gates)))
                gate, num_wires = possible_gates[pos]
                if num_wires == 1:
                    wires = [q]
                else:
                    to_choose = numbers.copy()
                    to_choose.remove(q) # removing the current qubit from the possibilities
                    wires = [q, random.choice(to_choose)]
                model[-1].append(Gate(gate, wires, pos))
            else:
                model[-1].append(None)
    return model

class QuantumCircuit:
    # TODO: Change params to vary the circuit to be executed
    def __init__(self, params, model = None) -> None:
        self.params = params
        # Create out device
        dev = qml.device("default.qubit", wires=self.params['num_qubits'])
        # AngleEmbedding encodes N features into n qubits
        # AmplitudeEmbedding encodes 2^n features into n qubits
        # for our dataset the image has 64 pixels, our features,
        # so 64 qubits for 'angle' and 6 for 'amplitude'
        from functools import partial
        self.embedding = partial(qml.AmplitudeEmbedding, normalize=True) if self.params['embedding'] == 'amplitude' else qml.AngleEmbedding
        self.model = create_model(self.params) if model is None else model
        self.circuit = qml.qnode(dev, interface="autograd")(self.__circuit)

    def layer(self, w, layer):
        for gate in range(self.params['num_qubits']):
            if self.model[layer][gate] is not None:
                self.model[layer][gate].gate(w[layer])

    def __circuit(self, weights, f):
        self.embedding(features=f, wires=range(self.params['num_qubits']))

        for i, layer_weights in enumerate(weights):
            self.layer(layer_weights, i)

        return qml.expval(qml.PauliZ(0))

    def variational_classifier(self, weights, bias, x):
        return self.circuit(weights, x) + bias

    def train(self, weights_init, bias_init, X_train, X_test, y_train, y_test):

        def cost(weights, bias, X, Y):
            predictions = [self.variational_classifier(weights, bias, x) for x in X]
            return square_loss(Y, predictions)

        def accuracy(labels, predictions):
            acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
            acc = acc / len(labels)
            return acc

        def square_loss(labels, predictions):
            # We use a call to qml.math.stack to allow subtracting the arrays directly
            return np.mean((labels - qml.math.stack(predictions)) ** 2)

        # Printing circuit model
        drawer = qml.draw(self.circuit)
        print(drawer(weights_init[0], X_train[0]))
        print('Encoded matrix:')
        print(model_to_encoded_matrix(self.model))

        opt = qml.AdamOptimizer(0.1, eps=1e-5)

        batch_size = self.params['batch_size']
        size_train = len(X_train)

        # train the variational classifier
        weights = weights_init
        bias = bias_init
        avg_acc = 0.0
        for w, b in zip(weights, bias):
            best_acc = -1
            cnt = 0
            for it in range(self.params['epochs']):
                for i in range(0, size_train, batch_size):
                    end = i + batch_size
                    # print(f"Batch [i: {i}, end: {end}]")
                    feats_train_batch = X_train[i:(end if end < size_train else size_train)]
                    Y_train_batch = y_train[i:(end if end < size_train else size_train)]

                    # Update the weights by one optimizer step
                    w, b, _, _ = opt.step(cost, w, b, feats_train_batch, Y_train_batch)

                # Compute predictions on train and validation set
                predictions_train = np.sign(self.variational_classifier(w, b, X_train))
                predictions_val = np.sign(self.variational_classifier(w, b, X_test))

                # Compute accuracy on train and validation set
                acc_train = accuracy(y_train, predictions_train)
                acc_val = accuracy(y_test, predictions_val)

                if best_acc < acc_val:
                    best_acc = acc_val
                    cnt = 0
                else:
                    cnt += 1

                if cnt > self.params['patience']:
                    break

                if (it + 1) % 2 == 0:
                    _cost = cost(w, b, X_train, y_train)
                    print(
                        f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
                        f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
                    )
            avg_acc += best_acc

        print('\nCircuit\'s average best accuracy: ', float(avg_acc / len(weights_init)), end='\n\n')
        return float(avg_acc / len(weights_init))
