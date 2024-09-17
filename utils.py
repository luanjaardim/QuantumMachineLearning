import pennylane as qml

state = 96
num_qubits = {
        'angle' : 64,
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

class Gate:
    def __init__(self, gate, wires) -> None:
        # Check if the port has the correct quantity of wires
        if gate in [ qml.CRX, qml.CRY, qml.CRZ, qml.CNOT ]: assert(len(wires) == 2)
        else: assert(len(wires) == 1)
        self.wires = wires
        if gate not in [qml.X, qml.CNOT, qml.Hadamard]: self.gate = lambda x: gate(x, wires=self.wires)
        else: self.gate = lambda _: gate(wires=self.wires) # X gate, Hadamard and CNOT do not receive parameters
