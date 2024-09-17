from circuit import QuantumCircuit, generate_params
import pennylane as qml
from utils import import_database

# qml.AngleEmbedding
# qml.AmplitudeEmbedding

X_train, X_test, y_train, y_test = import_database()

c = QuantumCircuit(generate_params())

drawer = qml.draw(c.circuit)
print(drawer([[1, 1] , [1, 1] , [1, 1], [1, 1] , [1, 1] , [1, 1]], X_train[0]))
