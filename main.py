import torch
from circuit import QuantumCircuit, generate_params, generate_weights
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

# qml.AngleEmbedding
# qml.AmplitudeEmbedding

X_train, X_test, y_train, y_test = import_database()

# Convert train data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create a TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Use DataLoader to create batches
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

params = generate_params()
c = QuantumCircuit(params)

drawer = qml.draw(c.circuit)

w = generate_weights(params)
b = torch.tensor(0.0, requires_grad=True)
# print(w)
# print(drawer(w, X_train[0]))
# print(c.variational_classifier(w, b, X_train[0]))

opt = qml.AdamOptimizer(0.01)

def cost(weights, bias, samples, labels):
    predictions = [c.variational_classifier(weights, bias, s) for s in samples]
    sum = 0
    for i, x in enumerate(predictions): sum += (labels[i] - x)**2
    return sum / len(samples)


for samples, labels in train_loader:

    print(c.variational_classifier(w, b, samples).shape)
    print(labels.shape)

    cost_with_data = partial(cost, samples = samples, labels = labels)

    w, b = opt.step(cost_with_data, w, b)

    acc = 0
    for sam_test, res in test_loader:
        acc += (np.sign(c.variational_classifier(w, b, sam_test)) == res).sum()

    print('Accuracy: ', acc)

    # acc = sum([ (np.sign(c.variational_classifier(w, b, sam_test)) == res) for sam_test, res in test_loader ])
    # print(acc)
