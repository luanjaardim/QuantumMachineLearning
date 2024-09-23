from circuit import QuantumCircuit, generate_params, generate_weights
import pennylane as qml
from pennylane import numpy as np
from utils import *
from functools import partial
import pygad
np.random.seed(state)

params = generate_params()

X_train, X_test, y_train, y_test = import_database(params)

num = 1
weights_init = generate_weights(params, num=num)
bias_init = np.zeros(num, requires_grad=True)

def fit_func(ga_instance, solution, solution_idx):
    model = encoded_matrix_to_model(solution.reshape((params['num_layers'], params['num_qubits'])))
    return QuantumCircuit(params, model).train(weights_init, bias_init, X_train, X_test, y_train, y_test)

# initial_population = list(map(
#     lambda model: [ x for xs in model_to_encoded_matrix(model) for x in xs ],
#     [QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model]))
initial_population = [
    [6, -1, 37, -1, 70, -1, 773, 24, -1, -1, -1, -1, -1, 1301, -1, -1, -1, -1, 0, 24, -1, 50, -1, 88, -1, -1, -1, -1, 835, -1, -1, 24, 38, -1, -1, -1],

    [0, -1, 34, -1, -1, -1, 0, 1299, -1, -1, -1, -1, 8, 1301, -1, 54, -1, 343, 1029, -1, -1, -1, 1351, -1, -1, 788, -1, -1, 323, 852, -1, -1, -1, -1, -1, 83],


    [6, -1, 37, -1, 70, -1, 773, 24, -1, -1, 0, -1, -1, 1301, -1, -1, -1, -1, 0, 24, -1, 50, -1, 88, -1, 24, -1, -1, -1, -1, 0, -1, -1, 56, -1, -1],


    [6, -1, 37, -1, 70, -1, 773, 24, -1, -1, -1, -1, -1, 1301, -1, -1, -1, -1, 0, 24, -1, 50, -1, 88, -1, -1, -1, -1, 835, -1, -1, 24, 38, -1, -1, -1],


    [6, -1, 36, -1, 70, -1, 773, 24, -1, -1, -1, 0, -1, 1301, -1, -1, -1, -1, 0, 24, -1, 50, -1, 88, -1, -1, -1, -1, 835, -1, -1, 24, 38, -1, -1, -1]
]

initial_population.extend(list(map(
    lambda model: [ x for xs in model_to_encoded_matrix(model) for x in xs ],
    [QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model])))

ga = pygad.GA(
        num_generations=params['num_generations'],
        num_parents_mating=2,
        fitness_func=fit_func,
        num_genes= params['num_layers'] * params['num_qubits'],
        sol_per_pop=4, # solutions per population
        initial_population=initial_population,
        mutation_percent_genes="default",
        save_best_solutions=True
)

ga.run()
small_circuit = 0
best_circuit = 0
max_id_gates = 0
max_fitness = 0
for i, (p, f) in enumerate(zip(ga.best_solutions, ga.best_solutions_fitness)):
    print(i)
    from functools import reduce
    cnt = reduce(lambda acc, x: acc+1 if x == -1 else acc, p)
    if cnt > max_id_gates:
        max_id_gates = cnt
        small_circuit = i
    if f > max_fitness:
        max_fitness = f
        best_circuit = i

print("Best Solution, with smallest circuit: ")
small_circuit_solution = ga.best_solutions[small_circuit]

d = qml.draw(QuantumCircuit(params, encoded_matrix_to_model(small_circuit_solution.reshape((params['num_layers'], params['num_qubits'])))).circuit)
print(d(weights_init[0], X_train[0]))
print(f"Fitness of the small solution : {ga.best_solutions_fitness[small_circuit]}")
print(f"Parameters of the small solution : {small_circuit_solution}")

print("\n\nBest Solution: ")
best_solution = ga.best_solutions[best_circuit]
d = qml.draw(QuantumCircuit(params, encoded_matrix_to_model(best_solution.reshape((params['num_layers'], params['num_qubits'])))).circuit)
print(d(weights_init[0], X_train[0]))
print(f"Fitness of the best solution : {ga.best_solutions_fitness[best_circuit]}")
print(f"Parameters of the best solution : {best_solution}")
