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
)

ga.run()

solution, solution_fitness, solution_idx = ga.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
