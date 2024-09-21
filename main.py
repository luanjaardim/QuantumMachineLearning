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

initial_population = list(map(
    lambda model: [ x for xs in model_to_encoded_matrix(model) for x in xs ],
    [QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model, QuantumCircuit(params).model]))

ga = pygad.GA(
        num_generations=params['num_generations'],
        num_parents_mating=2,
        fitness_func=fit_func,
        num_genes= params['num_layers'] * params['num_qubits'],
        sol_per_pop=4, # solutions per population
        initial_population=initial_population,
        mutation_percent_genes=5
)

ga.run()

solution, solution_fitness, solution_idx = ga.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
