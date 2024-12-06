import copy
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Individual:
    def __init__(self, size=2, coords=None, std_deviation=None):
        if coords == None:
            self.coords = [0] * size
        else:
            self.coords = coords
            size = len(coords)
        if std_deviation == None:
            self.std_deviation = [0] * size
        else:
            self.std_deviation = std_deviation
            size = len(coords)
        self.size = size
        self.velocity = [0]*size
        self.c1 = 4
        self.c2 = 1
        self.best_coords = [0] * size
        self.fitness = 0

def function(x_list: list[float], func: str) -> float:
    result = 0
    for xi in x_list:
        result += (eval(func))
    return result

def plot_function_and_population(func: str, population: list[Individual], bounds: list[int],interactive=True):
    """
    Построить график функции и решений из текущей популяции.
    Позволяет интерактивное вращение графика.
    :param func: Строка с функцией для вычисления z.
    :param population: Список объектов Individual.
    :param bounds: Границы графика [min, max].
    """
    if interactive:
        matplotlib.use('TkAgg')
    # Создание сетки для функции
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Вычисление значений функции
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            yi = Y[i, j]
            Z[i, j] = function([xi, yi], func)

    # Создание графика
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Добавление точек популяции
    for individual in population:
        x_val, y_val = individual.coords
        z_val = function([x_val, y_val], func)
        ax.scatter(x_val, y_val, z_val, color='red', s=50)

    ax.set_title("3D Plot of Function and Population")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=90, azim=0, roll=0)
    ax.set_proj_type('ortho')
    # Включение интерактивного режима

    plt.show(block=True)  # Блокирует выполнение, пока пользователь не закроет график

def PSO_init( population_size: int = 20, coords_size: int = 2, bounds: list[int] = [-500, 500],func = "1+1"):
    population = []
    if bounds is None:
        exit("Error: bounds is None")
    global_best_coords = [0] * coords_size
    for _ in range(population_size):
        coords = random.sample(range(bounds[0], bounds[1]), coords_size)
        individual = Individual(coords=coords)
        individual.best_coords = copy.deepcopy(coords)
        if function(individual.best_coords, func) < function(global_best_coords, func):
            global_best_coords = copy.copy(individual.best_coords)
        population.append(individual)

    return population, global_best_coords

def PSO_step(population: list[Individual],func = "1+1", bounds: list[int] = [-500, 500], gl_b_c = [0, 0]):
    global_best_coords = gl_b_c
    for individual in population:

        if function(individual.coords, func) < function(individual.best_coords, func):
            individual.best_coords = copy.deepcopy(individual.coords)

        if function(individual.best_coords, func) < function(global_best_coords, func):
            global_best_coords = copy.deepcopy(individual.best_coords)
    for individual in population:
        for i in range(individual.size):


            r1 = random.random()
            r2 = random.random()

            individual.velocity[i] += individual.c1 * r1 * (individual.best_coords[i] - individual.coords[i]) + individual.c2 * r2 * (global_best_coords[i] - individual.coords[i])
            #individual.velocity[i] = min(max(individual.velocity[i], -1050), 1050)
            individual.coords[i] += individual.velocity[i]
            if individual.coords[i] < bounds[0]:
                individual.coords[i] = bounds[0]
                #individual.velocity[i] *= -1
            if individual.coords[i] > bounds[1]:
                individual.coords[i] = bounds[1]
                #individual.velocity[i] *= -1
            #individual.coords[i] = min(max(individual.coords[i], bounds[0]), bounds[1])
    return population, global_best_coords


def main():
    func = "418.9829-xi*math.sin(math.sqrt(abs(xi)))"
    bounds = [-500, 500]
    print(function([420.9687, 420.9687], "418.9829-xi*math.sin(math.sqrt(abs(xi)))"))
    population, best_coords = PSO_init(400,5,func = func)
    #plot_function_and_population(func, population, bounds,False)
    for i in range(100+1):
        population, best_coords = PSO_step(population, func, bounds,best_coords)
        if i%1==0:
            print("GEN: ", i, " - ", best_coords, " - ", function(best_coords, func))
            #plot_function_and_population(func, population, bounds,False)
        #print(best_coords,function(best_coords, func))
    #

if __name__ == "__main__":
    main()