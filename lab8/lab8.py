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
        self.c1 = 1.5
        self.c2 = 0.5
        self.best_coords = [0] * size
        self.fitness = 0

def fitness(coefs,X_test,y_test) -> float:
    result= 0
    for i in range(len(X_test)):
        y_pred = coefs[0]*X_test[i]**coefs[1]
        result+= (y_test[i]-y_pred)**2
        result = result**0.5
        result = result/len(X_test)
    return result



def PSO_init( population_size,X_train,y_train, coords_size: int = 2, bounds: list[int] = [-500, 500]):
    population = []
    if bounds is None:
        exit("Error: bounds is None")
    global_best_coords = [0] * coords_size
    for _ in range(population_size):
        coords = random.sample(range(bounds[0], bounds[1]), coords_size)
        individual = Individual(coords=coords)
        individual.best_coords = copy.deepcopy(coords)
        if fitness(individual.best_coords,X_train,y_train) < fitness(global_best_coords,X_train,y_train ):
            global_best_coords = copy.copy(individual.best_coords)
        population.append(individual)

    return population, global_best_coords

def PSO_step(population: list[Individual],X_train,y_train, bounds: list[int] = [-500, 500], gl_b_c = [0, 0]):
    global_best_coords = gl_b_c
    for individual in population:

        if fitness(individual.coords,X_train,y_train ) < fitness(individual.best_coords,X_train,y_train ):
            individual.best_coords = copy.deepcopy(individual.coords)

        if fitness(individual.best_coords,X_train,y_train ) < fitness(global_best_coords,X_train,y_train ):
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
def get_user_input():
    print("=== Параметры PSO ===")
    generations = int(input("Введите количество поколений: "))
    population_size = int(input("Введите размер популяции: "))
    c1 = float(input("Введите значение параметра c1: "))
    c2 = float(input("Введите значение параметра c2: "))
    return generations, population_size, c1, c2


def plot_training_error(error_history):
    plt.figure()
    plt.plot(range(len(error_history)), error_history, label='Ошибка на обучении')
    plt.xlabel('Поколение')
    plt.ylabel('Ошибка')
    plt.title('График ошибки обучения')
    plt.legend()
    plt.show()


def plot_error_bars(train_error, test_error):
    categories = ['Тренировочная выборка', 'Тестовая выборка']
    errors = [train_error, test_error]

    plt.figure()
    plt.bar(categories, errors, color=['blue', 'orange'])
    plt.ylabel('Ошибка')
    plt.title('Диаграмма ошибки на выборках')
    plt.show()

def main():
    random.seed(7)
    data = [
        [2.2, 8.4], [3.5, 10.8], [5.5, 18], [6, 24], [9.7, 25.2], [7.7, 31.2],
        [11.3, 36], [8.2, 36], [6.5, 42], [8, 42], [20, 48], [10, 48], [15, 48],
        [10.4, 50], [13, 60], [14, 60], [19.7, 60], [32.5, 60], [31.5, 60],
        [12.5, 62], [15.4, 70], [20, 72], [7.5, 72], [16.3, 82], [15, 90],
        [11.4, 98.8], [21, 107], [16, 114], [25.9, 117.6], [24.6, 117.6],
        [29.5, 120], [19.3, 155], [32.6, 170], [35.5, 192], [38, 210],
        [48.5, 239], [47.5, 252], [70, 278], [66.6, 300], [66.6, 352.8],
        [50, 370], [79, 400], [90, 450], [78, 571.4], [100, 215], [150, 324],
        [100, 360], [100, 360], [190, 420], [115.8, 480], [101, 750],
        [161.1, 815], [284.7, 973], [227, 1181], [177.9, 1228], [282.1, 1368],
        [219, 2120], [423, 2300], [302, 2400], [370, 3240]
    ]


    train_size = int(len(data) * 0.7)

    # Перемешиваем данные
    random.shuffle(data)

    # Разделяем данные на обучающую и тестовую выборки
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Извлекаем X и y из обучающей и тестовой выборок
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    bounds = [-4, 5]
    generations, population_size, c1, c2 = get_user_input()

    population, best_coords = PSO_init(population_size, X_train, y_train, 2, bounds)
    error_history = []

    for individual in population:
        individual.c1 = c1
        individual.c2 = c2
    #plot_function_and_population(func, population, bounds,False)
    for i in range(generations):
        population, best_coords = PSO_step(population,X_train, y_train, bounds,best_coords)
        train_error = fitness(best_coords, X_train, y_train)
        error_history.append(train_error)
        if i%1==0:
            print(f"GEN: {i+1} - {best_coords} - Ошибка: {train_error}")

    test_error = fitness(best_coords, X_test, y_test)
    print("Тестовая ошибка: ", test_error)

    plot_training_error(error_history)
    plot_error_bars(error_history[-1], test_error)

if __name__ == "__main__":
    main()