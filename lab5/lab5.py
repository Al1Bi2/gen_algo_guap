from __future__ import annotations

import copy
import random
import math
import matplotlib

from lab4.main import crossover

matplotlib.use('TkAgg')
successful_mutations = 0
total_mutations = 0
class Individual:
    def __init__(self, size=2, phenotype=None, std_deviation=None):
        if phenotype == None:
            self.phenotype = [0] * size
        else:
            self.phenotype = phenotype
            size = len(phenotype)
        if std_deviation == None:
            self.std_deviation = [0] * size
        else:
            self.std_deviation = std_deviation
            size = len(phenotype)
        self.fitness = 0

    def mutation(self):

        for i in range(len(self.phenotype)):
            mutagene = random.gauss(0, self.std_deviation[i])
            self.phenotype[i] += mutagene
            if self.phenotype[i] > 500:
                self.phenotype[i] = 500
            if self.phenotype[i] < -500:
                self.phenotype[i] = -500


    def crossover(self, other: Individual, strategy="1+1") -> list[Individual] | Individual | None:
        child = Individual(size=len(other.phenotype))
        match strategy:
            case "1+1":
                return None
            case "m+1" | "m+n":

                #choices = random.sample(range(0, 1), len(self.phenotype))
                for idx in range(len(self.phenotype)):
                    parent_idx = random.randint(0,1)
                    if parent_idx == 0:
                        child.phenotype[idx] = self.phenotype[idx]
                        child.std_deviation[idx] = self.std_deviation[idx]
                    else:
                        child.phenotype[idx] = other.phenotype[idx]
                        child.std_deviation[idx] = other.std_deviation[idx]
                return child


def function(x_list: list[float], func: str) -> float:
    result = 0
    for xi in x_list:
        result += (eval(func))
    return result

def update_deviation(population: list[Individual], successful_mutations, total_mutations):
    if successful_mutations/total_mutations < 0.2:
        for i in range(len(population[0].std_deviation)):
            population[0].std_deviation[i] *= 0.82
    elif successful_mutations/total_mutations > 0.2:
        for i in range(len(population[0].std_deviation)):
            population[0].std_deviation[i] *= 1.22
    else:
        pass
    return population
def genetic_algo_cycle(population: list[Individual], strategy="1+1", extr="max", func="1+1",crossover_chance=1.0, mutation_chance=0.4):
    pop_size = len(population)
    for individual in population:
        individual.fitness = function(individual.phenotype, func)
    match strategy:
        case "1+1":
            global successful_mutations
            global total_mutations

            new_population = copy.deepcopy(population)
            new_population[0].mutation()

            population[0].fitness = function(population[0].phenotype, func)
            new_population[0].fitness = function(new_population[0].phenotype, func)
            if new_population[0].fitness < population[0].fitness:
                population = new_population
                successful_mutations += 1
            pass
            total_mutations += 1
            population = update_deviation(population, successful_mutations, total_mutations)
        case "m+1":
            copy_population = copy.deepcopy(population)
            new_population = []
            for i in range(len(population)//2):
                parent1  = copy_population.pop(random.randint(0, len(copy_population)-1))
                parent2 = copy_population.pop(random.randint(0, len(copy_population)-1))
                if random.random() < crossover_chance:
                    child = parent1.crossover(parent2, strategy=strategy)
                else:
                    child = parent1
                if random.random() < mutation_chance:
                    child.mutation()

                child.fitness = function(child.phenotype, func)

                new_population.append(child)

            population.extend(new_population)
            sorted_population = sorted(population,key= lambda ind:ind.fitness)
            population = copy.deepcopy(sorted_population[:pop_size])
            pass
        case "m+n":
            copy_population = copy.deepcopy(population)
            child_pop_size = 2*7 #n in (m+n)
            new_population = []
            for i in range(len(population) // 2):
                parent1 = copy_population.pop(random.randint(0, len(copy_population) - 1))
                parent2 = copy_population.pop(random.randint(0, len(copy_population) - 1))
                for _ in range(child_pop_size):
                    if random.random() < crossover_chance:
                        child = parent1.crossover(parent2, strategy=strategy)
                    else:
                        child = parent1
                    if random.random() < mutation_chance:
                        child.mutation()


                    child.fitness = function(child.phenotype, func)

                    new_population.append(child)

            population.extend(new_population)
            sorted_population = sorted(population, key=lambda ind: ind.fitness)
            population = copy.deepcopy(sorted_population[:pop_size])
            pass
        case "m,n":
            pass

    return population


def genetic_algorithm_init(population_size=20, strategy="1+1", gene_size=2, bounds=None, extr="max"):
    if bounds is None:
        bounds = [-500, 500]
    if strategy == "1+1":
        population_size = 1
    population = []
    for _ in range(population_size):
        phenotype = random.sample(range(bounds[0], bounds[1]), gene_size)
        std_deviation = [1.0] * gene_size
        individual = Individual(phenotype=phenotype, std_deviation=std_deviation)
        population.append(individual)
    return population


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_function_and_population(func: str, population: list[Individual], bounds: list[int]):
    """
    Построить график функции и решений из текущей популяции.
    Позволяет интерактивное вращение графика.
    :param func: Строка с функцией для вычисления z.
    :param population: Список объектов Individual.
    :param bounds: Границы графика [min, max].
    """
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
        x_val, y_val = individual.phenotype
        z_val = function([x_val, y_val], func)
        ax.scatter(x_val, y_val, z_val, color='red', s=50)

    ax.set_title("3D Plot of Function and Population")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Включение интерактивного режима
    plt.ion()  # Включить интерактивный режим
    plt.show(block=True)  # Блокирует выполнение, пока пользователь не закроет график

def test_mutation_crossover_accuracy(
    func: str,
    bounds: list[int],
    population_size: int,
    gene_size: int,
    max_generations: int,
    mutation_probs: list[float],
    crossover_probs: list[float],
    num_repeats: int = 3,
    target_accuracy: float = 1e-4,
    strategy = "m+1"
):
    """
    Тестирует достижение заданной точности для разных вероятностей мутации и кроссинговера с усреднением.
    :param func: Строка с функцией для оптимизации.
    :param bounds: Границы значений генов.
    :param population_size: Размер популяции.
    :param gene_size: Количество генов в индивиде.
    :param max_generations: Максимальное число поколений.
    :param mutation_probs: Список вероятностей мутации.
    :param crossover_probs: Список вероятностей кроссинговера.
    :param num_repeats: Число повторов для усреднения (по умолчанию 3).
    :param target_accuracy: Целевая точность (по умолчанию 1e-4).
    """
    results = []

    for mutation_prob in mutation_probs:
        for crossover_prob in crossover_probs:
            total_generations = 0
            total_best_fitness = 0
            success_count = 0
            print(f"mutation_prob = {mutation_prob}, crossover_prob = {crossover_prob}")
            for _ in range(num_repeats):
                population = genetic_algorithm_init(population_size, gene_size=gene_size,strategy=strategy)
                success = False

                for generation in range(max_generations):
                    population = genetic_algo_cycle(
                        population,
                        func=func,
                        strategy=strategy,
                        crossover_chance = crossover_prob,
                        mutation_chance=mutation_prob
                    )

                    # Найти лучшего индивида
                    best_individual = population[0]


                    if best_individual.fitness <= target_accuracy:
                        total_generations += generation + 1
                        total_best_fitness += best_individual.fitness
                        success = True
                        success_count += 1
                        break

                if not success:
                    total_generations += max_generations
                    total_best_fitness += best_individual.fitness

            # Усреднение результатов
            avg_generations = total_generations / num_repeats
            avg_best_fitness = total_best_fitness / num_repeats
            success_rate = success_count / num_repeats

            results.append({
                "mutation_prob": mutation_prob,
                "crossover_prob": crossover_prob,
                "avg_generations": avg_generations,
                "avg_best_fitness": avg_best_fitness,
                "success_rate": success_rate
            })

    # Группировка результатов по вероятности мутации и кроссинговера
    mutation_avg = {}
    crossover_avg = {}

    for mutation_prob in mutation_probs:
        filtered = [r for r in results if r["mutation_prob"] == mutation_prob]
        mutation_avg[mutation_prob] = {
            "avg_generations": sum(r["avg_generations"] for r in filtered) / len(filtered),
            "avg_best_fitness": sum(r["avg_best_fitness"] for r in filtered) / len(filtered),
            "success_rate": sum(r["success_rate"] for r in filtered) / len(filtered)
        }

    for crossover_prob in crossover_probs:
        filtered = [r for r in results if r["crossover_prob"] == crossover_prob]
        crossover_avg[crossover_prob] = {
            "avg_generations": sum(r["avg_generations"] for r in filtered) / len(filtered),
            "avg_best_fitness": sum(r["avg_best_fitness"] for r in filtered) / len(filtered),
            "success_rate": sum(r["success_rate"] for r in filtered) / len(filtered)
        }

    # Вывод результатов
    print(f"{'Mutation':<10} {'Crossover':<10} {'Avg Gen':<10} {'Avg Fitness':<12} {'Success Rate':<12}")
    print("-" * 54)
    for result in results:
        print(f"{result['mutation_prob']:<10.4f} {result['crossover_prob']:<10.4f} "
              f"{result['avg_generations']:<10.2f} {result['avg_best_fitness']:<12.4f} {result['success_rate']:<12.2%}")

    print("\nAveraged Results for Mutation Probabilities:")
    print(f"{'Mutation':<10} {'Avg Gen':<10} {'Avg Fitness':<12} {'Success Rate':<12}")
    print("-" * 44)
    for mutation_prob, data in mutation_avg.items():
        print(f"{mutation_prob:<10.4f} {data['avg_generations']:<10.2f} "
              f"{data['avg_best_fitness']:<12.4f} {data['success_rate']:<12.2%}")

    print("\nAveraged Results for Crossover Probabilities:")
    print(f"{'Crossover':<10} {'Avg Gen':<10} {'Avg Fitness':<12} {'Success Rate':<12}")
    print("-" * 44)
    for crossover_prob, data in crossover_avg.items():
        print(f"{crossover_prob:<10.4f} {data['avg_generations']:<10.2f} "
              f"{data['avg_best_fitness']:<12.4f} {data['success_rate']:<12.2%}")
    average_generation = sum(r["avg_generations"] for r in results) / len(results)
    print(f"Average Generation: {average_generation}")
    return results, mutation_avg, crossover_avg, average_generation


def main():
    strategy = "m+n"
    func = "418.9829-xi*math.sin(math.sqrt(abs(xi)))"
    bounds = [-500, 500]
    print(function([420.9687, 420.9687], "418.9829-xi*math.sin(math.sqrt(abs(xi)))"))
    population = genetic_algorithm_init(50,gene_size=2,strategy=strategy)
    for i in range(100):
        population = genetic_algo_cycle(population, func="418.9829-xi*math.sin(math.sqrt(abs(xi)))",strategy=strategy)
        if i%1 == 0:
            print("GEN: ", i, " - ", population[0].phenotype, " - ", population[0].fitness)
            plot_function_and_population(func, population, bounds)

    return
def test():
    func = "418.9829-xi*math.sin(math.sqrt(abs(xi)))"
    bounds = [-500, 500]
    population_size = 50
    #gene_size = 2
    max_generations = 500
    mutation_probs = [0.1 , 0.3,  0.5, 0.7, 0.9]
    crossover_probs = [0.2, 0.4,  0.6,  0.8, 1.0]
    strategy = "m+n"
    for gene_size in range(3,6):
        test_mutation_crossover_accuracy(
            func=func,
            bounds=bounds,
            population_size=population_size,
            gene_size=gene_size,
            max_generations=max_generations,
            mutation_probs=mutation_probs,
            crossover_probs=crossover_probs,
            num_repeats=1,
            strategy = strategy
    )

if __name__ == "__main__":
    main()
