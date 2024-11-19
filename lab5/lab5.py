from __future__ import annotations

import copy
import random
import math


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


def genetic_algo_cycle(population: list[Individual], strategy="1+1", extr="max", func="1+1"):
    pop_size = len(population)
    for individual in population:
        individual.fitness = function(individual.phenotype, func)
    match strategy:
        case "1+1":
            new_population = copy.deepcopy(population)
            new_population[0].mutation()

            population[0].fitness = function(population[0].phenotype, func)
            new_population[0].fitness = function(new_population[0].phenotype, func)
            if new_population[0].fitness < population[0].fitness:
                population = new_population
            pass
        case "m+1":
            copy_population = copy.deepcopy(population)
            new_population = []
            for i in range(len(population)//2):
                parent1  = copy_population.pop(random.randint(0, len(copy_population)-1))
                parent2 = copy_population.pop(random.randint(0, len(copy_population)-1))
                child = parent1.crossover(parent2, strategy=strategy)
                if random.random() < 0.4:
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
                    child = parent1.crossover(parent2, strategy=strategy)
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


def main():
    strategy = "m+n"
    print(function([420.9687, 420.9687], "418.9829-xi*math.sin(math.sqrt(abs(xi)))"))
    population = genetic_algorithm_init(50,gene_size=2,strategy=strategy)
    for i in range(100):
        population = genetic_algo_cycle(population, func="418.9829-xi*math.sin(math.sqrt(abs(xi)))",strategy=strategy)
        if i%2 == 0:
            print(population[0].phenotype, " - ", population[0].fitness)

    return


if __name__ == "__main__":
    main()
