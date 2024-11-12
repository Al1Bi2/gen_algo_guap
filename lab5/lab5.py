from __future__ import annotations
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
        for coord, coord_deviation in self.phenotype, self.std_deviation:
            coord += random.gauss(0, coord_deviation)

    def crossover(self, other: Individual, strategy="1+1") -> list[Individual] | Individual | None:
        child = Individual()
        match strategy:
            case "1+1":
                return None
            case "m+1":

                choices = random.sample(range(0, 1), len(self.phenotype))
                for idx, parent_idx in enumerate(choices):
                    if parent_idx == 0:
                        child.phenotype[idx] = self.phenotype[idx]
                        child.std_deviation[idx] = self.std_deviation[idx]
                    else:
                        child.phenotype[idx] = other.phenotype[idx]
                        child.std_deviation[idx] = other.std_deviation[idx]
                        return child

def function(x_list:list[float],func: str) -> float:
    result = 0
    for xi in x_list:
        result += (eval(func))
    return result


def genetic_algo_cycle( population: list[Individual], strategy="1+1",extr = "max",func = "1+1"):
    for individual in population:
        individual.fitness= function(individual.phenotype,func)
    match strategy:
        case "1+1":

            pass
        case "m+1":
            pass
        case "m+n":
            pass
        case "m,n":
            pass

    pass;


def genetic_algorithm_init(population_size=20, strategy="1+1", gene_size=2, bounds=None, extr = "max"):
    if bounds is None:
        bounds = [-500, 500]
    population = []
    for _ in range(population_size):
        phenotype = random.sample(range(bounds[0], bounds[1]), gene_size)
        std_deviation = [1.0] * gene_size
        individual = Individual(phenotype=phenotype, std_deviation=std_deviation)
        population.append(individual)
    return population


def main():
    print(function([420.9687, 420.9687],"418.9829-xi*math.sin(math.sqrt(abs(xi)))"))
    return


if __name__ == "__main__":
    main()
