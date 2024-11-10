import random
import matplotlib.pyplot as plt
import matplotlib
import math

import numpy as np

matplotlib.use('TkAgg')
path = [
    0, 21, 7, 25, 30, 27, 2, 35, 34, 19,
     1, 28, 20, 15, 49, 33, 29, 8, 48, 9,
     38, 32, 44, 14, 43, 41, 39, 18, 40, 12,
     24, 13, 23, 42, 6, 22, 47, 5, 26, 50,
     45, 11, 46, 17, 3, 16, 36, 4, 37, 10,
     31
]
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
# Генерация случайного пути (индивида)
def neighbor_to_path(neighbor_representation):
    size = len(neighbor_representation)
    path = [0]
    current_city = 0
    for _ in range(1, size):
        next_city = neighbor_representation[current_city]
        path.append(next_city)
        current_city = next_city
    return path

def path_to_neighbor(path):
    size = len(path)
    neighbor_representation = [0] * size
    for i in range(size - 1):
        neighbor_representation[path[i]] = path[i + 1]
    neighbor_representation[path[-1]] = path[0]
    return neighbor_representation
def generate_individual_path(cities):
    individual = cities[:]
    random.shuffle(individual)
    return individual
def generate_individual(cities):
    normal_individual = generate_individual_path(cities)
    return path_to_neighbor(normal_individual)

def fitness(individual, distances):
    total_distance = 0
    curren_city = 0
    next_city = 0
    for _ in range(len(individual)):
        current_city = next_city
        next_city = individual[current_city]
        total_distance += distances[current_city][next_city]
    return total_distance


# Кроссинговер (Alternating Edges)
def crossover_alternating_edges(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:]
    n = len(parent1)
    offspring = [-1] * n
    used = set()

    current_city = parent1[0]
    offspring[0] = current_city
    used.add(current_city)
    used.add(0)

    for i in range(1, n-1):
        if i % 2 == 1:

            next_city = parent2[current_city]
        else:

            next_city = parent1[current_city]

        if next_city in used:
            next_city = random.choice([city for city in parent1 if city not in used])

        offspring[current_city] = next_city
        used.add(next_city)
        current_city = next_city
    offspring[offspring.index(-1)]=0
    return offspring


def get_next_city(parent, current_city, child):
    idx = parent.index(current_city)
    next_city = parent[(idx + 1) % len(parent)]

    if next_city in child:
        available_cities = [city for city in parent if city not in child]
        return random.choice(available_cities) if available_cities else -1

    return next_city

def crossover_subtour_chunks(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        return parent1[:]

    size = len(parent1)
    child = [-1] * size

    start, end = sorted(random.sample(range(size), 2))
    subtour = parent1[start:end]

    child[start:end] = subtour

    pointer = end
    for city in parent2:
        if city not in subtour:
            if pointer >= size:
                pointer = 0
            child[pointer] = city
            pointer += 1

    return child


def crossover_heuristic(parent1, parent2, distances,crossover_rate):

    if random.random() < crossover_rate:
        return parent1[:]
    n = len(parent1)
    used = set()

    offspring = [-1] * n


    current_city = parent1[0]
    offspring[0] = current_city
    used.add(current_city)
    used.add(0)

    for i in range(1, n-1):
        next_city1 = parent1[current_city]
        next_city2 = parent2[current_city]

        if distances[current_city][next_city1] < distances[current_city][next_city2]:
            next_city = next_city1
        else:
            next_city = next_city2

        while next_city in used:
            next_city = random.choice([city for city in parent1 if city not in used])


        offspring[current_city] = next_city
        used.add(next_city)

        current_city = next_city
    offspring[offspring.index(-1)] = 0
    return offspring



def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        path = neighbor_to_path(individual)

        idx1, idx2 = random.sample(range(len(path)), 2)

        path[idx1], path[idx2] = path[idx2], path[idx1]

        individual = path_to_neighbor(path)
    return individual



def selection(population, fitnesses):
    tournament = random.sample(list(zip(population, fitnesses)), 3)
    return min(tournament, key=lambda x: x[1])[0]



def plot_path(individual, coordinates, generation, best_distance):
    current_city = 0
    path = [current_city]
    for _ in range(len(individual) - 1):
        next_city = individual[current_city]
        path.append(next_city)
        current_city = next_city
    x = [coordinates[city][0] for city in path] + [coordinates[path[0]][0]]
    y = [coordinates[city][1] for city in path] + [coordinates[path[0]][1]]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o-', label=f'Generation {generation}')
    plt.scatter(x[0], y[0], c='red', label='Start/End', zorder=5)
    for i, city in enumerate(individual):
        plt.text(coordinates[city][0], coordinates[city][1], str(city+1), fontsize=12, ha='right')

    plt.title(f'Path at Generation {generation} Best fit = {best_distance:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_tsp(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    coordinates = []
    reading_coords = False

    for line in lines:
        if "NODE_COORD_SECTION" in line:
            reading_coords = True
            continue
        if "EOF" in line:
            break
        if reading_coords:
            parts = line.split()
            x, y = float(parts[1]), float(parts[2])
            coordinates.append((x, y))

    return coordinates

def genetic_algorithm(cities, distances, coordinates, pop_size=100, generations=500,
                      crossover_rate=0.8, mutation_rate=0.05, elitism=True, elite_size=10, crossover_method="heuristic",population = None):
    # Инициализация популяции
    if population is None:
        population = [generate_individual(cities) for _ in range(pop_size)]
    for generation in range(generations):
        fitnesses = [fitness(ind, distances) for ind in population]
        # Сортируем популяцию по фитнесу (наименьший фитнес — лучший путь)
        population_sorted = sorted(population, key=lambda ind: fitness(ind, distances))

        new_population = []
        fitnesses_sorted = [fitness(ind, distances) for ind in population_sorted]
        # Элитарность: сохраняем лучших индивидов (по умолчанию 1)
        if elitism:
            new_population.extend(population_sorted[:elite_size])

        # Эволюция нового поколения для оставшихся индивидов
        while len(new_population) < pop_size:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            if crossover_method == "alternating":
                child = crossover_alternating_edges(parent1, parent2, crossover_rate)
            elif crossover_method == "subtour":
                child = crossover_subtour_chunks(parent1, parent2, crossover_rate)
            elif crossover_method == "heuristic":
                child = crossover_heuristic(parent1, parent2, distances, crossover_rate)
            child_safe = child
            child = mutate(child, mutation_rate)
            if child[0]==0:
                child = child_safe
            new_population.append(child)
            # Преобразование списка списков в множество кортежей для удаления дубликатов
            new_population = list(set(tuple(individual) for individual in new_population))

            # Преобразование обратно в список списков
            new_population = [list(individual) for individual in new_population]
        new_population_sorted = sorted(new_population, key=lambda ind: fitness(ind, distances))
        population = new_population_sorted
        fitnesses = [fitness(ind, distances) for ind in population]
        best_fitness = min(fitnesses)
        best_individual = min(population, key=lambda ind: fitness(ind, distances))
        #print(f"BP: {best_individual}, BF: {best_fitness}")



    best_individual = min(population, key=lambda ind: fitness(ind, distances))
    return best_individual, fitness(best_individual, distances),population
def run_genetic_algorithm_with_varying_parameters():
    coordinates = load_tsp("eil51.tsp")
    cities = list(range(len(coordinates)))
    distances = [[euclidean_distance(coordinates[i], coordinates[j]) for j in cities] for i in cities]

    crossover_methods = ["alternating", "heuristic"]
    crossover_rates = [i * 0.3 for i in range(1, 2)]  # От 0.1 до 1.0
    mutation_rates = [i * 0.1 for i in range(0, 2)]  # От 0.05 до 0.25

    steps_data = []

    for crossover_method in crossover_methods:
        print(f"Тестирование метода кроссинговера: {crossover_method}")

        for crossover_rate in crossover_rates:
            for mutation_rate in mutation_rates:
                print(f"  Кроссинговер с вероятностью {crossover_rate:.2f} и мутация с вероятностью {mutation_rate:.2f}")
                best_fitness = float('inf')
                generation = 0
                population = None

                while best_fitness > 460 and generation < 3000:
                    generation += 1

                    best_individual, best_fitness, population = genetic_algorithm(
                        cities, distances, coordinates, pop_size=100, generations=1,
                        crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                        elitism=True, elite_size=10, crossover_method=crossover_method,
                        population=population
                    )
                # Записываем количество шагов (итераций)
                steps_data.append((crossover_method, crossover_rate, mutation_rate,generation))

    # После выполнения всех тестов строим график
    steps_data = np.array(steps_data)
    print(steps_data)
    # Создаем сетку для подграфиков
    fig, axs = plt.subplots(len(crossover_methods), 1, figsize=(10, 15))

    for idx, crossover_method in enumerate(crossover_methods):
        ax = axs[idx]
        method_data = steps_data[steps_data[:, 0] == crossover_method]

        # Разделяем по вероятности кроссинговера и мутации
        for mutation_rate in mutation_rates:
            subset = method_data[method_data[:, 2] == mutation_rate]
            ax.plot(subset[:, 1], subset[:, 3], label=f"Mutation rate = {mutation_rate:.2f}")

        ax.set_title(f"{crossover_method.capitalize()} Crossover")
        ax.set_xlabel("Crossover Rate")
        ax.set_ylabel("Number of Steps")
        ax.legend()

    plt.tight_layout()
    plt.show()

def run_genetic_algorithm_steps():
    coordinates = load_tsp("eil51.tsp")
    cities = list(range(len(coordinates)))
    #cities = list(range(20))
    #coordinates = {i: (random.randint(0, 100), random.randint(0, 100)) for i in cities}
    distances = [[euclidean_distance(coordinates[i], coordinates[j]) for j in cities] for i in cities]
    population = None
    while True:


        generations = int(input("Введите число шагов (итераций) для выполнения (или 0 для выхода): "))

        if generations == 0:
            print("Выход из программы...")
            break


        best_solution, best_distance, population = genetic_algorithm(
            cities, distances, coordinates, generations=generations,crossover_rate=0.6,mutation_rate=0.2,population=population
        )
        print(f"Лучший путь: {best_solution}, длина пути: {best_distance}")

        plot_path(best_solution, coordinates, generations, best_distance)




def menu():
    while True:
        print("\n=== Меню ===")
        print("1. Запустить генетический алгоритм с заданным числом шагов")
        print("2. Исследовать")
        print("3. Выйти")
        choice = input("Выберите пункт меню: ")

        if choice == '1':
            run_genetic_algorithm_steps()
        elif choice == "2":
            run_genetic_algorithm_with_varying_parameters()
        elif choice == '3':
            print("Выход...")
            break
        else:
            print("Неверный ввод. Попробуйте снова.")



menu()
