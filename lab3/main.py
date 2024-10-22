import random
import matplotlib.pyplot as plt
import matplotlib
import math
matplotlib.use('TkAgg')
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
# Генерация случайного пути (индивида)
def convert_to_adjacency_representation(route):
    size = len(route)
    adjacency = [0] * size
    for i in range(size):
        adjacency[route[i]] = route[(i + 1) % size]  # Замыкаем цикл
    return adjacency
def generate_individual_path(cities):
    individual = cities[:]
    random.shuffle(individual)
    return individual
def generate_individual(cities):
    normal_individual = generate_individual_path(cities)
    return convert_to_adjacency_representation(normal_individual)

def fitness(individual, distances):
    total_distance = 0
    curren_city = 0
    next_city = 0
    for _ in range(len(individual)):
        current_city = next_city
        next_city = individual[current_city]  # Город, в который идем из текущего города
        total_distance += distances[current_city][next_city]
    return total_distance


# Кроссинговер (Alternating Edges)
def crossover_alternating_edges(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:]
    n = len(parent1)
    offspring = [-1] * n  # Инициализация потомка
    used = set()  # Множество посещённых городов

    # Стартовый город
    current_city = parent1[0]
    offspring[0] = current_city
    used.add(current_city)
    used.add(0)

    for i in range(1, n-1):
        if i % 2 == 1:
            # Извлекаем следующее ребро из parent2
            next_city = parent2[current_city]
        else:
            # Извлекаем следующее ребро из parent1
            next_city = parent1[current_city]

        # Если город уже был использован, выбираем случайный доступный город
        if next_city in used:
            next_city = random.choice([city for city in parent1 if city not in used])

        offspring[current_city] = next_city
        used.add(next_city)
        current_city = next_city
    offspring[offspring.index(-1)]=0
    return offspring


# Функция для поиска следующего города
def get_next_city(parent, current_city, child):
    # Найти следующий город в родителе, который еще не был добавлен в child
    idx = parent.index(current_city)
    next_city = parent[(idx + 1) % len(parent)]

    if next_city in child:  # Если следующий город уже присутствует в child, это будет цикл
        available_cities = [city for city in parent if city not in child]
        return random.choice(available_cities) if available_cities else -1

    return next_city

# Кроссинговер "Subtour Chunks" (куски подтуров)
def crossover_subtour_chunks(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        return parent1[:]

    size = len(parent1)
    child = [-1] * size

    # Определение случайного куска (подтура)
    start, end = sorted(random.sample(range(size), 2))
    subtour = parent1[start:end]

    # Вставляем этот кусок в ребенка
    child[start:end] = subtour

    # Заполняем оставшиеся города из второго родителя
    pointer = end
    for city in parent2:
        if city not in subtour:
            if pointer >= size:
                pointer = 0
            child[pointer] = city
            pointer += 1

    return child


def crossover_heuristic(parent1, parent2, distances,crossover_rate):
    """
    Эвристический кроссинговер для задачи коммивояжера.

    :param parent1: Список городов первого родителя (порядок посещения городов)
    :param parent2: Список городов второго родителя (порядок посещения городов)
    :param distances: Матрица расстояний между городами
    :return: Потомок (список городов)
    """
    if random.random() < crossover_rate:
        return parent1[:]
    n = len(parent1)
    used = set()  # Множество посещённых городов

    # Инициализация потомка с пустыми значениями
    offspring = [-1] * n


    current_city = parent1[0]
    offspring[0] = current_city
    used.add(current_city)
    used.add(0)

    for i in range(1, n-1):
        # Определяем соседние города для каждого родителя
        next_city1 = parent1[current_city]
        next_city2 = parent2[current_city]

        # Сравниваем длины рёбер и выбираем город с более коротким ребром
        if distances[current_city][next_city1] < distances[current_city][next_city2]:
            next_city = next_city1
        else:
            next_city = next_city2

        # Если город уже был посещен, выбираем случайный непосещённый город
        while next_city in used:
            next_city = random.choice([city for city in parent1 if city not in used])

        # Добавляем выбранный город в потомка
        offspring[current_city] = next_city
        used.add(next_city)
        # Обновляем текущий город
        current_city = next_city
    offspring[offspring.index(-1)] = 0
    return offspring

# Мутация с проверкой на циклы
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        size = len(individual)
        idx1, idx2 = random.sample(range(size), 2)

        # Меняем города местами, но проверяем корректность
        new_individual = individual[:]
        new_individual[idx1], new_individual[idx2] = new_individual[idx2], new_individual[idx1]
        # Проверяем, не создали ли мы цикл
        while not has_no_cycle(new_individual):
            idx1, idx2 = random.sample(range(size), 2)

            # Меняем города местами, но проверяем корректность
            new_individual = individual[:]
            new_individual[idx1], new_individual[idx2] = new_individual[idx2], new_individual[idx1]
        return new_individual
    return individual  # Иначе возвращаем исходную особь

# Функция для проверки наличия цикла
def has_no_cycle(individual):
    visited = set()
    current = 0  # Начинаем с первого города
    while current not in visited:
        visited.add(current)
        current = individual[current]  # Переходим в следующий город

    # Проверяем, что все города были посещены и не возникло лишнего цикла
    return len(visited) == len(individual)


# Селекция (турнирный отбор)
def selection(population, fitnesses):
    tournament = random.sample(list(zip(population, fitnesses)), 5)
    return min(tournament, key=lambda x: x[1])[0]


# Визуализация пути
def plot_path(individual, coordinates, generation):
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
        plt.text(coordinates[city][0], coordinates[city][1], str(city), fontsize=12, ha='right')

    plt.title(f'Path at Generation {generation}')
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
# Генетический алгоритм с элитарностью
def genetic_algorithm(cities, distances, coordinates, pop_size=100, generations=500,
                      crossover_rate=0.8, mutation_rate=0.05, elitism=True, elite_size=1, crossover_method="heuristic",population = None):
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
        for _ in range(pop_size):
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
            new_population.append(child)
        new_population_sorted = sorted(new_population, key=lambda ind: fitness(ind, distances))

        population = new_population_sorted[0:pop_size]
        fitnesses = [fitness(ind, distances) for ind in population]
        # Отслеживание лучшего индивида
        best_fitness = min(fitnesses)
        best_individual = min(population, key=lambda ind: fitness(ind, distances))
        print(f"Лучший путь: {best_individual}, длина пути: {best_fitness}")


    # Возвращаем лучшего индивида за все поколения
    best_individual = min(population, key=lambda ind: fitness(ind, distances))
    return best_individual, fitness(best_individual, distances),population


# Функция для выполнения генетического алгоритма с заданным числом шагов
def run_genetic_algorithm_steps():
    #coordinates = load_tsp("eil51.tsp")
    #cities = list(range(len(coordinates)))
    cities = list(range(10))
    coordinates = {i: (random.randint(0, 100), random.randint(0, 100)) for i in cities}
    distances = [[euclidean_distance(coordinates[i], coordinates[j]) for j in cities] for i in cities]
    population = None
    while True:
        # Ввод числа шагов от пользователя
        generations = int(input("Введите число шагов (итераций) для выполнения (или 0 для выхода): "))

        if generations == 0:
            print("Выход из программы...")
            break

        # Определение городов и расстояний


        # Запуск генетического алгоритма
        best_solution, best_distance,population = genetic_algorithm(
            cities, distances, coordinates, generations=generations,crossover_rate=0.7,mutation_rate=0.0,population=population
        )
        print(f"Лучший путь: {best_solution}, длина пути: {best_distance}")
        # Визуализация и вывод результата
        plot_path(best_solution, coordinates, generations)



# Меню
def menu():
    while True:
        print("\n=== Меню ===")
        print("1. Запустить генетический алгоритм с заданным числом шагов")
        print("2. Выйти")
        choice = input("Выберите пункт меню: ")

        if choice == '1':
            run_genetic_algorithm_steps()
        elif choice == '2':
            print("Выход...")
            break
        else:
            print("Неверный ввод. Попробуйте снова.")


# Запуск меню
menu()
