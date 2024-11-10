import random
import math


# Функции для представления операций
def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y if y != 0 else 1  # Чтобы избежать деления на ноль


def abs_func(x,y=None):
    return abs(x)


def sin_func(x,y=None):
    return math.sin(x)


def cos_func(x,y=None):
    return math.cos(x)


def exp_func(x,y=None):
    return math.exp(x)


def power(x, y):
    return x ** y


# Типы узлов
FUNCTIONS = [add, sub, mul, div, abs_func, sin_func, cos_func, exp_func, power]
TERMINALS = ['x1', 'x2', 'x3', 'x4', 'x5', 1, 2]  # Переменные и константы


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value  # Это будет либо функция, либо терминал
        self.left = left
        self.right = right  # для бинарных операторов

    def evaluate(self, variables):
        """Рекурсивная функция для вычисления значения дерева"""
        if self.value in TERMINALS:
            if isinstance(self.value, str):
                return variables[self.value]  # возвращаем значение переменной
            return self.value  # возвращаем константу
        else:
            # Применяем функцию на основе значения
            left = self.left.evaluate(variables) if self.left is not None else None
            right = self.right.evaluate(variables) if self.right is not None else None
            return self.value(left, right)


class Tree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def create(self, grow=True):
        self.root = self._create_tree(0, self.max_depth, grow)

    def get_random_node(self):
        total_nodes = self._count_nodes(self.root)
        random_index = random.randint(0, total_nodes - 1)
        print("R",total_nodes,random_index)
        return self._get_random_node(self.root, random_index)

    def evaluate(self, variables):
        return self.root.evaluate(variables)
    def _create_tree(self, depth, max_depth,grow= False):
        """Рекурсивно создаем дерево с максимальной глубиной max_depth"""
        if depth == max_depth:
            # Возвращаем терминал
            value = random.choice(TERMINALS)
            return Node(value)
        else:
            if grow:
                node_is_terminal = random.random()
                if node_is_terminal < 0.4:
                    value = random.choice(TERMINALS)
                    return Node(value)
            func = random.choice(FUNCTIONS)
            if func in [add, sub, mul, div, power]:
                # Двухаргументные функции
                left = self._create_tree(depth + 1, max_depth,grow)
                right = self._create_tree(depth + 1, max_depth,grow)
                return Node(func, left, right)
            else:
                # Одноаргументные функции
                left = self._create_tree(depth + 1, max_depth,grow)
                return Node(func, left)

    def _count_nodes(self, node: Node):
        # Рекурсивный подсчёт узлов в поддереве
        if node is None:
            return 0
        left_size = self._count_nodes(node.left)
        right_size = self._count_nodes(node.right)
        return 1 + left_size + right_size

    def _get_random_node(self, node, index):
        # Рекурсивный поиск случайного узла с данным индексом
        if node is None:
            return None
        left_size = self._count_nodes(node.left)

        if index == left_size:  # Мы нашли нужную вершину
            return node
        elif index < left_size:  # Ищем в левом поддереве
            return self._get_random_node(node.left, index)
        else:  # Ищем в правом поддереве, корректируем индекс
            return self._get_random_node(node.right, index - left_size - 1)

    def print(self):
        self._print(self.root)

    def _print(self, node: Node, depth=0):
        children = self._count_nodes(node)
        print(depth*"\t",node,":", children, "-", depth)
        if node.left is not None:
            self._print(node.left, depth + 1)
        if node.right is not None:
            self._print(node.right, depth + 1)


def initialize_population(pop_size, max_depth) -> list[Tree]:
    """Создаем популяцию деревьев"""
    population = []
    for _ in range(pop_size):
        grow = random.choice([True, False])
        size = random.choice(range(1, max_depth, 1))
        tree = Tree(size)
        tree.create(grow)
        population.append(tree)
    return population


def fitness_function(tree, target_function, variables):
    """Вычисляем фитнес для дерева, сравнивая с целевой функцией"""
    predicted = 0
    predicted += tree.evaluate(variables)
    return abs(predicted - target_function(variables))


def node_mutation(node, terminal_set, function_set):
    """
    Узловая мутация: заменяем случайный узел на новый узел того же типа.
    Если узел терминальный, заменяем на другой терминал из множества терминалов.
    Если узел функциональный, заменяем на случайный функциональный узел.
    """
    if node.left is None and node.right is None:
        new_terminal = random.choice(terminal_set)
        return new_terminal
    # Если узел функциональный (имеет поддеревья), заменяем его на случайный функциональный узел
    else:
        new_function = random.choice(function_set)
        new_node = new_function()
        new_node.left = node.left
        new_node.right = node.right
        return new_node


def pruning_mutation(node, terminal_set):
    """
    Усекающая мутация: заменяем поддерево на новый терминал.
    """
    # Если узел терминальный, то не делаем мутацию
    if node.left is None and node.right is None:
        return node

    # Заменяем поддерево (все дочерние узлы) на новый терминал
    new_terminal = random.choice(terminal_set)
    return new_terminal


def grow_mutation(tree):
    """Растущая мутация"""
    pass


def get_random_subtree(node):
    """Рекурсивно находим случайное поддерево, включая как внутренние узлы, так и листья."""
    # Если это терминальный узел (листье), возвращаем его
    if node.left is None and node.right is None:
        return node

    # С случайной вероятностью решаем: идти дальше в поддеревья или вернуть текущий узел
    if random.random() < 0.5:  # 50% вероятности, что мы вернем текущий узел
        return node

    # Иначе рекурсивно спускаемся в одно из поддеревьев (левое или правое)
    if random.random() < 0.5:  # 50% вероятности, что идем в левое поддерево
        if node.left:
            return get_random_subtree(node.left)
    else:  # Идем в правое поддерево
        if node.right:
            return get_random_subtree(node.right)

    # Если поддеревья отсутствуют, возвращаем текущий узел
    return node


def is_compatible(node1, node2):
    """Проверяем совместимость двух поддеревьев (по типу узлов)."""
    # Проверка на бинарные узлы
    if (node1.left is not None and node1.right is not None) and (node2.left is not None and node2.right is not None):
        return True  # Оба бинарные узлы
    # Проверка на унарные узлы (с одним дочерним узлом)
    if (node1.left is None and node1.right is None) and (node2.left is None and node2.right is None):
        return True  # Оба терминальные узлы
    # Проверка на унарные узлы
    if (node1.left is None and node1.right is not None) and (node2.left is None and node2.right is not None):
        return True  # Оба унарные узлы
    return False  # Узлы несовместимы


def get_tree_size(node):
    """Вычисляем размер дерева (количество узлов)."""
    if node is None:
        return 0
    return 1 + get_tree_size(node.left) + get_tree_size(node.right)


def replace_subtree(node, old_subtree, new_subtree):
    """Заменяем поддерево в исходном дереве."""
    if node == old_subtree:
        return new_subtree
    if node.left:
        node.left = replace_subtree(node.left, old_subtree, new_subtree)
    if node.right:
        node.right = replace_subtree(node.right, old_subtree, new_subtree)
    return node


def subtree_crossover(parent1, parent2, max_size):
    """
    Оператор кроссинговера поддеревьев между двумя деревьями (родителями).

    parent1, parent2 — деревья, которые участвуют в кроссинговере.
    max_size — максимальный размер дерева (максимальная сложность),
    который потомки могут иметь.
    """
    # Шаг 1: Выбираем случайные поддеревья в обоих родительских деревьях
    subtree1 = get_random_subtree(parent1)
    subtree2 = get_random_subtree(parent2)

    # Шаг 2: Проверяем совместимость поддеревьев
    if not is_compatible(subtree1, subtree2):
        # Если поддеревья несовместимы, ищем другое поддерево во втором родителе
        subtree2 = get_random_subtree(parent2)
        if not is_compatible(subtree1, subtree2):
            return parent1, parent2  # Если поддеревья все равно несовместимы, возвращаем без изменений

    # Шаг 3: Обмениваем поддеревья
    new_parent1 = replace_subtree(parent1, subtree1, subtree2)
    new_parent2 = replace_subtree(parent2, subtree2, subtree1)

    # Шаг 4: Проверяем размер потомков после кроссинговера
    if get_tree_size(new_parent1) <= max_size:
        parent1 = new_parent1
    if get_tree_size(new_parent2) <= max_size:
        parent2 = new_parent2
    return parent1, parent2


def selection(population: list[Tree], fitness: list[float], k=3) -> list[Tree]:
    candidates_indices = random.choices(range(len(population)), k=k)


def crossover(parent1, parent2, max_size):
    """Оператор кроссинговера поддеревьев"""

    new_parent1, new_parent2 = subtree_crossover(parent1, parent2, max_size)
    return new_parent1, new_parent2


def mutation(tree, terminal_set, function_set, max_size):
    node = tree.get_random_node()

    """Оператор мутации"""
    mutation_type = random.choice(['node', 'pruning', 'growing'])  # Выбор типа мутации
    if mutation_type == 'node':
        return node_mutation(node, terminal_set, function_set)
    elif mutation_type == 'pruning':
        return pruning_mutation(node, terminal_set)
    elif mutation_type == 'growing':
        return grow_mutation(node, terminal_set, function_set, tree.max_depth)
    return tree


def genetic_algorithm(pop_size, max_depth, generations, target_function, n):
    population = initialize_population(pop_size, max_depth)
    best_tree = None
    best_fitness = float('inf')

    for generation in range(generations):
        # Оценка фитнеса всех особей
        fitnesses = [fitness_function(tree, target_function, n, {}) for tree in population]

        # Выбираем лучших
        best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            best_tree = population[best_idx]

        # Селекция, мутация и кроссинговер
        new_population = []
        for tree in population:
            parent1, parent2 = random.choices(population, k=2)
            offspring1, offspring2 = subtree_crossover(parent1, parent2, tree.max_depth)
            offspring1 = mutation(offspring1, max_depth, FUNCTIONS, tree.max_depth)
            offspring2 = mutation(offspring2, max_depth, FUNCTIONS, tree.max_depth)
            new_population.append(offspring1)
            new_population.append(offspring2)

        population = new_population

    return best_tree


def main():
    print("Hello")


if __name__ == "__main__":
    main()
    tree = Tree(max_depth=2)
    tree.create(grow=False)
    tree.print()
    print("!")
    print(tree.get_random_node())
