import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class AntColonyAlgorithm:
    def __init__(self, graph ,ants_num, mode = "tsp", alpha = 0.1, beta = 0.1,start_node = 0,pheromone_evaporation = 0.5):
        self.graph  = graph #graph is adjacency matrix
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.pheromones_evaporation = pheromone_evaporation
        self.start_node = start_node
        self.ants_num = ants_num

        self.ants = []
        self.pheromones = [[0.1 for _ in range(len(self.graph))] for _ in range(len(self.graph))]

    def get_next_node(self, current_node, path,cyclic = True):
        neighbours = list(self.graph.neighbors(current_node))

        not_visited_neighbours = [n for n in neighbours if n not in path]

        if len(not_visited_neighbours) == 0:
            return None
        #a = self.graph[current_node][current_node+1]
        probabilities_list = [pow(self.pheromones[current_node][n], self.alpha) * pow((1 / self.graph[current_node][n]["weight"]), self.beta) for n in not_visited_neighbours]
        sum_of_probabilities = sum(probabilities_list)
        probabilities = [0]
        rand = np.random.rand()
        for(i, p) in enumerate(probabilities_list):
            probabilities.append(probabilities[i] + p / sum_of_probabilities)
            if rand < probabilities[-1]:
                next_node = not_visited_neighbours[i]



        return next_node
    def get_path_len(self,path):
        return sum([self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)])



    def update_pheromones(self,):
        for path in self.ants:
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1/self.get_path_len(path)
        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                self.pheromones[i][j] = (1 - self.pheromones_evaporation) * self.pheromones[i][j]





    def step(self,cyclic = True):
        for i in range(self.ants_num):
            path = []
            current_node =self.start_node
            path.append(current_node)

            while len(path) < len(self.graph):
                next_node = self.get_next_node(current_node, path)
                if next_node is None:
                    break
                path.append(next_node)
                current_node = next_node

                if cyclic and len(path) == len(self.graph):
                    path.append(self.start_node)
            self.ants.append(path)
    def run(self,iterations,cyclic = True):
        self.ants = []
        for i in range(iterations):
            self.step(cyclic)
            self.update_pheromones()
        best_path = self.ants[0]
        best_path_len = self.get_path_len(best_path)
        for path in self.ants:
            path_len = self.get_path_len(path)
            if path_len < best_path_len:
                best_path = path
                best_path_len = path_len
        return best_path,best_path_len


def euclidean_distance(coord1, coord2):
    return np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

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
            x, y  =  float(parts[1]), float(parts[2])
            coordinates.append((x, y))

    return coordinates

def plot_path(individual, coordinates, generation, best_distance):
    current_city = 0
    path = individual
    x = [coordinates[city][0] for city in path]
    y = [coordinates[city][1] for city in path]
    x_city = [coordinates[city][0] for city in range(len(coordinates))]
    y_city = [coordinates[city][1] for city in range(len(coordinates))]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_city, y_city )
    for i, city in enumerate(coordinates):
        plt.text(city[0], city[1], str(i+1), fontsize=12, ha='right')
    plt.plot(x, y, 'o-', label=f'Generation {generation}')

    plt.scatter(x_city[0], y_city[0], c='red', label='Start/End', zorder=5)
    for i, city in enumerate(individual):
        plt.text(coordinates[city][0], coordinates[city][1], str(city+1), fontsize=12, ha='right')

    plt.title(f'Path at Generation {generation} Best fit = {best_distance:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    cities = load_tsp("eil51.tsp")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))

    distances = [[euclidean_distance(cities[i], cities[j]) for i in range(len(cities))] for j in range(len(cities))]
    graph = nx.Graph()
    for i, row in enumerate(distances):
        for j, weight in enumerate(row):
            if weight != 0:  # Ignore zero weights (no edge)
                graph.add_edge(i, j, weight=weight)

    aco = AntColonyAlgorithm(graph,20)

    plot_path([], cities, 0, 0)
    k = 0
    while True:
        k+=1
        step_result = aco.run(10)

        plot_path(step_result[0], cities, k, step_result[1])
    print("Hello")

if __name__ == '__main__':
    main()
