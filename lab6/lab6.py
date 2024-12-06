import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random
import random

class AntColonyAlgorithm:
    def __init__(self, graph ,ants_num, mode = "tsp", alpha = 0.5, beta = 2.0,start_node = 0,pheromone_evaporation = 0.1):
        self.graph  = graph #graph is adjacency matrix
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.pheromones_evaporation = pheromone_evaporation
        self.start_node = start_node
        self.ants_num = ants_num

        self.ants = []
        self.pheromones = [[0.02 for _ in range(len(self.graph))] for _ in range(len(self.graph))]

    def get_next_node(self, current_node, path,cyclic = True):
        neighbours = list(self.graph.neighbors(current_node))

        not_visited_neighbours = [n for n in neighbours if n not in path]

        if len(not_visited_neighbours) == 0 :
            if self.mode == "tsp":
                return None
            else:
                next_node = path[-1]
                path = path[:-1]
                return next_node

        #a = self.graph[current_node][current_node+1]
        probabilities_list = [pow(self.pheromones[current_node][n], self.alpha) * pow((100 / self.graph[current_node][n]["weight"] if self.mode == "tsp" else 1), self.beta) for n in not_visited_neighbours]
        #probabilities_list = [pow(self.pheromones[current_node][n], self.alpha)  for n in not_visited_neighbours]
        sum_of_probabilities = sum(probabilities_list)
        probabilities = [0]
        rand = random.uniform(0, 1)
        for(i, p) in enumerate(probabilities_list):
            #dist = self.graph[current_node][not_visited_neighbours[i]]["weight"]

            probabilities.append(probabilities[i] + p / sum_of_probabilities)
            #print(f"""{current_node + 1}->{not_visited_neighbours[i] + 1} = {dist}""")
            if rand < probabilities[-1]:
                next_node = not_visited_neighbours[i]
                return next_node


    def get_path_len(self,path):
        return sum( [self.graph[path[i]][path[i + 1]]["weight"] if self.mode == "tsp" else 1
                     for i in range(len(path) - 1)])



    def update_pheromones(self,):
        for path in self.ants:
            for i in range(len(path) - 1):
                dist  = self.get_path_len(path)
                self.pheromones[path[i]][path[i + 1]] += 1000/dist
        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                self.pheromones[i][j] = (1 - self.pheromones_evaporation) * self.pheromones[i][j]





    def step(self,cyclic = True):

        while len(self.ants) < self.ants_num:
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
            if(len(set(path)) == len(self.graph)):
                self.ants.append(path)
    def run(self,iterations,cyclic = True, elitism = 0):

        for i in range(iterations):
            self.ants = self.ants[:elitism]
            self.step(cyclic)
            self.update_pheromones()
            self.ants = sorted(self.ants, key = self.get_path_len,reverse= not cyclic)

            print(f"Gen: {i}, Len: {self.get_path_len(self.ants[0])}")


        best_path = self.ants[0]
        best_path_len = self.get_path_len(best_path)
        best_i = 0;
        for i,path in enumerate(self.ants):
            path_len = self.get_path_len(path)
            if path_len < best_path_len:
                best_path = path
                best_path_len = path_len
                best_i = i

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

def plot_path(individual, coordinates, generation,edges = None):
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

    if edges is not None:
        for edge in edges:
            plt.plot([coordinates[edge[0]][0],coordinates[edge[1]][0]],[coordinates[edge[0]][1],coordinates[edge[1]][1]], 'k-', alpha = 0.5)
    for i, city in enumerate(individual):
        plt.text(coordinates[city][0], coordinates[city][1], str(city+1), fontsize=12, ha='right')
    dist = sum(euclidean_distance(coordinates[individual[i]], coordinates[individual[i + 1]]) for i in range(len(individual) - 1))
    plt.title(f'Path at Generation {generation} Best fit = {dist:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


optimal_path = [
    0, 21, 7, 25, 30, 27, 2, 35, 34, 19,
     1, 28, 20, 15, 49, 33, 29, 8, 48, 9,
     38, 32, 44, 14, 43, 41, 39, 18, 40, 12,
     24, 13, 23, 42, 6, 22, 47, 5, 26, 50,
     45, 11, 46, 17, 3, 16, 36, 4, 37, 10,
     31,0
]

def main():
    variant = int(input("Variant: \n1) 51 cities TSP\n 2) Graph gamilton\n"))
    cities = None
    graph = nx.Graph()
    edges = None
    mode = "tsp"
    cyclic = True
    if variant== 1:
        cities = load_tsp("eil51.tsp")
        graph = nx.Graph()
        graph.add_nodes_from(range(len(cities)))

        distances = [[euclidean_distance(cities[i], cities[j]) for i in range(len(cities))] for j in range(len(cities))]

        for i, row in enumerate(distances):
            for j, weight in enumerate(row):
                if weight != 0:  # Ignore zero weights (no edge)
                    graph.add_edge(i, j, weight=weight)
    elif variant == 2:

        edges = [
            (0,1),(0,7),(0,4),
            (1,2),(1,9),
            (2,3),(2,11),
            (3,4),(3,13),
            (4,5),
            (5,6),(5,14),
            (6,7),(6,16),
            (7,8),
            (8,9),(8,17),
            (9,10),
            (10,11),(10,18),
            (11,12),
            (12,13),(12,19),
            (13,14),
            (14,15),
            (15,16),(15,19),
            (16,17),
            (17,18),
            (18,19)

        ]
        graph.add_edges_from(edges)
        pos = nx.shell_layout(graph,nlist = [range(15,20),range(5,15),range(0,5)])
        cities = list(pos.values())

        mode = "other"
        cyclic  = False
    else:
        exit(0)
    aco = AntColonyAlgorithm(graph,20,alpha=1.5,beta = 3,pheromone_evaporation=0.1,mode = mode)
    plot_path([], cities, 0,edges)
    k = 20

    k+=1
    step_result = aco.run(k,cyclic=cyclic)

    plot_path(step_result[0], cities, k,edges)
    print("Hello")

    #plot_path(optimal_path, cities, "Optimal",edges)





if __name__ == '__main__':
    main()
