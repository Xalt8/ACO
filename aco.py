import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
from dataclasses import dataclass, field
import random

# Read in the city coordinates from file
with open('coordinates.txt', 'r') as file:
    COORDINATES = [tuple(map(int, line.strip().split(','))) for line in file]

# Read in the shortest path from file
with open('shortest_path.txt', 'r', encoding='utf8') as f:
    SHORTEST_PATH = [int(i)-1 for i in f.readline().split()]


def calculate_distance(city1_coords:tuple[int, int], city2_coords:tuple[int, int])-> float:
    """ Takes 2 cities and returns the distance between them """    
    return round(np.sqrt(np.abs(city1_coords[0] - city2_coords[0])**2 + np.abs(city1_coords[1] - city2_coords[1])**2),2)


def plot_path(coordinates:list[tuple[int, int]], path:list[int]) -> None:
    """ Plots the cities
    """
    fig = plt.subplots(figsize=(5,5))
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    tour = [coordinates[i] for i in path] + [coordinates[0]]
    x_tour = [i[0] for i in tour]
    y_tour = [i[1] for i in tour]

    plt.scatter(x=x_coords, y=y_coords, s=50, marker='o', c='lightgreen', edgecolors='forestgreen', zorder=2)
    plt.plot(x_tour, y_tour, color='red', linestyle='-', zorder=1, linewidth=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()



@dataclass
class ACO:
    city_coordinates:list[tuple[int, int]]
    shortest_path:list[int]
    num_ants:int = 10
    best_path:list[int] = None
    best_path_distance:float = np.inf
    iterations:int = 100
    alpha = 0.1 # Used in global update of pheromone
    beta = 2 # Used in scoring city
    tau = 0.0005 # initial pheromone
    q0= 0.90 # expoitation vs exploration
    rho = 0.1    # used to decay the phermone trail
    distance_graph:dict = None
    pheromone_graph:dict = None
    best_path_list:list[list[int]] = field(default_factory=list, init=False)
    pheromone_graph_list:list[dict] = field(default_factory=list, init=False)
    

    def __post_init__(self):
        if self.distance_graph == None:
            self.distance_graph  = {city1:{city2:calculate_distance(city1_coords=city1_coords, city2_coords=city2_coords) 
                                           for city2, city2_coords in enumerate(self.city_coordinates) 
                                           if city1 != city2} for city1, city1_coords in enumerate(self.city_coordinates)}
        if self.pheromone_graph == None:
            self.pheromone_graph = {city1:{city2:self.tau for city2, _ in enumerate(self.city_coordinates) 
                                           if city1 != city2} for city1, _ in enumerate(self.city_coordinates)}
        if self.pheromone_graph_list == None:
            self.pheromone_graph_list = [self.pheromone_graph]

    def score_city(self, current_city, city:int) -> float:
        """ Scores the city based on phermone and distance """
        return self.pheromone_graph[current_city][city] * (1 / self.distance_graph[current_city][city])**self.beta
        

    def choose_city(self, visited_cities:list[int]) -> int:
        unvisited_cities:list[int] = list(set(np.arange(len(self.city_coordinates))) - set(visited_cities)) 
        scores = np.array([self.score_city(current_city=visited_cities[-1], city=city) for city in unvisited_cities])
        q = random.random() 
        # Exploitation
        if q < self.q0:
            chosen_city = unvisited_cities[np.argmax(scores)]
            return chosen_city
        # Exploration
        elif q > self.q0:
            sum_array = np.sum(scores)
            prob_dist = scores/sum_array
            chosen_city = int(np.random.choice(a=unvisited_cities, size=1, p=prob_dist)[0])
            return chosen_city
        

    def local_pheromone_update(self, current_city:int, other_city:int) -> None:
        """ Updates the graph with the local pheromone update rule
            for each node visited """
        self.pheromone_graph[current_city][other_city] = (1 - self.rho) * self.pheromone_graph[current_city][other_city] + (self.rho * self.tau)


    def get_tour_from_path(self, path:list[int]) -> list[tuple[int, int]]:
        """ Takes a path of city indices and returns a list of connected city tuples including
            first and last cities """
        return [(city1, city2) for city1, city2 in zip(path[:-1], path[1:])] + [(path[0], path[-1])]


    def global_update_pheromone(self) -> None:
        """ Updates the pheromone graph with the best tour """
        tour = self.get_tour_from_path(path=self.best_path)
        for city1, city2 in tour:
            self.pheromone_graph[city1][city2] = (1 - self.alpha) * self.pheromone_graph[city1][city2] + \
                                                    self.alpha * (self.best_path_distance ** -1)
        

    def get_tour_length(self, visited_cities:list[int]) -> float:
        """ Takes a list of cities and returns the distance travelled """
        tour:list[int] = self.get_tour_from_path(path=visited_cities)
        tour_coordinates:list[tuple[int, int]] = [(self.city_coordinates[city1], self.city_coordinates[city2]) 
                                                   for city1, city2 in tour]
        distance = 0
        for city1, city2 in tour_coordinates:
            distance += calculate_distance(city1_coords=city1, city2_coords=city2)
        return round(distance,2)

        
    def optimize(self) -> None:
        for i in range(self.iterations):
            print(f"Iteration {i}, best distance: {self.best_path_distance}")
            for _ in range(self.num_ants):
                visited_cities = [0]
                while len(visited_cities) != len(self.city_coordinates):
                    chosen_city = self.choose_city(visited_cities=visited_cities)
                    self.local_pheromone_update(current_city=visited_cities[-1], other_city=chosen_city)
                    visited_cities.append(chosen_city)
                # Apply pheromone to first & last cities
                self.local_pheromone_update(current_city=visited_cities[-1], other_city=visited_cities[0])
                tour_length = self.get_tour_length(visited_cities=visited_cities)
                if tour_length < self.best_path_distance:
                    self.best_path_distance = tour_length
                    self.best_path = visited_cities
            self.global_update_pheromone()
            self.best_path_list.append(self.best_path)
            self.pheromone_graph_list.append(self.pheromone_graph)
        print(f"Shortest path distance -> {self.get_tour_length(self.shortest_path)}")
        self.animate_graph()
        


    def animate_graph(self) -> None:
        
        fig = plt.figure(figsize=(5,5))
        x_coords = [coord[0] for coord in self.city_coordinates]
        y_coords = [coord[1] for coord in self.city_coordinates]
        plt.title('Pheromone Graph - Shortest Path')
        plt.scatter(x=x_coords, y=y_coords, s=100, marker='o', c='darkorange', edgecolors='maroon', zorder=2)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_facecolor('xkcd:off white')
        pheromone_lines = []
        # subtitle_text = plt.text(x=0.5, y=0.95, s="Start", transform=plt.gca().transAxes, fontsize=12, ha='center')
        def update_data(frame) -> list:
            # Remove previous line plots
            # for line in pheromone_lines:
            #     line.remove()
            pheromone_lines.clear()
            pheromone_graph = self.pheromone_graph_list[frame]
            for start_city, neighbours in pheromone_graph.items():
                for end_city, pheromone_value in neighbours.items():
                    start_city_coords = self.city_coordinates[start_city]
                    end_city_coords = self.city_coordinates[end_city] 
                    line, = plt.plot([start_city_coords[0], end_city_coords[0]],
                                     [start_city_coords[1], end_city_coords[1]], 
                                     linewidth=pheromone_value, color='royalblue', zorder=1)
                    pheromone_lines.append(line)
            return pheromone_lines

        animation = FuncAnimation(fig=fig, func=update_data, frames=self.iterations, interval=100, repeat=False, blit=True)
        metadata = {'title':'pheromone_graph', 'artist':'Karan Singh'}
        plt.tight_layout()
        # animation.save(filename="pheromone_graph2.gif", fps=30, dpi=100, metadata=metadata)
        # print("animation saved")
        plt.show()


    def animate_graph2(self) -> None:
        
        fig = plt.figure(figsize=(5,5))
        x_coords = [coord[0] for coord in self.city_coordinates]
        y_coords = [coord[1] for coord in self.city_coordinates]
        plt.title('Pheromone Graph - Shortest Path')
        plt.scatter(x=x_coords, y=y_coords, s=100, marker='o', c='darkorange', edgecolors='maroon', zorder=3)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_facecolor('xkcd:off white')
        
        # Create Line2D objects for best path and pheromone graph
        best_path_line = []
        pheromone_lines = []


        def update_data(frame) -> list:
            # Remove previous line plots
            for line in pheromone_lines:
                line.remove()
            pheromone_lines.clear()

            for line in best_path_line:
                line.remove()
            best_path_line.clear()

            best_path:list[int] = self.best_path_list[frame]
            tour:list[tuple[int, int]] = self.get_tour_from_path(path=best_path)
            # Update best path line
            for city1, city2 in tour:
                city1_coords = self.city_coordinates[city1]
                city2_coords = self.city_coordinates[city2]
                line, = plt.plot([city1_coords[0], city2_coords[0]], 
                                    [city1_coords[1], city2_coords[1]],
                                    color='royalblue', linewidth=5, zorder=2)     
                best_path_line.append(line)

            # Pheromone lines for the pheromone graph
            pheromone_graph = self.pheromone_graph_list[frame]
            for start_city, neighbours in pheromone_graph.items():
                for end_city, pheromone_value in neighbours.items():
                    start_city_coords = self.city_coordinates[start_city]
                    end_city_coords = self.city_coordinates[end_city] 
                    line1, = plt.plot([start_city_coords[0], end_city_coords[0]],
                                     [start_city_coords[1], end_city_coords[1]],
                                      color='royalblue', linewidth=pheromone_value, zorder=1)
                    pheromone_lines.append(line1)
            
            return [best_path_line] + pheromone_lines

        animation = FuncAnimation(fig=fig, func=update_data, frames=self.iterations, interval=100, repeat=False)
        metadata = {'title':'pheromone_graph', 'artist':'Karan Singh'}
        plt.tight_layout()
        # animation.save(filename="pheromone_graph2.gif", fps=30, dpi=100, metadata=metadata)
        # print("animation saved")
        plt.show()
        



if __name__ == "__main__":

    aco = ACO(city_coordinates=COORDINATES, shortest_path=SHORTEST_PATH)
    aco.optimize()

    # plot_path(coordinates=COORDINATES, path=SHORTEST_PATH)
