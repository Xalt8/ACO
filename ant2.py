from dataclasses import dataclass, field
import numpy as np
from city import City, CITIES, SHORTEST_PATH, get_city
import random
from sa1 import calculate_distance, get_tour_length, plot

# ====================
# TYPES
# ====================
cityName = str
graph = dict[cityName, dict[cityName, dict[str, float]]]


@dataclass
class Ant:
    graph:graph
    visited_nodes: list[cityName] = field(default_factory=list, init=False)
    current_node:cityName = None  
    beta = 2     # Importance of heuristics
    q0 = 0.9     # Higher number leads to more exploitation 
    rho = 0.1    # used to decay the phermone trail
    tau = 0.0005 # initial pheromone

    
    def __post_init__(self):
        if self.current_node == None:
            self.current_node = self.choose_start_node()
            self.visited_nodes.append(self.current_node)

    
    def choose_start_node(self) -> cityName:
        ''' Randomly chooses a city'''
        return random.choice(list(self.graph.keys()))


    def get_unvisited_nodes(self) -> list[cityName]:
        ''' Returns a list of unvisited nodes'''
        return [node for node in self.graph.keys() if node not in self.visited_nodes]


    def score_node(self, node:cityName) -> float:
        ''' Scores the node based on phermone and distance'''
        return self.graph[self.current_node][node]['pheromone'] * \
                    (1/self.graph[self.current_node][node]['distance'])**self.beta    


    def choose_node(self, unvisited_nodes:list) -> int:
        ''' Chooses the next node to visit
        ================================================
        q: random value between 0,1
        q0: determines whether to explore or exploit
        score_array: applies the score_node function to all unvisited 
                     nodes and returns an array of results 
        Exploitation: chooses node based on the highest score from all unvisited
        Exploration: randomly chooses a node based on a probablity distribution 
        '''
        q = random.random() 
        score_array = np.array([self.score_node(node) for node in unvisited_nodes])

        # Exploitation
        if q < self.q0:
            node = unvisited_nodes[np.argmax(score_array)]
                
        # Exploration
        elif q > self.q0:
            sum_array = np.sum(score_array)
            prob_dist = score_array/sum_array
            node = int(np.random.choice(a=unvisited_nodes, size=1, p=prob_dist))
            
        return str(node)


    def local_phermone_update(self, node:cityName) -> None:
        ''' Updates the graph with the local pheromone update rule
            for each node visited'''
        self.graph[self.current_node][node]['pheromone'] = \
            (1 - self.rho) * self.graph[self.current_node][node]['pheromone'] + (self.rho * self.tau)


    def visit_node(self, node:cityName) -> None:
        ''' Adds the node to the visited_nodes list'''
        self.current_node = node
        self.visited_nodes.append(node)
        # self.tour.append(node)
        

    def run(self) -> list[City]:
        ''' While there are unvisited nodes choose a node
            apply local phermone when visiting node and update 
            visited nodes list'''
        
        while self.get_unvisited_nodes():
            unvisited_nodes = self.get_unvisited_nodes()
            chosen_node = self.choose_node(unvisited_nodes)
            self.local_phermone_update(chosen_node)
            self.visit_node(chosen_node)
        # Close the loop -> connect the first & last cities
        first_node = self.visited_nodes[0]
        self.local_phermone_update(first_node)

        return self.visited_nodes


@dataclass
class AntColony:

    city_list:list[City]
    graph:graph = None
    num_ants:int = 10
    best_path:list[City] = None
    best_path_distance:float = np.inf
    alpha = 0.1
    iterations:int = 500
    tau = 0.0005 # initial pheromone

    def __post_init__(self):
        if self.graph == None:
            self.graph = self.create_graph()


    def create_graph(self) -> graph:
        ''' Takes a list of City objects and creates
            a fully connected graph with 0 as initial phermone level 
            and calculated disctance '''
        graph = dict()
        for city1 in self.city_list:
            graph[city1.name] = {city2.name:
                {'pheromone':self.tau, 
                'distance':calculate_distance(city1, city2)}
                    for city2 in self.city_list if city2!=city1}
        return graph


    def spawn_ants(self) -> list[Ant]:
        ''' Creates a list of Ant objects'''
        self.ants = [Ant(self.graph) for _ in range(self.num_ants)]


    def set_global_best(self) -> None:
        ''' For all ants -> makes a tour, calculate the distance of tour 
            and if the distance is better than the gbest tour it updates the gbest tour'''
        for ant in self.ants:
            tour = ant.run()
            cities = [get_city(city, CITIES) for city in tour]
            tour_distance = get_tour_length(cities)
            if tour_distance < self.best_path_distance:
                self.best_path_distance = tour_distance
                self.best_path = tour
    

    def global_update_pheromone(self) -> None:
        ''' Takes a pair of cities from the best_path and updates the
            pheromone on the graph '''

        for city1, city2 in zip(self.best_path[:-1], self.best_path[1:]):
            self.graph[city1][city2]['pheromone'] = \
                (1 - self.alpha) * self.graph[city1][city2]['pheromone'] + \
                    self.alpha * (self.best_path_distance ** -1)
        # Close the loop -> connect the first & last cities
        city1, city2 = self.best_path[-1], self.best_path[0]
        self.graph[city1][city2]['pheromone'] = \
                (1 - self.alpha) * self.graph[city1][city2]['pheromone'] + \
                    self.alpha * (self.best_path_distance ** -1)


    def optimise(self) -> None:
        ''' Main optimising loop'''

        for i in range(self.iterations):

            self.spawn_ants()
            self.set_global_best()
            self.global_update_pheromone()

            print(f'{i} best distance:{self.best_path_distance}')


if __name__ == '__main__':

    aco = AntColony(city_list=CITIES)
    aco.optimise()

    optimised_tour = [get_city(city, CITIES) for city in aco.best_path]

    plot(city_list=CITIES, final_tour=optimised_tour, shortest_path=SHORTEST_PATH)    

    