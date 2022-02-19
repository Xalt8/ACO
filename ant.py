import numpy as np
from dataclasses import dataclass, field
import networkx as nx
import random
from graph import get_graph, coords, get_distance, short_path, plot_graph


@dataclass
class Ant:

    visited_nodes: list[int] = field(default_factory=list, init=False)
    tour: list[tuple] = field(default_factory=list, init=False)
    current_node:int = field(init=False)  
    graph: nx.Graph
    beta = 2    # Importance of heuristics
    q0 = 0.9    # Higher number leads to more exploitation 
    rho = 0.1   # used to decay the phermone trail
    tau = 0.0005 # initial pheromone

    
    def __post_init__(self):
        self.current_node = self.choose_start_node()
        self.visited_nodes.append(self.current_node)

    def choose_start_node(self) -> int:
        ''' Randomly chooses a node from a 
            list of nodes in a graph'''
        return random.choice(list(self.graph.nodes))
    
    def get_unvisited_nodes(self) -> list[int]:
        ''' Returns a list of unvisited nodes'''
        return [node for node in list(self.graph.nodes) if node not in self.visited_nodes]
    
    def score_node(self, node:int) -> float:
        ''' Scores the node based on phermone and distance'''
        return self.graph[self.current_node][node]['phermone'] * \
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
            
        return node

    def local_phermone_update(self, node:int) -> None:
        ''' Updates the graph with the local pheromone update rule
            for each node visited'''
        self.graph[self.current_node][node]['phermone'] = \
            (1 - self.rho) * self.graph[self.current_node][node]['phermone'] + (self.rho * self.tau)

  
    def visit_node(self, node:int) -> None:
        ''' Adds the node to the visited_nodes list'''
        self.visited_nodes.append(node)
        self.tour.append((self.current_node, node))
        self.current_node = node


    def complete_tour(self) -> None:
        ''' Takes the last node visited makes an edge to the first node'''
        self.tour.append((self.tour[-1][1], self.tour[0][0])) 
    
    def run(self) -> list[tuple]:
        ''' Main loop
            While there are unvisited nodes choose a node
            apply local phermone when visiting node and visit the node'''
        while self.get_unvisited_nodes():
            chosen_node = self.choose_node(self.get_unvisited_nodes())
            self.local_phermone_update(chosen_node)
            self.visit_node(chosen_node)
        
        self.complete_tour()
        return self.tour


@dataclass
class AntColony:

    graph: nx.Graph
    num_ants:int = 10
    best_path_distance:float = np.inf
    alpha = 0.1
    best_path:list[tuple] = field(default_factory=list,  init=False) 
    ants:list[Ant] = field(default_factory=list,  init=False)
    

    def spawn_ants(self) -> list[Ant]:
        ''' Creates a list of Ant objects'''
        self.ants = [Ant(self.graph) for _ in range(self.num_ants)]

    def calculate_trip_distance(self, path:list[tuple]) -> float:
        ''' Takes a list of edges(tuples) and returns the total 
            Euclidean distance of all edges '''
        total_distance = []
        for node1, node2 in path:
            total_distance.append(
                get_distance(
                    (self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']), 
                    (self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y'])))
        return np.around(np.sum(total_distance), decimals=2)

    def set_global_best(self) -> None:
        ''' For all ants -> makes a tour, calculate the distance of tour 
            and if the distance is better than the gbest tour it updates the gbest tour'''
        for ant in self.ants:
            tour = ant.run()
            tour_distance = self.calculate_trip_distance(tour)
            if tour_distance < self.best_path_distance:
                self.best_path_distance = tour_distance
                self.best_path = tour

    def global_update_pheromone(self) -> None:
        ''' Takes the best tour and adds pheromone to the edges. '''
        for node1, node2 in self.best_path:
            self.graph[node1][node2]['phermone'] = \
                (1 - self.alpha) * self.graph[node1][node2]['phermone'] + \
                    self.alpha * (self.best_path_distance ** -1)
    
    def optimize(self):
        
        iterations = 500
                
        for i in range(iterations):

            if self.calculate_trip_distance(short_path) == self.best_path_distance:
                print(f"Found shortest path in {i} iterations")
                break
            
            self.spawn_ants()
            self.set_global_best()
            self.global_update_pheromone()

            print(f'{i} best distance:{self.best_path_distance}')
            
        return self.best_path

if __name__ == '__main__':
        
    graph = get_graph(coords=coords)
    colony = AntColony(graph=graph)
    result = colony.optimize()
    

    