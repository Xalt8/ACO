import numpy as np
from dataclasses import dataclass, field
import networkx as nx
import random
from graph import graph


@dataclass
class Ant:

    visited_nodes: list[int] = field(default_factory=list, init=False)
    tour: list[tuple] = field(default_factory=list, init=False)
    current_node:int = field(init=False)  
    graph: nx.Graph
    alpha = 1 # Importance of phermone
    beta = 1 # Importance of heuristics
    q0 = 0.5
    rho = 0.99 # used to decay the phermone trail

    
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
        return self.graph[self.current_node][node]['phermone']**self.alpha * \
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
  
    def visit_node(self, node:int) -> None:
        ''' Adds the node to the visited_nodes list'''
        self.visited_nodes.append(node)
        self.tour.append((self.current_node, node))
        self.current_node = node


    def complete_tour(self) -> None:
        ''' Takes the last node visited makes an edge to the first node'''
        #  Takes the visited_nodes list and another list with the first 
        #   node as last element in a list and zips them together
        # return list(zip(self.visited_nodes, self.visited_nodes[1:] + self.visited_nodes[:1]))

        self.tour.append((self.tour[-1][1], self.tour[0][0])) 
    
    def run(self) -> list[tuple]:
        ''' Main loop'''
        # While there are nodes to visit choose and visit them
        while self.get_unvisited_nodes():
            chosen_node = self.choose_node(self.get_unvisited_nodes())
            self.visit_node(chosen_node)
        
        self.complete_tour()
        return self.tour



if __name__ == '__main__':
    
    ant = Ant(graph)
    print(ant.run())