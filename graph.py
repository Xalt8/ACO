from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



# https://stevedower.id.au/research/oliver-30 -> tsp 30 city problem source file

with open('coordinates.txt', 'r') as f:
    lines = [(line.strip().split(', ')) for line in f]
    coords = [(int(l[0]), int(l[1])) for l in lines]


with open('shortest_path.txt', 'r') as f:
    short_path = [int(j) for j in [i.split(" ") for i in f][0] if j!=""]


def create_complete_graph(coords:list[tuple]) -> nx.Graph:
    comp_graph = nx.complete_graph(np.arange(1, len(coords) +1))
    attrs = {node: {'x':_x, 'y':_y} for node, (_x, _y) in zip(list(comp_graph.nodes), coords)}
    nx.set_node_attributes(comp_graph, attrs)
    return comp_graph
    

def get_distance(node1:tuple, node2:tuple) -> float:
    '''Returns the distance between 2 nodes/coordinates'''
    return np.linalg.norm(np.array(node1) - np.array(node2))


def set_distance(graph:nx.Graph) -> None:
    ''' Sets the distance as edges of all nodes '''
    for node1, node2 in list(graph.edges):
        graph[node1][node2]['distance'] = get_distance(
            node1=(graph.nodes[node1]['x'], graph.nodes[node1]['y']),
            node2= (graph.nodes[node2]['x'], graph.nodes[node2]['y']))

def set_initial_phermone(graph:nx.Graph) -> None:
    ''' Sets the phermone to 1 for all edges '''
    for node1, node2 in list(graph.edges):
        graph[node1][node2]['phermone'] = 1

def calculate_trip_distance(path:list[tuple], graph:nx.Graph) -> float:
    ''' Takes a list of node edges and returns the total 
        Euclidean distance of the trip '''
    total_distance = []
    for node1, node2 in path:
        total_distance.append(
            get_distance(
                (graph.nodes[node1]['x'], graph.nodes[node1]['y']), 
                (graph.nodes[node2]['x'], graph.nodes[node2]['y'])))
    return np.around(np.sum(total_distance), decimals=2)


def plot_graph(graph:nx.Graph) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(14.562, 9))
    _pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes}
    nx.draw_networkx_nodes(graph, pos=_pos)
    nx.draw_networkx_labels(graph, pos=_pos)
    # nx.draw_networkx_edges(G=complete_graph, edgelist=0, pos=_pos)

    # _pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes}
    # nx.draw(graph, pos=_pos, node_size=400, ax=ax, node_color='skyblue')
    # nx.draw_networkx_labels(graph, pos=_pos)
    ax.set_xlim(0, 100)
    ax.set_ylim(0,110)
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.show()


graph = create_complete_graph(coords)
set_distance(graph) 
set_initial_phermone(graph)


if __name__ == '__main__':
    pass
    
    # graph = create_complete_graph(coords)
    # set_distance(graph) 
    # set_initial_phermone(graph)
    # plot_graph(graph)