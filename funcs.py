import numpy as np
import random
import matplotlib.pyplot as plt
from city import City


def calculate_distance(city1:City, city2:City)-> float:
    ''' Takes 2 cities and returns the distance between them'''
    return round(np.sqrt(np.abs(city1.x - city2.x)**2 + np.abs(city1.y - city2.y)**2),2)


def generate_tour(city_list:list[City])-> list[City]:
    ''' Takes a list of cities and randomly selects the order of visits'''
    tour = city_list.copy()
    random.shuffle(tour)
    return tour


def get_tour_length(tour:list[City]) -> float:
    '''Takes a list of cities and returns the distance travelled'''
    distance = 0
    for city1, city2 in zip(tour[1:], tour[:-1]):
        distance += calculate_distance(city1, city2)
    # Close the loop -> connect the first & last cities
    distance += calculate_distance(tour[0], tour[-1])
    return round(distance,2)    


def plot(city_list:list[City], final_tour:list[City], shortest_path:list[City]) -> plt.Axes:
    ''' Draws 2 graphs -> shortest path & optimised path '''
    
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Draw the cities
    for city in city_list:
        ax1.text(city.x, city.y, city.name, color='black', size= 6,
                    bbox=dict(boxstyle="circle", facecolor='lightblue', edgecolor='blue'))
        ax2.text(city.x, city.y, city.name, color='black', size= 6,
                    bbox=dict(boxstyle="circle", facecolor='lightblue', edgecolor='blue'))
    
    # Get max & min x values & add 10 pt buffer
    x_max = max([city.x for city in city_list]) + 10
    x_min = min([city.x for city in city_list]) - 10
    # Get max & min y values & add 10 pt buffer
    y_max = max([city.y for city in city_list]) + 10
    y_min = min([city.y for city in city_list]) - 10
    
    # Set the limits and title
    ax1.set(title='Final tour', xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax2.set(title='Shortest tour', xlim=(x_min, x_max), ylim=(y_min, y_max))
    
    # Draw the start tour
    for city1, city2 in zip(final_tour[:-1], final_tour[1:]):
        ax1.plot([city1.x, city2.x], [city1.y, city2.y], color='green', linestyle='dotted')
    # Close the loop -> connect the first & last cities
    ax1.plot([final_tour[0].x, final_tour[-1].x], [final_tour[0].y, final_tour[-1].y], color='green', linestyle='dotted')

    # Set the start tour length
    ax1.text(y_max *.25, x_max, 'Tour length = '+str(get_tour_length(final_tour)), color='firebrick', size= 10,
                    bbox=dict(boxstyle="round", facecolor='snow', edgecolor='red'))

     # Draw the shortest path
    for city1, city2 in zip(shortest_path[:-1], shortest_path[1:]):
        ax2.plot([city1.x, city2.x], [city1.y, city2.y], color='red', linestyle='-')
    # Close the loop -> connect the first & last cities
    ax2.plot([shortest_path[0].x, shortest_path[-1].x], [shortest_path[0].y, shortest_path[-1].y], color='red')

    # Set the start tour length
    ax2.text(y_max *.25, x_max, 'Tour length = '+str(get_tour_length(shortest_path)), color='firebrick', size= 10,
                    bbox=dict(boxstyle="round", facecolor='snow', edgecolor='red'))
    
    plt.tight_layout()
    plt.show()


if __name__ =='__main__':
    pass