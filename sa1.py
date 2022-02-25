import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

# Source: 
# https://www.theprojectspot.com/tutorials/page/2
# https://www.youtube.com/watch?v=9tYliONxYWE&ab_channel=NoureddinSadawi
# https://www.youtube.com/watch?v=TC9WNwM2noM&ab_channel=IITKharagpurJuly2018


@dataclass
class City:
    name:str
    x:int
    y:int


# Read in the city coordinates from file
with open('coordinates.txt', 'r') as f:
    lines = [(line.strip().split(', ')) for line in f]
# Create a list of cities names starting from 1
cities = [City(str(i), int(l[0]), int(l[1])) for i, l in enumerate(lines, 1)]

# Read in the shortest path cities
# with open('shortest_path.txt', 'r') as f:
#     short_path = [int(j) for j in [i.split(" ") for i in f][0] if j!=""]

with open('shortest_path.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()

short_path = [int(i) for i in lines[0].split()]


# Create a list of cities with the shortest path
short_path_cities = []
for city_num in short_path:
    for city in cities:
        if str(city_num) == city.name:
            short_path_cities.append(City(name=str(city_num), x=city.x, y=city.y))


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



@dataclass
class SimulatedAnnealing:

    cities_list:list[City]
    cooling_rate:float = 0.99
    temp:int = None
    tour:list[City] = None
    tour_length:float = None
    iterations:int = 200
    best_tour:list[City] = None
    best_tour_length:float = np.inf
    

    def __post_init__(self):
        if self.temp == None:
            self.temp = self.get_initial_temp()
        if self.tour == None:
            self.tour = generate_tour(self.cities_list)
        if self.tour_length == None:
            self.tour_length = get_tour_length(self.tour)
        if self.best_tour == None:
            self.set_best_tour()
    

    def __str__(self) -> str:
        return f'Number of cities={len(self.cities_list)}, temp={round(self.temp,2)}, cooling rate={self.cooling_rate}'

    
    def get_initial_temp(self):
        ''' Generates an average temp from 10 randomly generated tours'''
        return np.mean(np.array([get_tour_length(generate_tour(self.cities_list)) \
                        for _ in range(10)], dtype=np.float64))
        

    def swap_cities(self, tour:list[City], city1_i, city2_i) -> list[City]:
        ''' Takes a list of cities & 2 index values and 
            returns a tour with cities swapped at those index values'''
        tour_list = tour.copy()
        tour_list[city1_i], tour_list[city2_i] = tour_list[city2_i], tour_list[city1_i]
        return tour_list


    def set_best_tour(self) -> None:
        ''' Sets the current tour to the best tour if it
            is better than the best tour'''
        if self.tour_length < self.best_tour_length:
            self.best_tour = self.tour
            self.best_tour_length = self.tour_length


    def optimise(self):
        ''' Main optimising loop'''
        for i in range(self.iterations):
            
            print(f'{i} Best tour length = {self.best_tour_length}')

            # Swap 2 random cities -> new_tour
            r1, r2 = np.random.choice(np.arange((len(self.tour))), size=2, replace=False)
            new_tour = self.swap_cities(self.tour, r1, r2)

            # Compare the fitness of the new_tour with the OG
            new_tour_length = get_tour_length(new_tour)
            if new_tour_length < self.tour_length:
                # If new tour is better than OG -> select it
                self.tour = new_tour
                self.tour_length = new_tour_length
            else: # new_tour is worse than OG
                acceptance_prob = np.exp((new_tour_length - self.tour_length) / self.temp)
                r3 = np.random.uniform(low=0, high=1, size=1)[0]
                if r3 <= acceptance_prob:
                    self.tour = self.tour = new_tour
                    self.tour_length = new_tour_length
            
            # Set the gbest tour
            self.set_best_tour()

            # Reduce the temp
            self.temp = self.temp * self.cooling_rate

    
    def optimise2(self):
        ''' Main optimising loop'''
        for i in range(self.iterations):
            # if round(self.temp,2) == 0.0:
            #     break
            print(f'{i} Best tour length = {self.best_tour_length}, temp={round(self.temp,2)}')

            for _ in range(200):
                # Swap 2 random cities -> new_tour
                r1, r2 = np.random.choice(np.arange((len(self.tour))), size=2, replace=False)
                new_tour = self.swap_cities(self.tour, r1, r2)

                # Compare the fitness of the new_tour with the OG
                new_tour_length = get_tour_length(new_tour)
                if new_tour_length < self.tour_length:
                    # If new tour is better than OG -> select it
                    self.tour = new_tour
                    self.tour_length = new_tour_length
                else: # new_tour is worse than OG
                    acceptance_prob = np.exp((new_tour_length - self.tour_length) / self.temp)
                    r3 = np.random.uniform(low=0, high=1, size=1)[0]
                    if r3 <= acceptance_prob:
                        self.tour = self.tour = new_tour
                        self.tour_length = new_tour_length
                
            # Set the gbest tour
            self.set_best_tour()

            # Reduce the temp
            self.temp = self.temp * self.cooling_rate



def plot(city_list:list[City], final_tour:list[City], shortest_path:list[City]) -> plt.Axes:
    ''' Draws 3 graphs -> start tour, shortest path & optimised path '''
    
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



if __name__ == '__main__':

    sa = SimulatedAnnealing(cities_list = cities, temp=10000, cooling_rate=0.95)
    sa.optimise2()
    # print(f'shortest path = {get_tour_length(short_path_cities)}')

    gbest_tour = sa.best_tour
    plot(city_list=cities, final_tour=gbest_tour, shortest_path=short_path_cities)

    

            
