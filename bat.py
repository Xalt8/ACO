from re import X
from unicodedata import name
import numpy as np
from numba.experimental import jitclass
from numba import int64, types, njit, typed, typeof, float64
import random


# Source: https://www.youtube.com/watch?v=4OfJa3SfU84


city_spec=[
    ('name', types.unicode_type),
    ('x', int64),
    ('y', int64)]

@jitclass(city_spec)
class City:
    def __init__(self, name:str, x:int, y:int) -> None:
        self.name = name
        self.x = x
        self.y = y

@njit
def calculate_distance(city1, city2)-> float:
    ''' Takes 2 cities and returns the distance between them'''
    return round(np.sqrt(np.abs(city1.x - city2.x)**2 + np.abs(city1.y - city2.y)**2),2)


@njit
def get_tour_length(tour:list[City]) -> float:
    '''Takes a list of cities and returns the distance travelled'''
    distance = 0
    for city1, city2 in zip(tour[1:], tour[:-1]):
        distance += calculate_distance(city1, city2)
    # Close the loop -> connect the first & last cities
    distance += calculate_distance(tour[0], tour[-1])
    return round(distance,2)    


@njit
def get_city(city_name:str, city_list:list[City])-> City:
    ''' Takes a city name and returns a City object'''
    return [city for city in city_list if city.name==city_name][0]  
    

# Read in the city coordinates from file
with open('coordinates.txt', 'r') as f:
    lines = [(line.strip().split(', ')) for line in f]
# Create a list of cities names starting from 1
CITIES = typed.List()
for i, line in enumerate(lines, 1):
    CITIES.append(City(name=str(i), x=int(line[0]), y=int(line[1])))


bat_spec=[
    ('pop_size', int64),
    ('num_move', int64),
    ('city_list', typeof(CITIES)),
    ('population', int64[:,:]),
    ('fitness', float64[:]),
    ('velocity', float64[:,:]),
    ('loudness', float64[:,:]),
    ('pulse_rate', float64[:,:]),
    ('frequency', float64[:,:]),
    ('frequency_range', float64[:]),
    ('alpha', float64),
    ('gamma', float64),
    ('best_tour', float64[:]), # Change this to list[City]
    ('gamma', float64),
    ('best_tour', int64[:]),
    ('best_fitness', float64),
    ('best_bat', int64)
    ]

@jitclass(bat_spec)
class BatAlgorithm:
    def __init__(self, city_list:list[City], pop_size:int = 10, num_move:int = 10) -> None:
        
        self.city_list = city_list
        self.pop_size = pop_size
        self.num_move = num_move
        self.population = self.initialise_population()
        self.fitness = self.calculate_fitness()
        self.velocity = np.zeros(shape=(self.pop_size, len(self.city_list)), dtype=np.float64)
        self.loudness = np.ones(shape=(self.pop_size, 1), dtype=np.float64)
        self.pulse_rate = np.ones(shape=(self.pop_size, 1), dtype=np.float64)
        self.frequency = np.zeros(shape=(self.pop_size, len(self.city_list)), dtype=np.float64)
        self.frequency_range = np.array([0,1], dtype=np.float64)
        self.alpha = 0.9 
        self.gamma = 0.9
        self.best_tour = np.empty(shape=(len(self.city_list)), dtype=np.int64) # Change this to list[City]
        self.best_bat = np.argmin(self.fitness)
        self.best_fitness = self.fitness[self.best_bat]
        


    def initialise_population(self) -> np.ndarray:
        ''' Returns a 2D array -> pop_size number of index values 
            of cities_list in random order '''
        population = np.empty(shape=(self.pop_size, len(self.city_list)), dtype=np.int64)
        indices = np.arange(len(self.city_list), dtype=np.int64)
        for i,_ in enumerate(population):
            population[i] = np.random.choice(indices, len(indices), replace=False)
        return population


    def get_cities_from_position(self, position:np.ndarray) -> list[City]:
        ''' Takes a position and returns a tour as list of cities'''
        tour = typed.List()
        for city_index in position:
            tour.append(self.city_list[city_index])
        return tour


    def calculate_fitness(self) -> np.ndarray:
        ''' Takes the population and calculates the fitness'''
        fitness = np.empty(shape=(self.pop_size))
        for i, pop in enumerate(self.population):
            tour = typed.List()
            for city_index in pop:
                tour.append(self.city_list[city_index])

            fitness[i] = get_tour_length(tour)

        return fitness
    

    def feasible_position(self, position:np.ndarray) -> np.ndarray:
        ''' Returns a feasible postion from continuous values by replacing 
            a floating point value with an integer that is the closest in value'''
        new_position = np.empty(position.size, dtype=np.int64)
        indicies = np.arange(len(self.city_list), dtype=np.int64)
        if indicies.size == new_position.size:#replace assert
            for j, n_pos in enumerate(new_position):
                if n_pos in indicies:
                    indicies = np.delete(indicies, np.where(n_pos==indicies)[0][0])
                    new_position[j] = int(n_pos)
                else:
                    closest_ind = np.argmin(np.array([np.abs(n_pos-val) for val in indicies], dtype=np.int64))
                    new_position[j] = indicies[closest_ind]
                    indicies = np.delete(indicies, closest_ind)
        else:
            print('Indices and position not the same size!!')
            return
        return new_position

    
    def random_walk(self, position:np.ndarray)-> np.ndarray:
        ''' Takes the position and adds it to the 
            product of average loudness & and a random value '''
        r1 = random.uniform(-1,1)
        new_position = position + (r1 * np.mean(self.loudness))
        return self.feasible_position(new_position)


    
    def optimise_continuous(self) -> None:
        
        # setgbest
        # self.best_bat = np.argmin(self.fitness)

        for i in range(self.num_move):
            print(i, self.best_fitness)
            for index, bat in enumerate(self.population):
                # Update frequency -> fmin+(fmax - fmin)*beta 
                self.frequency[index] = self.frequency_range[0] + (self.frequency_range[1]-self.frequency_range[0]) * \
                        np.random.rand(self.population.shape[1])
                # update velocity -> old_velocity + (position - gbest)*frequency
                self.velocity[index] = self.velocity[index] + (self.population[index] - self.population[self.best_bat]) * \
                                        self.frequency[index]
                # update position
                temp_position = np.floor(self.population[index] + self.velocity[index]).astype(np.int64)
                new_position = self.feasible_position(temp_position)
                # local search
                if random.random() > self.pulse_rate[index]:
                    new_position = self.random_walk(new_position)
                # Calculate fitness
                tour = self.get_cities_from_position(new_position)
                new_pos_fitness = get_tour_length(tour)
                # Fly randomly & generate new solutions
                if (random.random() < self.loudness[index]) & (new_pos_fitness < self.fitness[self.best_bat]):
                    self.population[index] = new_position
                    self.fitness[index] = new_pos_fitness
                    if self.loudness[index] > 0.05:
                        self.loudness[index] *= self.alpha
                    else:
                        self.loudness[index] = 0.05
                    self.pulse_rate[index] = self.pulse_rate[index] * (1 - np.exp(-self.gamma * i))
                # Rank the bats and get best bat
                if self.fitness[index] < self.fitness[self.best_bat]:
                    self.best_bat = index
                    self.best_fitness = self.fitness[index]
                    self.best_tour = self.population[index]
                    

if __name__=='__main__':
   
    
    batman = BatAlgorithm(city_list=CITIES)
    batman.optimise_continuous()