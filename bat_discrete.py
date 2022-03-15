import numpy as np
from numba.experimental import jitclass
from numba import int64, types, njit, typed, typeof, float64
import random
from funcs import plot
from bat import City, CITIES, get_tour_length
from typing import List


@njit
def initialise_population(city_list:list[City], num_bats:int) -> np.ndarray:
    ''' Takes a list of City objects and number of bats (num_bats) and 
        returns an 2D integer array with num_bat number of randomly assigned cities'''
    population = np.empty(shape=(num_bats, len(city_list)), dtype=np.int64)
    indices = np.arange(len(city_list), dtype=np.int64)
    for i in range(num_bats):
        population[i] = np.random.choice(indices, len(indices), replace=False)
    return population


@njit
def get_tour_from_position(position:np.ndarray, city_list:list[City]) -> list[City]:
    ''' Takes a position and returns a tour as list of cities'''
    tour = typed.List()
    for city_index in position:
        tour.append(city_list[city_index])
    return tour


@njit
def get_position_from_tour(tour:list[City], city_list:list[City]) -> np.ndarray:
    '''Takes a list of City objects and returns their index numbers from city_list'''
    position = np.empty(shape=(len(city_list)), dtype=np.int64)
    for j, tour_city in enumerate(tour):
        for i, city in enumerate(city_list):
            if tour_city.name == city.name:
                position[j] = i     
    return position


@njit
def calculate_fitness(city_list:list[City], population:np.ndarray) -> np.ndarray:
    fitness = np.empty(shape=(population.shape[0]), dtype=np.float64)
    for i, pop in enumerate(population):
        tour = get_tour_from_position(position=pop, city_list=city_list)
        fitness[i] = get_tour_length(tour)

    return fitness


@njit
def calculate_hamming_distance(a:np.ndarray, b:np.ndarray) -> int:
    if a.size == b.size:
        return np.sum(np.where(a!=b,1,0))
    else:
        print('Error a and b not the same length!!')


@njit
def calculate_velocity(population:np.ndarray, best_bat_index:int) -> np.ndarray:
    ''' Takes a population of bats and an index value of the best bat & 
        calculates the hamming distance between all bats and the best bat
        and returns a random value between 1 and the hamming distance'''
    velocity = np.zeros(shape=(population.shape[0]), dtype=np.int64)
    for i, pop in enumerate(population):
        if i == best_bat_index:
            continue
        else:
            hamming_distance = calculate_hamming_distance(pop, population[best_bat_index])
            if hamming_distance > 1:
                velocity[i] = random.randint(1, hamming_distance) 
    return velocity

@njit
def twoOpt(pop:np.ndarray, velocity:int, city_list:list[City]) -> np.ndarray:
    ''' 2-Opt local search * velocity number of times'''
    if velocity == 0:
        return pop
    
    best_tour = get_tour_from_position(pop, city_list)
    best_tour_length = get_tour_length(best_tour)

    # Randomly select velocity number of indices of the pop less 3 because i+2  
    selected_indices = np.random.choice(len(pop)-3, velocity, replace=False)
    for i in selected_indices:
        k = i+2
        position = np.hstack((pop[:i], pop[i:k+1][::-1], pop[k+1:]))
        tour = get_tour_from_position(position, city_list)
        tour_length = get_tour_length(tour)
        
        if tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = tour
    new_position = get_position_from_tour(best_tour, city_list)    
    return new_position



@njit
def discrete_batman(city_list:list[City], num_bats:int):
    
    # Inititialise variables
    population = initialise_population(city_list, num_bats)
    fitness = calculate_fitness(city_list, population)
    pulse_rate = np.ones(shape=(num_bats), dtype=np.float64) 
    loudness = np.ones(shape=(num_bats), dtype=np.float64)   
    best_bat = np.argmin(fitness)
    best_tour_position = population[best_bat]
    best_tour = get_tour_from_position(best_tour_position, city_list)
    best_tour_length = get_tour_length(best_tour)
    velocity = calculate_velocity(population=population, best_bat_index=best_bat)


    # Main loop
    for i in range(10):
        print(i, best_tour_length)

        for bat_id, _ in enumerate(population):
            population[bat_id] = twoOpt(population[bat_id], velocity[bat_id], city_list)
        
        fitness = calculate_fitness(city_list, population)
        best_bat = np.argmin(fitness)
        best_tour_position = population[best_bat]
        best_tour = get_tour_from_position(best_tour_position, city_list)
        best_tour_length = get_tour_length(best_tour)
        velocity = calculate_velocity(population=population, best_bat_index=best_bat)                        



if __name__ == '__main__':
    
    discrete_batman(city_list=CITIES, num_bats=10) 
    # population = initialise_population(CITIES, 10)

    # new_pos = twoOpt(population[0], 3, CITIES)

    # print(population[0])
    # ham = calculate_hamming_distance(population[0], new_pos)
    # print(f'Hamming distance = {ham}')


    