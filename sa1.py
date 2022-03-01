import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
from opt import twoOpt
from city import City, get_city, CITIES, SHORTEST_PATH
from funcs import plot, generate_tour, get_tour_length

# Source: 
# https://www.theprojectspot.com/tutorials/page/2
# https://www.youtube.com/watch?v=9tYliONxYWE&ab_channel=NoureddinSadawi
# https://www.youtube.com/watch?v=TC9WNwM2noM&ab_channel=IITKharagpurJuly2018



@dataclass
class SimulatedAnnealing:

    cities_list:list[City]
    cooling_rate:float = 0.99
    temp:int = None
    tour:list[City] = None
    tour_length:float = None
    iterations:int = 500
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
            if round(self.temp,2) <= 0.0:
                break
            print(f'{i} Best tour length = {self.best_tour_length}, temp={round(self.temp,2)}')

            # for _ in range(50):
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
                    self.tour = new_tour
                    self.tour_length = new_tour_length

            # Reduce the temp
            self.temp = self.temp * self.cooling_rate

            # 2-opt local search
            tour_cities = [city.name for city in self.tour]
            two_opt_cities = twoOpt(tour_cities)
            self.tour = [get_city(city, self.cities_list) for city in two_opt_cities] 
            self.tour_length = get_tour_length(self.tour) 

            # Set the gbest tour
            self.set_best_tour()
            
            

if __name__ == '__main__':

    sa = SimulatedAnnealing(cities_list = CITIES, temp=50, cooling_rate=0.99)
    sa.optimise()
    
    gbest_tour = sa.best_tour
    plot(city_list=CITIES, final_tour=gbest_tour, shortest_path=SHORTEST_PATH)

    

            
