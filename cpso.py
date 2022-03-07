import numpy as np
from pso import PSO
from dataclasses import dataclass
from city import CITIES, City, SHORTEST_PATH
from opt import twoOpt
from funcs import plot



@dataclass
class CPSO(PSO):
    
    def set_constricted_velocity(self):
        for particle in self.particles:
            c1 = 2.05
            c2 = 2.05
            ep = c1 + c2
            X = 2/(abs(2-ep-np.sqrt((ep**2)-4*ep)))
            dims = particle['position'].shape
            cognitive = (c1 * np.random.uniform(0, 1, dims) * (particle['pbest_pos'] - particle['position']))
            informers = (c2 * np.random.uniform(0, 1, dims) * (particle['lbest_pos'] - particle['position']))
            new_velocity = X * (particle['velocity'] + cognitive + informers)
            particle['velocity'] = new_velocity


    def random_back(self, position:np.ndarray, velocity:np.ndarray) -> np.ndarray:
        ''' Checks to see if the position + velocity value return a valid result
            If not valid -> check for the closest value available in the indicies
            and take that value as the new position        
        '''
        indicies = np.arange(len(self.city_list))
        assert indicies.size == position.size, 'position & number of indices not same size'
        
        new_position = np.floor(position + velocity).astype(np.int64)
        for j, n_pos in enumerate(new_position):
            if n_pos in indicies:
                indicies = np.delete(indicies, np.where(n_pos==indicies)[0][0])
            else:
                closest_ind = np.argmin([np.abs(n_pos-val) for val in indicies])
                new_position[j] = indicies[closest_ind]
                indicies = np.delete(indicies, closest_ind)
        return new_position

    
    def move_random_back(self):
        for particle in self.particles:
            new_pos = self.random_back(particle['position'], particle['velocity'])
            particle['position'] = np.floor(new_pos)


    def local_search(self):
        ''' Gets the tour by using the particle's position as index
            Does a 2-Opt local search and returns the result as the index
            positions from city_list'''
        for particle in self.particles:
            tour = [self.city_list[int(city)].name for city in particle['position']]
            two_opt_cities = twoOpt(tour)
            particle['position'] = np.array([[city.name for city in self.city_list].index(toc) 
                                            for toc in two_opt_cities], dtype=np.int64) 


    def optimise(self):
        
        iterations = 500
        
        self.initialise()
        self.pick_informants_ring_topology()

        for i in range(iterations):

            print(f"Iteration: {i} gbest_val: {round(self.gbest_val, 2)}")    

            self.calculate_fitness()
            self.set_pbest()
            self.set_lbest()
            self.set_gbest()  
            self.set_constricted_velocity()
            self.move_random_back()
            # self.local_search()


if __name__ == '__main__':
    
    cpso = CPSO(city_list=CITIES, min_max='min')
    cpso.optimise()
    tour = [CITIES[int(city)] for city in cpso.gbest_pos]

    plot(city_list=CITIES, final_tour=tour, shortest_path=SHORTEST_PATH)

