import numpy as np
from dataclasses import dataclass
from city import CITIES, City
from funcs import get_tour_length


@dataclass
class PSO:
    city_list: list[City]
    num_particles: int = 20
    particles: list = None
    gbest_val: float = None
    gbest_pos: np.ndarray = None
    min_max: str = 'min'
    

    def __post_init__(self):
        if self.particles == None:
            self.particles = [dict() for _ in range(self.num_particles)]
        
        assert self.min_max in ['min', 'max'], "Enter 'min' or 'max'!"
        if self.min_max == 'min':
            self.gbest_val = np.Inf
        else:
            self.gbest_val = -np.Inf
        

    def initialise(self):
        ''' Takes a list of cities and random shuffles the index values
            and puts that as the initial position
            Sets the initial velocity to 0 in the same shape as position
            Sets the pbest val to infinity'''

        city_indices = np.arange(len(self.city_list))
        for particle in self.particles:
            position = city_indices.copy()
            np.random.shuffle(position)
            particle['position'] = position
            
            particle['velocity'] = np.zeros(particle['position'].size).astype(np.int64)

            if self.min_max == 'max':
                particle['pbest_val'] = -np.Inf 
            else:
                particle['pbest_val'] = np.Inf   
    
    
    def pick_informants_ring_topology(self):
        for index, particle in enumerate(self.particles):
            particle['informants'] = []
            particle['informants'].append(self.particles[(index-1) % len(self.particles)])
            particle['informants'].append(self.particles[index])
            particle['informants'].append(self.particles[(index+1) % len(self.particles)])
            
            if self.min_max == 'max':
                particle['lbest_val'] = -np.Inf 
            else:
                particle['lbest_val'] = np.Inf
            

    def calculate_fitness(self):
        for particle in self.particles:
            particle['fitness'] = get_tour_length([CITIES[int(i)] for i in particle['position']]) 
            
    
    def set_pbest(self):
        for particle in self.particles:
            if self.min_max == 'max':
                if particle['fitness'] > particle['pbest_val']:
                    particle['pbest_val'] = particle['fitness']
                    particle['pbest_pos'] = particle['position']
            else:
                if particle['fitness'] < particle['pbest_val']:
                    particle['pbest_val'] = particle['fitness']
                    particle['pbest_pos'] = particle['position']


    def set_lbest(self):
        for particle in self.particles:
            for informant in particle['informants']:
                if self.min_max == 'max':
                    if(informant['pbest_val'] > particle['lbest_val']):
                        informant['lbest_val'] = particle['pbest_val']
                        informant['lbest_pos'] = particle['pbest_pos']
                else:
                    if(informant['pbest_val'] < particle['lbest_val']):
                        informant['lbest_val'] = particle['pbest_val']
                        informant['lbest_pos'] = particle['pbest_pos']
    

    def set_gbest(self):
        for particle in self.particles:
            if self.min_max == 'max':
                if particle['lbest_val'] > self.gbest_val:
                    self.gbest_val = particle['lbest_val']
                    self.gbest_pos = particle['lbest_pos']
            else:
                if particle['lbest_val'] < self.gbest_val:
                    self.gbest_val = particle['lbest_val']
                    self.gbest_pos = particle['lbest_pos']


if __name__ == '__main__':
    pass

    