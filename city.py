from dataclasses import dataclass
import numpy as np

@dataclass
class City:
    name:str
    x:int
    y:int

# Read in the city coordinates from file
with open('coordinates.txt', 'r') as f:
    lines = [(line.strip().split(', ')) for line in f]
# Create a list of cities names starting from 1
CITIES = [City(str(i), int(l[0]), int(l[1])) for i, l in enumerate(lines, 1)]


# Read in the shortest path from file
with open('shortest_path.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()

short_path = [int(i) for i in lines[0].split()]


# Create a list of cities with the shortest path
SHORTEST_PATH = []
for city_num in short_path:
    for city in CITIES:
        if str(city_num) == city.name:
            SHORTEST_PATH.append(City(name=str(city_num), x=city.x, y=city.y))


class NotFoundError(Exception):
    pass


def get_city(city_name:str, city_list:list[City])-> City:
    ''' Takes a city name and returns a City object'''
    for city in city_list:
        if city.name == city_name:
            return city
        
    raise NotFoundError(f'The city {city_name} could not be found!')


def calculate_distance(city1:City, city2:City)-> float:
    ''' Takes 2 cities and returns the distance between them'''
    return round(np.sqrt(np.abs(city1.x - city2.x)**2 + np.abs(city1.y - city2.y)**2),2)


if __name__ == '__main__':
    
    print(f"short_path -> {short_path}")
    short_path_tuples = [(city1, city2) for city1, city2 in zip(short_path[0:-1], short_path[1:])]
    short_path_tuples.append((short_path[-1], short_path[0]))
    print(f"\nshort_path_tuples -> {short_path_tuples}")
    short_path_distances = [calculate_distance(get_city(str(tups[0]), CITIES), get_city(str(tups[1]), CITIES)) for tups in short_path_tuples] 
    print(f"short_path_distances -> {short_path_distances}")
    print(f"short_distance -> {sum(short_path_distances)}")

    print(CITIES[0], CITIES[2])
    print(f"distance between cities 1 & 3 -> {calculate_distance(CITIES[0], CITIES[2])}")