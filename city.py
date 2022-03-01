from dataclasses import dataclass

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


if __name__ == '__main__':
    
    pass