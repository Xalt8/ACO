from sa1 import get_tour_length
from city import CITIES, get_city


# Source: https://github.com/BraveDistribution/pytsp/blob/master/pytsp/k_opt_tsp.py



def twoOptSwap(tour:list[str], i:int, k:int):
    ''' Takes a list of cities and 2 index numbers 
        generates a new tour by:
        1. Starting the tour from the 0 index to the i index
        2. Takes the index values from i to k and reverses them
        3. Takes the index values from k+1 to the end ''' 
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]



def twoOpt(tour:list[str]) -> list[str]:

    city_list = [get_city(city, CITIES) for city in tour]
    best_tour_length = get_tour_length(city_list)

    for i in range(1, len(tour)-1):
        for k in range(i+1,len(tour)-1):
            new_tour = twoOptSwap(tour, i, k)
            city_list = [get_city(city, CITIES) for city in new_tour]
            new_tour_length = get_tour_length(city_list)


if __name__ == '__main__':

    pass
    