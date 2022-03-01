# from sa1 import get_tour_length, generate_tour
from funcs import get_tour_length, generate_tour
from city import CITIES, get_city


# Source: https://github.com/BraveDistribution/pytsp/blob/master/pytsp/k_opt_tsp.py



def twoOptSwap(tour:list[str], i:int, k:int):
    ''' Takes a list of cities and 2 index numbers 
        generates a new tour by:
        1. Starting the tour from the 0 index to i-index
        2. Takes the values from i-index to k-index and reverses them
        3. Takes the values from k+1 index to the end ''' 
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]



def twoOpt(tour:list[str]) -> list[str]:

    best_tour = tour
    city_list = [get_city(city, CITIES) for city in tour]
    best_tour_length = get_tour_length(city_list)
    improved = True

    while improved:
        # print(f'best_length={best_tour_length}')

        improved = False
        for i in range(1, len(tour)-1):
            for k in range(i+1, len(tour)-1):
                new_tour = twoOptSwap(tour, i, k)
                new_city_list = [get_city(city, CITIES) for city in new_tour]
                new_tour_length = get_tour_length(new_city_list)

                if new_tour_length < best_tour_length:
                    best_tour_length = new_tour_length
                    best_tour = new_tour
                    improved = True
                    break
            if improved:
                break
    return best_tour


if __name__ == '__main__':

    initial_tour = generate_tour(CITIES)
    initial_cities = [city.name for city in initial_tour]

    twoOpt(initial_cities)
