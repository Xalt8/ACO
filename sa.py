import numpy as np
import matplotlib.pyplot as plt


# Source: https://www.youtube.com/watch?v=35fzyblVdmA&ab_channel=ComputationalScientist


class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y   

    @staticmethod
    def get_distance(a:int, b:int) -> float:
        return np.sqrt(np.abs(a.x - b.x) + np.abs(a.y + b.y))

    @staticmethod
    def get_total_distance(coords:list) -> float:
        dist = 0
        for first, second in zip(coords[:-1], coords[1:]):
            dist += Coordinate.get_distance(first, second)
        dist += Coordinate.get_distance(coords[0], coords[-1])
        return dist

if __name__ == '__main__':
    
    # Read in the coordinates from file
    # with open('coordinates.txt', 'r') as f:
    #     lines = [(line.strip().split(', ')) for line in f]
    #     coords = [Coordinate(int(l[0]), int(l[1])) for l in lines]

    coords = [Coordinate(np.random.uniform(), np.random.uniform()) for _ in range(10)]

    #Plot
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for first, second in zip(coords[:-1], coords[1:]):
        ax1.plot([first.x, second.x], [first.y, second.y], 'b')
    ax1.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')

    for c in coords:
        ax1.plot(c.x, c.y, 'ro')

    
    # Simulated Annealing Algorithm
    cost0 = Coordinate.get_total_distance(coords)
    
    T = 30
    factor = 0.99
    T_init = T

    for i in range(1000):
        print(f'{i} cost = {cost0}')

        T = T * factor
        for j in range(100):
            # Exchange two coordinates and get a new neighbour solution
            r1, r2 = np.random.randint(0, len(coords), size=2) # Get two numbers from the indices of the coords

    #         # switch r1 & r2
            temp = coords[r1]
            coords[r1] = coords[r2] 
            coords[r2] = temp

    #         # Get the new cost
            cost1 = Coordinate.get_total_distance(coords=coords)

            if cost1 < cost0:
                cost0 = cost1
            else:
                x = np.random.uniform()
                if x < np.exp((cost0-cost1)/T):
                    cost0 = cost1
                else:
                    # switch r1 & r2
                    temp = coords[r1]
                    coords[r1] = coords[r2] 
                    coords[r2] = temp

    # #Plot
    for first, second in zip(coords[:-1], coords[1:]):
        ax2.plot([first.x, second.x], [first.y, second.y], 'b')
    ax2.plot([coords[0].x, coords[-1].x], [coords[0].y, coords[-1].y], 'b')

    for c in coords:
        ax2.plot(c.x, c.y, 'ro')

    plt.show()



