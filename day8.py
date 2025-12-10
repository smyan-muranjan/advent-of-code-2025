from typing import List, Tuple

def getCoordinates() -> List[Tuple[int, int, int]]:
    with open("day8.txt") as f:
        coords = [tuple(map(int, line.strip().split(","))) for line in f]
    return coords

def getSquaredSortedCoordinates():
    coordinates = getCoordinates()
    n = len(coordinates)
    all_connections: List[Tuple[int, int, int]] = []
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            coord1 = coordinates[i]
            coord2 = coordinates[j]
            squared_dist = (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2
            all_connections.append((squared_dist, i, j))
    all_connections.sort()
    return all_connections, n

class UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))
        self.sizes = [1] * size
    
    def find(self, i):
        if self.parents[i] == i: # this is the representative for the set, it has not parent
            return i
        return self.find(self.parents[i])
    
    def unite(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        
        if irep != jrep:
            if self.sizes[irep] < self.sizes[jrep]:
                irep, jrep = jrep, irep
            self.parents[jrep] = irep
            self.sizes[irep] += self.sizes[jrep]
            return True # Union occurred
        return False # Already connected

def partOne():
    NUM_CONNECTIONS = 1000
    all_connections, n = getSquaredSortedCoordinates()
    uf = UnionFind(n)
    edges_to_process = all_connections[:NUM_CONNECTIONS]
    for _, i, j in edges_to_process:
        uf.unite(i, j) 
        
    circuit_sizes = sorted(uf.sizes, reverse=True)
    return circuit_sizes[0] * circuit_sizes[1] * circuit_sizes[2]

def partTwo() -> int:
    coordinates = getCoordinates()
    all_connections, n = getSquaredSortedCoordinates()
    uf = UnionFind(n)
    num_components = n
    
    for _, i, j in all_connections:
        if uf.unite(i, j):
            num_components -= 1
            if num_components == 1:
                return coordinates[i][0] * coordinates[j][0]
    return -1

def main():
    print(f"Part 1: {partOne()}")
    print(f"Part 2: {partTwo()}")
    
if __name__ == "__main__":
    main()