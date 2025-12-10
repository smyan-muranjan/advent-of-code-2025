from functools import lru_cache

def getLines():
    with open("day7.txt") as file:
        lines = [list(line.rstrip("\n")) for line in file]
    return lines

def partOne():
    lines = getLines()
    rows = len(lines)
    cols = len(lines[0])
    splits = 0
    visited = set()

    def dfs(r, c):
        if (r, c) in visited:
            return
        visited.add((r, c))
        
        if r < 0 or r >= rows - 1 or c < 0 or c >= cols:
            return
        
        nonlocal splits
        if lines[r + 1][c] == '^':
            splits += 1
            dfs(r + 1, c - 1)
            dfs(r + 1, c + 1)
        else:
            dfs(r + 1, c)

    for col in range(cols):
        if lines[0][col] == 'S':
            dfs(0, col)
    
    return splits

def partTwo():
    lines = getLines()
    rows = len(lines)
    cols = len(lines[0])

    @lru_cache(maxsize=None)
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0

        if r == rows - 1:
            return 1

        if lines[r + 1][c] == '^':
            return dfs(r + 1, c - 1) + dfs(r + 1, c + 1)
        else:
            return dfs(r + 1, c)

    for col in range(cols):
        if lines[0][col] == 'S':
             return dfs(0, col)

def main():
    print(f"Part 1: {partOne()}")
    print(f"Part 2: {partTwo()}")
    
if __name__ == "__main__":
    main()