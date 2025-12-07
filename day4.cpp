#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<char>> getGrid() {
    std::ifstream inputFile("day4.txt");
    if (!inputFile.is_open()) {
        return {{}};
    }
    std::string line;
    std::vector<std::vector<char>> grid;
    while (std::getline(inputFile, line)) {
        std::vector<char> curr;
        for (char c : line) {
            curr.push_back(c);
        }
        grid.push_back(curr);
    }
    inputFile.close();
    return grid;
}

int partOne() {
    std::vector<std::vector<char>> grid = getGrid();
    int res = 0;
    std::vector<std::vector<int>> dirs = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    int rows = grid.size();
    int cols = grid[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] != '@') {
                continue;
            }
            int curr = 0;
            for (std::vector<int> d : dirs) {
                int dr = r + d[0];
                int dc = c + d[1];
                if (0 <= dr && dr < rows && 0 <= dc && dc < cols && grid[dr][dc] == '@') {
                    curr++;
                }
            }
            if (curr < 4) {
                res++;
            }
        }
    }
    return res;
}

int partTwo() {
    std::vector<std::vector<char>> grid = getGrid();
    std::vector<std::vector<int>> dirs = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    int rows = grid.size();
    int cols = grid[0].size();
    int removed;
    int res = 0;
    do {
        removed = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] != '@') {
                    continue;
                }
                int curr = 0;
                for (std::vector<int> d : dirs) {
                    int dr = r + d[0];
                    int dc = c + d[1];
                    if (0 <= dr && dr < rows && 0 <= dc && dc < cols && grid[dr][dc] == '@') {
                        curr++;
                    }
                }
                if (curr < 4) {
                    removed ++;
                    grid[r][c] = '.';
                }
            }
        }
        res += removed;
    } while (removed > 0);
    
    return res;
}

int main() {
    std::cout << "Part 1: " << partOne() << std::endl;
    std::cout << "Part 2: " << partTwo() <<  std::endl;
    return 0;
}