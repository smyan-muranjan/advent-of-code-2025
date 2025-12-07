#include <iostream>
#include <fstream>
#include <vector>


std::pair<std::vector<std::vector<int>>, std::vector<char>> getGridAndOperations() {
    std::ifstream inputFile("day6.txt");
    std::vector<std::vector<int>> grid;
    std::vector<char> operations;
    
    if (!inputFile.is_open()) {
        return {grid, operations};
    }
    
    std::string line;
    
    // Read grid (4 rows)
    for (int i = 0; i < 4; i++) {
        std::getline(inputFile, line);
        std::vector<int> row;
        int j = 0;
        while (j < line.length()) {
            if (line[j] == ' ') {
                j++;
            } else {
                int curr = 0;
                while (j < line.length() && line[j] != ' ') {
                    curr *= 10;
                    curr += (line[j] - '0');
                    j++;
                }
                row.push_back(curr);
            }
        }
        grid.push_back(row);
    }
    
    // Read operations row
    std::getline(inputFile, line);
    int j = 0;
    while (j < line.length()) {
        if (line[j] == ' ') {
            j++;
        } else {
            operations.push_back(line[j]);
            j++;
        }
    }
    
    inputFile.close();
    return {grid, operations};
}


long long partOne() {
    std::pair<std::vector<std::vector<int>>, std::vector<char>> gridAndOperations = getGridAndOperations();
    std::vector<std::vector<int>> grid = gridAndOperations.first;
    std::vector<char> operations = gridAndOperations.second;
    
    long long res = 0;
    for (int i = 0; i < operations.size(); i++) {
        char op = operations[i];
        if (op == '*') {
            long long curr = 1;
            for (int k = 0; k < 4; k++) {
                curr *= grid[k][i];
            }
            res += curr;
        } else {
            long long curr = 0;
            for (int k = 0; k < 4; k++) {
                curr += grid[k][i];
            }
            res += curr;
        }
    }
    return res;
}

long long partTwo() {
    std::pair<std::vector<std::vector<int>>, std::vector<char>> gridAndOperations = getGridAndOperations();
    std::vector<std::vector<int>> grid = gridAndOperations.first;
    std::vector<char> operations = gridAndOperations.second;
    
    std::vector<std::vector<std::string>> gridStrings;
    int maxLen = 0;
    for (int k = 0; k < 4; k++) {
        std::vector<std::string> row;
        for (int i = 0; i < grid[k].size(); i++) {
            std::string numStr = std::to_string(grid[k][i]);
            maxLen = std::max(maxLen, (int)numStr.length());
            row.push_back(numStr);
        }
        gridStrings.push_back(row);
    }
    
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < gridStrings[k].size(); i++) {
            while (gridStrings[k][i].length() < maxLen) {
                gridStrings[k][i] = " " + gridStrings[k][i];
            }
        }
    }
    
    long long res = 0;
    
    for (int i = operations.size() - 1; i >= 0; i--) {
        char op = operations[i];
        
        std::vector<long long> numbers;
        for (int digitPos = maxLen - 1; digitPos >= 0; digitPos--) {
            std::string numStr = "";
            for (int k = 0; k < 4; k++) {
                char digit = gridStrings[k][i][digitPos];
                if (digit != ' ') {
                    numStr += digit;
                }
            }
            if (!numStr.empty()) {
                numbers.push_back(std::stoll(numStr));
            }
        }
        
        if (op == '*') {
            long long curr = 1;
            for (long long num : numbers) {
                curr *= num;
            }
            res += curr;
        } else {
            long long curr = 0;
            for (long long num : numbers) {
                curr += num;
            }
            res += curr;
        }
    }
    
    return res;
}

int main() {
    std::cout << "Part One: " << partOne() << std::endl;
    std::cout << "Part Two: " << partTwo() << std::endl;
    return 0;
}