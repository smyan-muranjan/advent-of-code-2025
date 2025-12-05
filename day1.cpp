#include <iostream>
#include <fstream>
#include <string>

int partOne() {
    int res = 0;
    int curr = 50;
    std::string line;
    std::ifstream inputFile("day1.txt");

    if (!inputFile.is_open()) {
        return -1;
    }

    while (std::getline(inputFile, line)) {
        char dir = line[0];
        int len = std::stoi(line.substr(1));

        if (dir == 'R') {
            curr = (curr + len) % 100;
        } else { // dir == 'L'
            curr = (curr + 100 - len) % 100;
        }
        if (curr == 0) {
            res++;
        }
    }

    inputFile.close();
    return res;
}

int partTwo() {
    int res = 0;
    int curr = 50;
    std::string line;
    std::ifstream inputFile("day1.txt");

    if (!inputFile.is_open()) {
        return -1;
    }
    
    while (std::getline(inputFile, line)) {
        char dir = line[0];
        int len = std::stoi(line.substr(1));

        if (dir == 'R') {
            res += (curr + len) / 100;
            curr = (curr + len) % 100;
        } else { // dir == 'L'
            int distToZero = (curr == 0) ? 100 : curr;
            if (len >= distToZero) {
                res += 1 + (len - distToZero) / 100;
            }
            
            int rawPos = (curr - len) % 100;
            curr = (rawPos < 0) ? rawPos + 100 : rawPos;
        }
    }

    inputFile.close();
    return res;
}

int main() {
    std::cout << "Part 1: " << partOne() << std::endl;
    std::cout << "Part 2: " << partTwo() << std::endl;
    return 0;
}