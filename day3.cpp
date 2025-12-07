#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int partOne() {
    std::fstream inputFile("day3.txt");
    if (!inputFile.is_open()) {
        return -1;
    }
    
    std::string line;
    int res = 0;

    while (std::getline(inputFile, line)) {
        int max_digit = -1;
        int curr_max = 0;

        for (char c : line) {
            int digit = c - '0';

            int voltage = (max_digit * 10) + digit;
            if (voltage > curr_max) {
                curr_max = voltage;
            }
            if (digit > max_digit) {
                max_digit = digit;
            }
        }
        res += curr_max;
    }
    inputFile.close();
    return res;
}

long dfs(std::string curr, std::string &line, int idx) {
    if (curr.length() == 12) {
        return std::stol(curr);
    }
    int digits_left = 12 - curr.length();
    long max = -1;
    for (int i = idx; i <= line.length() - digits_left; i++) {
        std::string next = curr + line[i];
        long val = dfs(next, line, i + 1);
        if (val > max) {
            max = val;
        }
    }
    return max;
}

long partTwoDfs() {
    std::fstream inputFile("day3.txt");
    if (!inputFile.is_open()) {
        return -1;
    }
    
    std::string line;
    long res = 0;

    while (std::getline(inputFile, line)) {
        res += dfs("", line, 0);
    }
    inputFile.close();
    return res;
}

int main() {
    std::cout << "Part One: " << partOne() << std::endl;
    std::cout << "Part Two: " << partTwoDfs() << std::endl;
    return 0;
}