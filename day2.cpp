#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <charconv>
#include <regex>

 long partOneAndTwo(std::regex pattern) {

    std::ifstream inputFile("day2.txt");
    if (!inputFile.is_open()) {
        return -1;
    }

    std::string line;
    std::getline(inputFile, line);
    inputFile.close();

    std::stringstream stream(line);
    std::string segment;
    long res = 0;

    while (std::getline(stream, segment, ',')) {
        int idx = segment.find("-");
        long start = std::stol(segment.substr(0, idx));
        long end = std::stol(segment.substr(idx + 1));
        for (long i = start; i <= end; i++) {
            if (std::regex_match(std::to_string(i), pattern)) {
                res += i;
            }
        }
    }
    return res;
} 

int main() {
    std::cout << "Part 1: " << partOneAndTwo(std::regex("^(\\d+)\\1")) << std::endl;
    std::cout << "Part 2: " << partOneAndTwo(std::regex("^(\\d+)\\1+$")) << std::endl;
    return 0;
}