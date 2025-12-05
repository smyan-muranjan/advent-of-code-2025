#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <charconv>

int partOne() {
    std::ifstream inputFile("day2.txt");
    if (!inputFile.is_open()) return -1;

    std::string line;
    std::getline(inputFile, line);

    // Pointers to define the bounds of our data
    const char* ptr = line.data();
    const char* end = line.data() + line.size();

    int value; // Re-use this variable for every number

    while (ptr < end) {
        // 1. Parse the integer at the current pointer location
        // std::from_chars is the fastest standard conversion function (no locale overhead)
        auto result = std::from_chars(ptr, end, value);

        // 2. PROCESS YOUR DATA HERE
        // Example: logic to check if 'value' meets criteria
        // if (value > 10) count++; 
        std::cout << "Processing: " << value << "\n";

        // 3. Move the pointer forward
        ptr = result.ptr;

        // 4. Skip the comma if we aren't at the end
        if (ptr < end && *ptr == ',') {
            ptr++;
        }
    }
    
    return 0;
}

int main() {
    partOne();
}