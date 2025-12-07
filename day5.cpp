#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

struct Interval {
    long long start;
    long long end;
};

std::vector<Interval> getMergedIntervals() {
    std::ifstream inputFile("day5.txt");
    std::string line;
    std::vector<Interval> intervals;
    while (std::getline(inputFile, line) && line != "") {
        int i = line.find("-");
        Interval curr = {
            std::stoll(line.substr(0, i)), std::stoll(line.substr(i + 1))
        };
        intervals.push_back(curr);
    }
    inputFile.close();
    std::sort(intervals.begin(), intervals.end(),
          [](const Interval& a,
             const Interval& b) {
              return a.start < b.start;
          });
    std::vector<Interval> merged;
    for (Interval curr : intervals) {
        if (merged.empty() || merged.back().end < curr.start) {
            merged.push_back(curr);
        } else {
            merged.back().end = std::max(merged.back().end, curr.end);
        }
    }
    return merged;
}

int partOne() {
    std::vector<Interval> merged = getMergedIntervals();
    std::ifstream inputFile("day5.txt");
    std::string line;
    // Skip intervals and blank line
    while (std::getline(inputFile, line) && line != "") {}
    int res = 0;
    while (std::getline(inputFile, line)) {
        long long curr = std::stoll(line);
        for (Interval i : merged) {
            if (i.start <= curr && curr <= i.end) {
                res++;
            }
        }
    }
    inputFile.close();
    return res;
}

long long partTwo() {
    std::vector<Interval> merged = getMergedIntervals();
    long long res = 0;
    for (Interval i : merged) {
        res += (i.end - i.start + 1);
    }
    return res;
}

int main() {
    std::cout << "Part 1: " << partOne() << std::endl;
    std::cout << "Part 2: " << partTwo() << std::endl;
    return 0;
}