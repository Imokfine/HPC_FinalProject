#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits> 
#include <cmath>
#include <random>
#include <ctime>

extern std::default_random_engine gen;


// Define data structure
struct DataPoint {
    // attributes (suppose only 2 attributes)
    double at1;
    double at2;
    int cluster;
    // Operator
    bool operator==(const DataPoint& other) const {
        return (at1 == other.at1) && (at2 == other.at2) && (cluster == other.cluster);
    }

    bool operator!=(const DataPoint& other) const {
        return !(*this == other);
    }
};


std::vector<DataPoint> generateDataPoints(size_t numPoints);
void initializeCentroid(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, size_t k);
std::vector<DataPoint> readCSV(const std::string& filename);
void writeToCSV(const std::vector<DataPoint>& dataPoints, const std::string& filename);
void printInitUsage();
void compareDataPoints();

#endif

