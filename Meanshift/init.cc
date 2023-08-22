#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <ctime>
#include "utils.h"


int main(int argc, char* argv[]) {

    std::vector<DataPoint> dataPoints;
    std::vector<DataPoint> centroids;
    size_t k = 100;
    size_t numPoints = 100000;
    std::string file_datapoints = "datapoints_init.csv";
    std::string file_centroids = "centroids_init.csv";

    printInitUsage();
    int c;
    while ((c = getopt(argc, argv, "k:n:")) != -1) {
        switch (c) {
        case 'k':
            k = atoi(optarg); break;
        case 'n':
            numPoints = atoi(optarg); break;
        default:
            return -1;
        }
    }

    dataPoints = generateDataPoints(numPoints);
    initializeCentroid(dataPoints, centroids, k);

    writeToCSV(dataPoints, file_datapoints);
    writeToCSV(centroids, file_centroids);

    return 0;
}


