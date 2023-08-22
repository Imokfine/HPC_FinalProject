#include "utils.h"
std::default_random_engine gen;

// Generate random data points
std::vector<DataPoint> generateDataPoints(size_t numPoints) {
    gen.seed(std::time(nullptr));
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    std::vector<DataPoint> dataPoints;
    dataPoints.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        DataPoint point;
        point.at1 = dis(gen); // Generate a random first attribute value
        point.at2 = dis(gen); // Generate a random second attribute value
	point.cluster = 0;
	dataPoints.push_back(point);
    }

    return dataPoints;
}

// Initialize Centroids
void initializeCentroid(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, size_t k) {
    if (dataPoints.empty()) {
        std::cerr << "The vector dataPoints is empty" << std::endl;
        return;
    }

    gen.seed(std::time(nullptr));
    std::uniform_int_distribution<std::vector<DataPoint>::size_type> idxran{ 0, dataPoints.size() - 1 };

    for (size_t i = 0; i < k; i++) {
        std::vector<DataPoint>::size_type idx = idxran(gen);
        centroids.push_back(dataPoints[idx]);
    }
}

// Read the CSV file that holds the DataPoint
std::vector<DataPoint> readCSV(const std::string& filename) {
    std::vector<DataPoint> datapoints;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Cannot open file ：" << filename << std::endl;
        return datapoints; // Returns empty datapoints
    }

    if (file) {
        std::string line;
        std::getline(file, line); // Read the header row of the CSV file and ignore it

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            DataPoint datapoint;

            std::getline(ss, cell, ',');
            datapoint.at1 = std::stod(cell);

            std::getline(ss, cell, ',');
            datapoint.at2 = std::stod(cell);

            std::getline(ss, cell, ',');
            datapoint.cluster = std::stod(cell);

            datapoints.push_back(datapoint);
        }
    }

    return datapoints;
}


// Write the data points and the clusters they belong to to a CSV file
void writeToCSV(const std::vector<DataPoint>& dataPoints, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        // Write the header of the CSV file
        outputFile << "Attribute1,Attribute2,Cluster" << std::endl;

        // Write the data points and the cluster to which they belong row by row
        for (const auto& dataPoint : dataPoints) {
            outputFile << dataPoint.at1 << "," << dataPoint.at2 << "," << dataPoint.cluster << std::endl;
        }

        outputFile.close();
        std::cout << "Data points written to " << filename << " successfully." << std::endl;
    }
    else {
        std::cerr << "Unable to open the file: " << filename << std::endl;
    }
}

void printInitUsage() {
    std::cout << "This program will generate data points and initialize centroids randomly\n";
    std::cout << "usage:\n";
    std::cout << "      -n   size    : generate n data points (default: 100000)\n";
    std::cout << "      -k   size    : initialize k centroids (default: 100)\n";
    std::cout << "     \n";
}

void compareDataPoints() {
    std::vector<DataPoint> dataPoints_cpu;
    std::vector<DataPoint> centroids_cpu;
    std::vector<DataPoint> dataPoints_gpu;
    std::vector<DataPoint> centroids_gpu;
    std::vector<DataPoint> dataPoints_omp;
    std::vector<DataPoint> centroids_omp;
    std::vector<DataPoint> dataPoints_mpi;
    std::vector<DataPoint> centroids_mpi;

    dataPoints_cpu = readCSV("datapoints_cpu.csv");
    centroids_cpu = readCSV("centroids_cpu.csv");
    dataPoints_gpu = readCSV("datapoints_gpu.csv");
    centroids_gpu = readCSV("centroids_gpu.csv");
    dataPoints_omp = readCSV("datapoints_omp.csv");
    centroids_omp = readCSV("centroids_omp.csv");
    dataPoints_mpi = readCSV("datapoints_mpi.csv");
    centroids_mpi = readCSV("centroids_mpi.csv");

    bool cpugpu_same = true;
    bool cpuomp_same = true;
    bool cpumpi_same = true;

    for (size_t i = 0; i < dataPoints_cpu.size(); i++) {
        if (dataPoints_cpu[i].cluster != dataPoints_gpu[i].cluster) {
            cpugpu_same = false;
        }
        if (dataPoints_cpu[i].cluster != dataPoints_omp[i].cluster) {
            cpuomp_same = false;
        }
        if (dataPoints_cpu[i].cluster != dataPoints_mpi[i].cluster) {
            cpumpi_same = false;
        }
    }

    // Print results
    if (cpugpu_same) {
        std::cout << "Data Points of CPU and Data Points of GPU are same\n";
    }
    else {
        std::cout << "The result of GPU is wrong\n";
    }
    if (cpuomp_same) {
        std::cout << "Data Points of CPU and Data Points of OpenMP are same\n";
    }
    else {
        std::cout << "The result of OpenMP is wrong\n";
    }

    if (cpumpi_same) {
        std::cout << "Data Points of CPU and Data Points of MPI are same\n";
    }
    else {
        std::cout << "The result of MPI is wrong\n";
    }


}
