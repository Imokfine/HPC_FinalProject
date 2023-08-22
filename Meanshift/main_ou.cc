#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio> 
#include <omp.h> 
#include "utils.h"
#include "onlineupdate.h"

int main(int argc, char* argv[]) {

	std::vector<DataPoint> dataPoints;
	std::vector<DataPoint> centroids;
	double windowSize = 10;
	double shiftEpsilon = 0.1;

	std::string datapoints_cpu = "datapoints_cpu_ou.csv";
	std::string centroids_cpu = "centroids_cpu_ou.csv";


	bool timing = true;
	double timeTotalCpu = 0.0;
	struct timeval expoStart, expoEnd;

	int c;
	while ((c = getopt(argc, argv, "tw:e:s:")) != -1) {
		switch (c) {
		case 't':
			timing = true; break;
		case 'w':
			windowSize = atoi(optarg); break;
		case 'e':
			shiftEpsilon = atoi(optarg); break;
		case 's':
		default:
			return -1;
		}
	}

	std::cout << "windowSize = " << windowSize << "\n";
	std::cout << "shiftEpsilon = " << shiftEpsilon << "\n";
	std::cout << "-------------------------------------\n";

	// Read initialized data points and centroids from csv files
	dataPoints = readCSV("datapoints_init.csv");


	gettimeofday(&expoStart, NULL);

	meanshift_ou(dataPoints, centroids, windowSize, shiftEpsilon);

	gettimeofday(&expoEnd, NULL);
	timeTotalCpu = ((expoEnd.tv_sec + expoEnd.tv_usec * 0.000001) - (expoStart.tv_sec + expoStart.tv_usec * 0.000001)); // seconds

	writeToCSV(dataPoints, datapoints_cpu);
	writeToCSV(centroids, centroids_cpu);


	if (timing) {
		std::cout << "-------------------------------------\n";
		printf("kmeans on the cpu (online update) took: %f seconds\n", timeTotalCpu);
	}

	return 0;
}

