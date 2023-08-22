#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <omp.h>
#include "utils.h"
#include "kmeans.h"

// Include the struct and functions definitions here

int main(int argc, char* argv[]) {

	int ompthreads = 48;
	int blockSize = 256;

	std::vector<DataPoint> dataPoints;
	std::vector<DataPoint> centroids;
	int maxIterations = 10000;
	std::string datapoints_cpu = "datapoints_cpu.csv";
	std::string centroids_cpu = "centroids_cpu.csv";
	std::string datapoints_gpu = "datapoints_gpu.csv";
	std::string centroids_gpu = "centroids_gpu.csv";
	std::string datapoints_omp = "datapoints_omp.csv";
	std::string centroids_omp = "centroids_omp.csv";

	bool timing = true;
	bool cpu = true;
	bool gpu = true;
	bool omp = true;
	double timeTotalCpu = 0.0;
	float timeTotalGpu = 0.0;
	double timeTotalOmp = 0.0;
	double omp_start, omp_end;
	struct timeval expoStart, expoEnd;

	int c;
	while ((c = getopt(argc, argv, "tcgoi:s:b:")) != -1) {
		switch (c) {
		case 't':
			timing = true; break;
		case 'c':
			cpu = false; break;	 //Skip the CPU test
		case 'g':
			gpu = false; break;	 //Skip the OpenMP test
		case 'o':
			omp = false; break;	 //Skip the OpenMP test
		case 'i':
			maxIterations = atoi(optarg); break;
		case 's':
			ompthreads = atoi(optarg); break;
		case 'b':
			blockSize = atoi(optarg); break;
		default:
			return -1;
		}
	}

	// Read initialized data points and centroids from csv files
	dataPoints = readCSV("datapoints_init.csv");
	centroids = readCSV("centroids_init.csv");

	if (cpu) {
		gettimeofday(&expoStart, NULL);

		KMeans(dataPoints, centroids, maxIterations);

		gettimeofday(&expoEnd, NULL);
		timeTotalCpu = ((expoEnd.tv_sec + expoEnd.tv_usec * 0.000001) - (expoStart.tv_sec + expoStart.tv_usec * 0.000001)); // seconds

		writeToCSV(dataPoints, datapoints_cpu);
		writeToCSV(centroids, centroids_cpu);
	}

	// Read initialized data points and centroidsagain
	dataPoints = readCSV("datapoints_init.csv");
	centroids = readCSV("centroids_init.csv");

	if (gpu) {

		cudaEvent_t start, finish;
		cudaEventCreate(&start);
		cudaEventCreate(&finish);

		// Start timing
		cudaEventRecord(start);

		// Initialize CUDA environment
		cudaSetDevice(0);

		KMeans_gpu(dataPoints, centroids, maxIterations, blockSize);

		cudaEventRecord(finish);
		cudaEventSynchronize(finish);

		cudaEventElapsedTime(&timeTotalGpu, start, finish);

		printf("The blocksize in CUDA is：%d\n", blockSize);
		writeToCSV(dataPoints, datapoints_gpu);
		writeToCSV(centroids, centroids_gpu);

	}

	// Read initialized data points and centroidsagain
	dataPoints = readCSV("datapoints_init.csv");
	centroids = readCSV("centroids_init.csv");

	if (omp) {
		omp_start = omp_get_wtime();

	KMeans_omp(dataPoints, centroids, maxIterations, ompthreads);

		omp_end = omp_get_wtime();
		timeTotalOmp = omp_end - omp_start; // seconds

		printf("The number of processor used in OpenMP is：%d\n", ompthreads);

		writeToCSV(dataPoints, datapoints_omp);
		writeToCSV(centroids, centroids_omp);
	}

	if (timing) {
		if (cpu) {
			printf("kmeans on the cpu took: %f seconds\n", timeTotalCpu);
		}
		if (gpu) {
			printf("kmeans using CUDA took: %f seconds\n", timeTotalGpu * 0.001f);
		}
		if (omp) {
			printf("kmeans using openmp took: %f seconds\n", timeTotalOmp);
		}
		
	}

    return 0;
}

