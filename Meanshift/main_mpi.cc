#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio> 
#include "meanshift_mpi.h"
#include "utils.h"


int main(int argc, char* argv[]) {
	std::vector<DataPoint> dataPoints;
	std::vector<DataPoint> centroids;
	std::string datapoints_mpi = "datapoints_mpi.csv";
	std::string centroids_mpi = "centroids_mpi.csv";

	bool timing = true;
	double timeTotalMPI = 0.0;
	double t1, t2;

	double windowSize = 10;
	double shiftEpsilon = 0.1;

	int c;
	while ((c = getopt(argc, argv, "ti:w:e:")) != -1) {
		switch (c) {
		case 't':
			timing = true; break;
		case 'w':
			windowSize = atoi(optarg); break;
		case 'e':
			shiftEpsilon = atoi(optarg); break;
		default:
			return -1;
		}
	}

	// Read initialized data points and centroids from csv files
	dataPoints = readCSV("datapoints_init.csv");
	centroids = readCSV("centroids_init.csv");

	int rank, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// Create an MPI data type to describe the DataPoint structure
	MPI_Datatype MPI_DataPoint;
	int blocklengths[3] = { 1, 1, 1 };
	MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	MPI_Aint offsets[3];
	offsets[0] = offsetof(DataPoint, at1);
	offsets[1] = offsetof(DataPoint, at2);
	offsets[2] = offsetof(DataPoint, cluster);
	MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_DataPoint);
	MPI_Type_commit(&MPI_DataPoint);

	if(rank == 0){
	    std::cout << "windowSize = " << windowSize << "\n";
	    std::cout << "shiftEpsilon = " << shiftEpsilon << "\n";
	    std::cout << "-------------------------------------\n";
	}

	// Initialize the data points and cluster centers
	t1 = MPI_Wtime();
	meanshift_mpi(dataPoints, centroids, windowSize, shiftEpsilon, rank, nprocs, MPI_COMM_WORLD, MPI_DataPoint);
	t2 = MPI_Wtime();

	timeTotalMPI = t2 - t1;

	if (rank == 0) {
		printf("The number of processor in MPI is：%d\n", nprocs);

		if (timing) {
			std::cout << "-------------------------------------\n";
			printf("kmeans using mpi took: %f seconds\n", timeTotalMPI);
		}

		for (size_t i = 0; i < centroids.size(); i++) {
			centroids[i].cluster = i;
		}

		writeToCSV(dataPoints, datapoints_mpi);
		writeToCSV(centroids, centroids_mpi);

	}

	MPI_Type_free(&MPI_DataPoint);
	MPI_Finalize();

	return 0;
}

