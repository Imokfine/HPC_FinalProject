#ifndef KMEANS_MPI_H
#define KMEANS_MPI_H

#include <mpi.h>
#include "utils.h"


// mpi version
void KMeans_mpi(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, int maxIterations,
	MPI_Comm comm, MPI_Datatype MPI_DataPoint);

#endif
