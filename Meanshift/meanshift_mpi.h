#ifndef MEANSHIFT_MPI_H
#define MEANSHIFT_MPI_H

#include <mpi.h>
#include "utils.h"


// mpi version
void meanshift_mpi(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids,
	double windowSize, double shiftEpsilon, int rank, int size, MPI_Comm comm, MPI_Datatype MPI_DataPoint);

#endif
