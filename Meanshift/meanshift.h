#ifndef MEANSHIFT_GPU_H
#define MEANSHIFT_GPU_H

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

void meanshift(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids,
	double window_size, double shift_epsilon);
void meanshift_omp(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids,
	double windowSize, double shiftEpsilon, int ompthreads);
__global__ void meanshiftKernel(DataPoint* dataPoints, int numPoints, double windowSize);
__device__ bool globalStop = true;
void meanshift_gpu(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, 
	double windowSize, double shiftEpsilon, int blockSize);

#endif
