#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"


__global__ void assignClusters(DataPoint* dataPoints, DataPoint* centroids, int numDataPoints, int numCentroids);
__global__ void updateCentroids(DataPoint* dataPoints, DataPoint* centroids, int numDataPoints, int numCentroids);
void KMeans_gpu(std::vector<DataPoint>& h_dataPoints, std::vector<DataPoint>& h_centroids, int maxIterations, int blockSize);
void KMeans(std::vector<DataPoint>& h_dataPoints, std::vector<DataPoint>& h_centroids, int maxIterations);
void KMeans_omp(std::vector<DataPoint>& h_dataPoints, std::vector<DataPoint>& h_centroids, int maxIterations, int ompthreads);

#endif
