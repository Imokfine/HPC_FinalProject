#include "kmeans.h"

__global__ void assignClusters(DataPoint* dataPoints, DataPoint* centroids, int numDataPoints, int numCentroids) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < numDataPoints) {
        double minDistance = 1e10; // Initialize to a large number
        int assignedCluster = -1;

        for (int i = 0; i < numCentroids; ++i) {
            //double distance = sqrt(pow(dataPoints[index].at1 - centroids[i].at1, 2) + pow(dataPoints[index].at2 - centroids[i].at2, 2));

	    DataPoint myDataPoint = dataPoints[index];
	    DataPoint myCentroid  = centroids[i];
	    double tmp1 = myDataPoint.at1 - myCentroid.at1;
   	    double tmp2 = myDataPoint.at2 - myCentroid.at2;
	    double distance = tmp1*tmp1 + tmp2*tmp2;

	    if (distance < minDistance) {
                minDistance = distance;
                assignedCluster = i;
            }
        }
        dataPoints[index].cluster = assignedCluster;
    }
}

__global__ void updateCentroids(DataPoint* dataPoints, DataPoint* centroids, int numDataPoints, int numCentroids) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < numCentroids) {
        double sumAt1 = 0.0;
        double sumAt2 = 0.0;
        int count = 0;

        for (int i = 0; i < numDataPoints; ++i) {
            if (dataPoints[i].cluster == index) {
                sumAt1 += dataPoints[i].at1;
                sumAt2 += dataPoints[i].at2;
                count++;
            }
        }
        if (count > 0) {
            centroids[index].at1 = sumAt1 / count;
            centroids[index].at2 = sumAt2 / count;
        }
    }
}

void KMeans_gpu(std::vector<DataPoint>& h_dataPoints, std::vector<DataPoint>& h_centroids, int maxIterations, int blockSize) {
    int iterations = 0;
    bool centersUpdated = true;

    int numDataPoints = h_dataPoints.size();
    int numCentroids = h_centroids.size();

    // Allocate device memory
    DataPoint* d_dataPoints;
    DataPoint* d_centroids;
    cudaMalloc(&d_dataPoints, numDataPoints * sizeof(DataPoint));
    cudaMalloc(&d_centroids, numCentroids * sizeof(DataPoint));

    // Copy data to device memory
    cudaMemcpy(d_dataPoints, h_dataPoints.data(), numDataPoints * sizeof(DataPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), numCentroids * sizeof(DataPoint), cudaMemcpyHostToDevice);

    std::vector<DataPoint> h_oldCentroids;
    while (iterations < maxIterations && centersUpdated){
	centersUpdated = false;
        h_oldCentroids = h_centroids;

        // Launch CUDA kernels
        int numBlocks = (numDataPoints + blockSize - 1) / blockSize;
        assignClusters << <numBlocks, blockSize >> > (d_dataPoints, d_centroids, numDataPoints, numCentroids);
        numBlocks = (numCentroids + blockSize - 1) / blockSize;
        updateCentroids << <numBlocks, blockSize >> > (d_dataPoints, d_centroids, numDataPoints, numCentroids);

        // Copy centroids back to host memory
        cudaMemcpy(h_centroids.data(), d_centroids, numCentroids * sizeof(DataPoint), cudaMemcpyDeviceToHost);
        for (int i = 0; i < numCentroids; i++) {
	    if (h_centroids[i].at1 != h_oldCentroids[i].at1 || h_centroids[i].at2 != h_oldCentroids[i].at2) {
     		    centersUpdated = true;
                break;
            }
        }

        iterations++;
    } 

    // Copy dataPoints back to host memory
    cudaMemcpy(h_dataPoints.data(), d_dataPoints, numDataPoints * sizeof(DataPoint), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dataPoints);
    cudaFree(d_centroids);
}

void KMeans(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, int maxIterations) {

    int iterations = 0;
    bool centersUpdated = true;

    while (iterations < maxIterations && centersUpdated) {
        centersUpdated = false;
        std::vector<DataPoint> centroids_old = centroids;

        // --------------- Assign data points to clusters --------------- 
        for (auto& dataPoint : dataPoints) {
            double minDistance = std::numeric_limits<double>::max(); // Initialize to the maximum value
            int assignedCluster = -1;

            for (size_t i = 0; i < centroids.size(); i++) {
                // Calculate the distance between data points and centroids (Euclidean distance)
                //double distance = std::sqrt(std::pow(dataPoint.at1 - centroids[i].at1, 2) + std::pow(dataPoint.at2 - centroids[i].at2, 2));

                DataPoint myCentroid  = centroids[i];
                double tmp1 = dataPoint.at1 - myCentroid.at1;
                double tmp2 = dataPoint.at2 - myCentroid.at2;
                double distance = tmp1*tmp1 + tmp2*tmp2;

                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = i;
                }
            }

            // Assign data points to the nearest cluster center
            dataPoint.cluster = assignedCluster;
        }

        // --------------- Update cluster centers --------------- 
        std::vector<int> clusterCounts(centroids.size(), 0); // Used to count the number of data points per cluster
        std::vector<double> sum1(centroids.size(), 0.0); // Used to calculate the at1 sum for each cluster
        std::vector<double> sum2(centroids.size(), 0.0); // Used to calculate the at2 sum for each cluster

        // Iterate through each data point and add its coordinate values to the sum of the corresponding clusters
        for (const auto& dataPoint : dataPoints) {
            int clusterIndex = dataPoint.cluster;
            clusterCounts[clusterIndex]++;
            sum1[clusterIndex] += dataPoint.at1;
            sum2[clusterIndex] += dataPoint.at2;
        }

        // Calculates the new central coordinates for each cluster
        for (size_t i = 0; i < centroids.size(); i++) {
            if (clusterCounts[i] > 0) {
                centroids[i].at1 = sum1[i] / clusterCounts[i];
                centroids[i].at2 = sum2[i] / clusterCounts[i];
            }
        }

        // Check if centroids has updated
        for (size_t i = 0; i < centroids.size(); i++) {
            if (centroids != centroids_old) {
                centersUpdated = true;
            }
        }

        iterations++;
    }
}

void KMeans_omp(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, int maxIterations, int ompthreads) {

    omp_set_num_threads(ompthreads);

    int iterations = 0;
    bool centersUpdated = true;

    while (iterations < maxIterations && centersUpdated) {
        centersUpdated = false;
        std::vector<DataPoint> centroids_old = centroids;

        // --------------- Assign data points to clusters ---------------
        #pragma omp parallel for
        for (auto& dataPoint : dataPoints) {
            double minDistance = std::numeric_limits<double>::max(); // Initialize to the maximum value
            int assignedCluster = -1;

            for (size_t i = 0; i < centroids.size(); i++) {
                // Calculate the distance between data points and centroids (Euclidean distance)
                //double distance = std::sqrt(std::pow(dataPoint.at1 - centroids[i].at1, 2) + std::pow(dataPoint.at2 - centroids[i].at2, 2));
                DataPoint myCentroid  = centroids[i];
                double tmp1 = dataPoint.at1 - myCentroid.at1;
                double tmp2 = dataPoint.at2 - myCentroid.at2;
                double distance = tmp1*tmp1 + tmp2*tmp2;

                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = i;
                }
            }

            // Assign data points to the nearest cluster center
            dataPoint.cluster = assignedCluster;
        }

        // --------------- Update cluster centers ---------------
        std::vector<int> clusterCounts(centroids.size(), 0);
        std::vector<double> sum1(centroids.size(), 0.0);
        std::vector<double> sum2(centroids.size(), 0.0);


        // Iterate through each data point and add its coordinate values to the sum of the corresponding clusters
        for (const auto& dataPoint : dataPoints) {
            int clusterIndex = dataPoint.cluster;
            #pragma omp atomic
            clusterCounts[clusterIndex]++;
            #pragma omp atomic
            sum1[clusterIndex] += dataPoint.at1;
            #pragma omp atomic
            sum2[clusterIndex] += dataPoint.at2;
        }

        #pragma omp parallel for
        // Calculates the new central coordinates for each cluster
        for (size_t i = 0; i < centroids.size(); i++) {
            if (clusterCounts[i] > 0) {
                centroids[i].at1 = sum1[i] / clusterCounts[i];
                centroids[i].at2 = sum2[i] / clusterCounts[i];
            }
        }

        #pragma omp parallel for
        // Check if centroids has updated
        for (size_t i = 0; i < centroids.size(); i++) {
            if (centroids != centroids_old) {
                centersUpdated = true;
            }
        }

        iterations++;
    }
}
