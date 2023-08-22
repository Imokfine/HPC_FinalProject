#include "kmeans_mpi.h"


// -------------------------------- MPI VER.-----------------------------------
void KMeans_mpi(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, int maxIterations, MPI_Comm comm, MPI_Datatype MPI_DataPoint) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculates the range of data points that each process needs to process
    int numPointsPerProcess = (rank != size - 1) ? dataPoints.size() / size : (dataPoints.size() / size) + (dataPoints.size() % size);
    int startPoint = rank * (dataPoints.size() / size);
    int endPoint = startPoint + numPointsPerProcess;

    int iterations = 0;
    bool centersUpdated = true;

    while (iterations < maxIterations && centersUpdated) {
        centersUpdated = false;
        std::vector<DataPoint> centroids_old = centroids;

        // --------------- Assign data points to clusters ---------------                  
        for (int i = startPoint; i < endPoint; i++) {
            auto& dataPoint = dataPoints[i];
            double minDistance = std::numeric_limits<double>::max(); // Initialize to the maximum value
            int assignedCluster = -1;

            for (size_t j = 0; j < centroids.size(); j++) {
                // Calculate the distance between the data point and the centroid (Euclidean distance)
                //double distance = std::sqrt(std::pow(dataPoint.at1 - centroids[j].at1, 2) + std::pow(dataPoint.at2 - centroids[j].at2, 2));
		DataPoint myDataPoint = dataPoints[i];
                DataPoint myCentroid  = centroids[j];
                double tmp1=myDataPoint.at1 - myCentroid.at1;
                double tmp2=myDataPoint.at2 - myCentroid.at2;
                double distance = tmp1*tmp1 + tmp2*tmp2;

                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = j;
                }
            }

            // Assign data points to the nearest cluster center
            dataPoint.cluster = assignedCluster;
        }

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                int length = (i != size - 1) ? dataPoints.size() / size : (dataPoints.size() / size) + (dataPoints.size() % size);
                MPI_Recv(&dataPoints[i * (dataPoints.size() / size)], length, MPI_DataPoint, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else {
            MPI_Send(&dataPoints[startPoint], numPointsPerProcess, MPI_DataPoint, 0, 0, MPI_COMM_WORLD);
        }

        MPI_Bcast(&dataPoints[0], dataPoints.size(), MPI_DataPoint, 0, MPI_COMM_WORLD);

        // Use MPI_Barrier to ensure that all processes have completed the allocation of data points
        MPI_Barrier(comm);


        // --------------- Update cluster centers --------------- 
        std::vector<int> clusterCounts(centroids.size(), 0); // Used to count the number of data points per cluster
        std::vector<double> sum1(centroids.size(), 0.0); // Used to calculate the at1 sum for each cluster
        std::vector<double> sum2(centroids.size(), 0.0); // Used to calculate the at2 sum for each cluster

        // Iterate through each data point and add its coordinate values to the sum of the corresponding clusters
        for (int i = startPoint; i < endPoint; i++) {
            int clusterIndex = dataPoints[i].cluster;
            clusterCounts[clusterIndex]++;
            sum1[clusterIndex] += dataPoints[i].at1;
            sum2[clusterIndex] += dataPoints[i].at2;
        }

        // Use MPI_Allreduce to aggregate clustering statistics for each process
        std::vector<int> globalClusterCounts(centroids.size(), 0);
        std::vector<double> globalSum1(centroids.size(), 0.0);
        std::vector<double> globalSum2(centroids.size(), 0.0);

        MPI_Allreduce(clusterCounts.data(), globalClusterCounts.data(), centroids.size(), MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(sum1.data(), globalSum1.data(), centroids.size(), MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(sum2.data(), globalSum2.data(), centroids.size(), MPI_DOUBLE, MPI_SUM, comm);


        // Calculates the new central coordinates for each cluster
        for (size_t i = 0; i < centroids.size(); i++) {
            if (globalClusterCounts[i] > 0) {
                centroids[i].at1 = globalSum1[i] / globalClusterCounts[i];
                centroids[i].at2 = globalSum2[i] / globalClusterCounts[i];
            }
        }
        centersUpdated = (centroids != centroids_old) ? true : false;


        iterations++;
    }
}
