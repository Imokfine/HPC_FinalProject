#include <mpi.h>
#include "meanshift_mpi.h"

void meanshift_mpi(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, double windowSize, double shiftEpsilon,
    int rank, int size, MPI_Comm comm, MPI_Datatype MPI_DataPoint) {
    bool stop = false;

    // Calculates the range of data points that each process needs to process
    int numPointsPerProcess = (rank != size - 1) ? dataPoints.size() / size : (dataPoints.size() / size) + (dataPoints.size() % size);
    int startPoint = rank * (dataPoints.size() / size);
    int endPoint = startPoint + numPointsPerProcess;

    std::vector<DataPoint> shiftedPoints(dataPoints.size());
    double windowSizeSquared = windowSize * windowSize;
    double shiftEpsilonSquared = shiftEpsilon * shiftEpsilon;
    double distanceSquared;
    double weight;
    double sum1, sum2;

    int count = 0;
    while (!stop) {
        count++;
        stop = true;

        // Compute shift for each data point in the sub-vector
        for (int i = startPoint; i < endPoint; i++) {
            double shift_x = 0.0;
            double shift_y = 0.0;
            double scale = 0.0;

            double myDataPointIdxAt1 = dataPoints[i].at1;
            double myDataPointIdxAt2 = dataPoints[i].at2;
            double myDataPointIAt1, myDataPointIAt2;

            // The inner loop iterates over dataPoints to calculate the shift
            for (const auto& point : dataPoints) {
                // Calculate Euclidean distance
                 // double distance = sqrt(pow(dataPoints[idx].at1 - dataPoints[i].at1, 2) + pow(dataPoints[idx].at2 - dataPoints[i].at2, 2));        // Original code

                myDataPointIAt1 = point.at1;
                myDataPointIAt2 = point.at2;


                distanceSquared = (myDataPointIdxAt1 - myDataPointIAt1) * (myDataPointIdxAt1 - myDataPointIAt1) + (myDataPointIdxAt2 - myDataPointIAt2) * (myDataPointIdxAt2 - myDataPointIAt2);

                if (distanceSquared <= windowSizeSquared) {
                    //              double weight = exp(-(distance * distance) / (windowSize * windowSize)); // Original code
                    weight = exp(-(distanceSquared) / (windowSizeSquared)); // No need to multiply distance and windowSize, so we save compute on the sqrt of distance

                    //              shift_x += dataPoints[i].at1 * weight;  // Original code
                    //              shift_y += dataPoints[i].at2 * weight;  // Original code
                    shift_x += myDataPointIAt1 * weight;
                    shift_y += myDataPointIAt2 * weight;
                    scale += weight;
                }
            }

            shift_x /= scale;
            shift_y /= scale;

            shiftedPoints[i].at1 = shift_x;
            shiftedPoints[i].at2 = shift_y;

            //      double shift_distance = std::sqrt(std::pow(myDataPointIdxAt1 - shiftedPoints[idx].at1, 2) + std::pow(myDataPointIdxAt2 - shiftedPoints[idx].at2, 2)); // Original code
            sum1 = myDataPointIdxAt1 - shift_x;
            sum2 = myDataPointIdxAt2 - shift_y;
            double shift_distanceSquared = sum1 * sum1 + sum2 * sum2; // This seems to run faster than the sqrt(pow+pow)

            //      if (shift_distance > shiftEpsilon) { // Original code
            if (shift_distanceSquared > shiftEpsilonSquared) {
                stop = false;
            }
        }

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                int length = (i != size - 1) ? dataPoints.size() / size : (dataPoints.size() / size) + (dataPoints.size() % size);
                MPI_Recv(&shiftedPoints[i * (dataPoints.size() / size)], length, MPI_DataPoint, i, 0, comm, MPI_STATUS_IGNORE);
            }
        }
        else {
            MPI_Send(&shiftedPoints[startPoint], numPointsPerProcess, MPI_DataPoint, 0, 0, comm);
        }

        // Every process checks if it has to continue
        int local_continue = !stop;
        int global_continue;
        MPI_Allreduce(&local_continue, &global_continue, 1, MPI_INT, MPI_LOR, comm);

        stop = !global_continue;

        if (!stop) {
            dataPoints = shiftedPoints;
        }

        // Broadcast updated dataPoints to all processes
        MPI_Bcast(&dataPoints[0], dataPoints.size(), MPI_DataPoint, 0, comm);
   
    }

    int cluster_id = 0;
    for (auto& p : dataPoints) {
        bool found = false;
        for (auto& centroid : centroids) {
            double centroid_distance = std::sqrt(std::pow(p.at1 - centroid.at1, 2) + std::pow(p.at2 - centroid.at2, 2));
            // If the centroid_distance is less than shiftEpsilon, the point belongs to this cluster and set found true
            if (centroid_distance <= shiftEpsilon) {
                found = true;
                p.cluster = centroid.cluster;
                break;
            }
        }

        // If there is no centroid, set the current point to the centroid
        if (!found) {

            p.cluster = cluster_id;
            centroids.push_back(p);

            cluster_id++;
        }
    }

    if (rank == 0) {
        std::cout << "mpi_count = " << count << "\n";
    }
}



