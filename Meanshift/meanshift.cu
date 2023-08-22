#include "meanshift.h"
#include "utils.h"

// Batch updates
void meanshift(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, double windowSize, double shiftEpsilon) {
    bool stop = false;
    std::vector<DataPoint> shiftedPoints(dataPoints.size());
    double windowSizeSquared = windowSize*windowSize; 
    double shiftEpsilonSquared = shiftEpsilon*shiftEpsilon; 
    double distanceSquared;
    double weight;
    double sum1,sum2;


    int count = 0;
    while (!stop) {
        count++;
        stop = true;

        // The outer loop iterates over dataPoints
        for (int i = 0; i < dataPoints.size(); i++) {
            double shift_x = 0.0;
            double shift_y = 0.0;
            double scale = 0.0;


            double myDataPointIdxAt1 = dataPoints[i].at1;
            double myDataPointIdxAt2 = dataPoints[i].at2;
            double myDataPointIAt1,myDataPointIAt2;

            // The inner loop iterates over dataPoints to calculate the shift
            for (const auto& point : dataPoints) {
                // Calculate Euclidean distance
	// double distance = sqrt(pow(dataPoints[idx].at1 - dataPoints[i].at1, 2) + pow(dataPoints[idx].at2 - dataPoints[i].at2, 2));        // Original code

                myDataPointIAt1=point.at1;
                myDataPointIAt2=point.at2;
	

                distanceSquared = ( myDataPointIdxAt1 - myDataPointIAt1) * ( myDataPointIdxAt1 - myDataPointIAt1) + ( myDataPointIdxAt2 - myDataPointIAt2) * ( myDataPointIdxAt2 - myDataPointIAt2);

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

//        double shift_distance = std::sqrt(std::pow(myDataPointIdxAt1 - shiftedPoints[idx].at1, 2) + std::pow(myDataPointIdxAt2 - shiftedPoints[idx].at2, 2)); // Original code
        sum1 = myDataPointIdxAt1 - shift_x;	
        sum2 = myDataPointIdxAt2 - shift_y;	
        double shift_distanceSquared = sum1*sum1 + sum2*sum2; // This seems to run faster than the sqrt(pow+pow)

//      if (shift_distance > shiftEpsilon) { // Original code
        if (shift_distanceSquared > shiftEpsilonSquared) {
                stop = false;
            }
        }

        if (!stop) {
            dataPoints = shiftedPoints;
        }
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
}

void meanshift_omp(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, double windowSize, double shiftEpsilon, int ompthreads) {

    omp_set_num_threads(ompthreads);

    bool stop = false;
    std::vector<DataPoint> shiftedPoints(dataPoints.size());
    double windowSizeSquared = windowSize * windowSize;
    double shiftEpsilonSquared = shiftEpsilon * shiftEpsilon;
    double distanceSquared;
    double weight;
    double sum1, sum2;

    double myDataPointIdxAt1, myDataPointIdxAt2;
    double myDataPointIAt1, myDataPointIAt2;

    while (!stop) {
        stop = true;
#pragma omp parallel for private(distanceSquared, weight, sum1, sum2)
        // The outer loop iterates over dataPoints
        for (int i = 0; i < dataPoints.size(); i++) {
            double shift_x = 0.0;
            double shift_y = 0.0;
            double scale = 0.0;
            
                myDataPointIdxAt1 = dataPoints[i].at1;
                myDataPointIdxAt2 = dataPoints[i].at2;
 
            // The inner loop iterates over dataPoints to calculate the shift
//#pragma omp parallel for reduction(+:shift_x, shift_y, scale)
            for (const auto& point : dataPoints) {
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

        if (!stop) {
            dataPoints = shiftedPoints;
        }
    }

    int cluster_id = 0;
#pragma omp parallel for
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
#pragma omp critical
            centroids.push_back(p);

            cluster_id++;
        }
    }
}


__global__ void meanshiftKernel(DataPoint* dataPoints, DataPoint* shiftedPoints, int numPoints, double windowSize, double shiftEpsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Use the shared memory to save register space, as we are dealing with doubles and therefore we can run out of register space quickly, specially for large block sizes
        double windowSizeSquared;
        double shiftEpsilonSquared;

    if (idx < numPoints) {
        double shift_x = 0.0;
        double shift_y = 0.0;
        double scale = 0.0;

                windowSizeSquared = windowSize*windowSize;                      // There should be no need to sync since all threads write the same value - if windowSize was brought already squared from the host, this multiplication would not be needed
                shiftEpsilonSquared = shiftEpsilon*shiftEpsilon;        // There should be no need to sync since all threads write the same value - if shiftEpsilon was brought already squared in the host, this multiplication would not be needed

                double myDataPointIdxAt1 = dataPoints[idx].at1;
                double myDataPointIdxAt2 = dataPoints[idx].at2;
                double myDataPointIAt1,myDataPointIAt2;
                double distanceSquared;
                double weight;
                double sum1,sum2;

        for (int i = 0; i < numPoints; i++) {

//            double distance = sqrt(pow(dataPoints[idx].at1 - dataPoints[i].at1, 2) + pow(dataPoints[idx].at2 - dataPoints[i].at2, 2));        // Original code

                        myDataPointIAt1=dataPoints[i].at1;
                        myDataPointIAt2=dataPoints[i].at2;

            distanceSquared = ( myDataPointIdxAt1 - myDataPointIAt1) * ( myDataPointIdxAt1 - myDataPointIAt1) + ( myDataPointIdxAt2 - myDataPointIAt2) * ( myDataPointIdxAt2 - myDataPointIAt2);

            if (distanceSquared <= windowSizeSquared) {
//              double weight = exp(-(distance * distance) / (windowSize * windowSize)); // Original code
                 weight = exp(-(distanceSquared) / (windowSizeSquared)); // No need to multiply distance and windowSize, so we save compute on the sqrt of distance

//              shift_x += dataPoints[i].at1 * weight;  // Original code
//              shift_y += dataPoints[i].at2 * weight;  // Original code
                shift_x += myDataPointIAt1 * weight; // myDataPointIAt1 is already in a register
                shift_y += myDataPointIAt2 * weight; // myDataPointIAt2 is already in a register
                scale += weight;
            }
        }

        shift_x /= scale;
        shift_y /= scale;

        shiftedPoints[idx].at1 = shift_x;
        shiftedPoints[idx].at2 = shift_y;

//        double shift_distance = std::sqrt(std::pow(myDataPointIdxAt1 - shiftedPoints[idx].at1, 2) + std::pow(myDataPointIdxAt2 - shiftedPoints[idx].at2, 2)); // Original code
                sum1 = myDataPointIdxAt1 - shift_x;     // shift_x and shiftedPoints[idx].at1 have the same values but shiftedPoints[idx].at1 is in the global memory and shift_x in the registers
                sum2 = myDataPointIdxAt2 - shift_y;     // shift_y and shiftedPoints[idx].at2 have the same values but shiftedPoints[idx].at2 is in the global memory and shift_y in the registers
        double shift_distanceSquared = sum1*sum1 + sum2*sum2; // This seems to run faster than the sqrt(pow+pow)

//      if (shift_distance > shiftEpsilon) { // Original code
        if (shift_distanceSquared > shiftEpsilonSquared) {
            globalStop = false;
        }
    }
}

// CUDA function to parallelize the meanshift algorithm
void meanshift_gpu(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, double windowSize, double shiftEpsilon, int blockSize) {
    bool stop = false;
    std::vector<DataPoint> shiftedPoints(dataPoints.size());
    int numPoints = dataPoints.size();
    size_t dataSize = numPoints * sizeof(DataPoint);

    // Copy dataPoints to the GPU memory
    DataPoint* d_dataPoints;
    cudaMalloc(&d_dataPoints, dataSize);

    DataPoint* d_shiftedPoints;
    cudaMalloc(&d_shiftedPoints, dataSize);

    // Set the number of threads per block and launch the CUDA kernel
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    int count = 0;
    while (!stop) {
        count++;
        stop = true;

        cudaMemcpy(d_dataPoints, dataPoints.data(), dataSize, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(globalStop, &stop, sizeof(bool), 0, cudaMemcpyHostToDevice);

        // Launch the CUDA kernel to calculate shifted points in parallel
        meanshiftKernel << <numBlocks, blockSize >> > (d_dataPoints, d_shiftedPoints, numPoints, windowSize, shiftEpsilon);
        cudaDeviceSynchronize();

        // Copy the results (shifted points) back to the CPU
        cudaMemcpy(dataPoints.data(), d_dataPoints, dataSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(shiftedPoints.data(), d_shiftedPoints, dataSize, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&stop, globalStop, sizeof(bool), 0, cudaMemcpyDeviceToHost);

        if (!stop) {
            dataPoints = shiftedPoints;
        }

    }

    // Free GPU memory
    cudaFree(d_dataPoints);
    cudaFree(d_shiftedPoints);

    int cluster_id = 0;
    for (auto& p : dataPoints) {
        bool found = false;
        for (auto& centroid : centroids) {
            double centroid_distance = sqrt(pow(p.at1 - centroid.at1, 2) + pow(p.at2 - centroid.at2, 2));
            if (centroid_distance <= shiftEpsilon) {
                found = true;
                p.cluster = centroid.cluster;
                break;
            }
        }

        if (!found) {
            p.cluster = cluster_id;
            centroids.push_back(p);
            cluster_id++;
        }
    }
}

