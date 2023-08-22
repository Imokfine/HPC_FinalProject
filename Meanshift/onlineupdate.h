#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio> 
#include <omp.h> 
#include "utils.h"

void meanshift_ou(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, double windowSize, double shiftEpsilon);
