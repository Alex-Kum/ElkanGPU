#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKer(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void multip(const int* a, const int* b, int* c) {
    int i = threadIdx.x;
    if (i < 5) {
        c[i] = a[i] * b[i];
    }
}

/*__global__ void zweiD(const int* a, const int* b, int* c) {
    int i = threadIdx.x;
    if (i < 4) {
        c[i] = a[i] * b[i];
    }
}*/

__device__ void addV(double* x,  double* y, int dim) {
    double const* end = x + dim;
    while (x < end) {
        *(x++) += *(y++);
    }
}

__device__ void subV(double* x, double* y, int dim) {
    double const* end = x + dim;
    while (x < end) {
        *(x++) -= *(y++);
    }
}

__device__ double dist(const double* data, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * data[y * dim + i];
    }
    return result;
}

__device__ double distp2c(const double* data, const double* center, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * center[y * dim + i];
    }
    return result;
}

/*__global__ void dist2(const double* data, int x, int y, int dim, int n, double* res) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * data[y * dim + i];
    }
    *res = result;
}*/

__global__ void innerProd(double* centerCenterDist, double* s, const double* data, int dim, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
        //double distance = 2.0;
        centerCenterDist[index] = sqrt(distance) / 2.0;

        if (centerCenterDist[index] < s[c1]) {
            s[c1] = centerCenterDist[index];
        }
    }
}

__device__ double innerProdp2c(double* data, double* center, int x, int y, int dim) {
    return dist(data, x, x, dim) - 2 * distp2c(data, center, x, y, dim) + dist(center, y, y, dim);
}

//__device__ void updateBound( double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n) {
//        upper[i] += centerMovement[assignment[i]];
//        for (int j = 0; j < k; ++j) {
//            lower[i * numLowerBounds + j] -= centerMovement[j];
//        }
//    }
//}

__global__ void updateBound(double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
}

__global__ void elkanFun(double* data, double* center, unsigned short* assignment, double* lower, double* upper, 
    double* s, double* centerCenterDistDiv2, int** clusterSize, double* sumNewCenters, double* centerMovement, int k, int dim, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short closest = assignment[i];
        bool r = true;

        if (upper[i] > s[closest]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    //upper[i] = sqrt(pointCenterDist2(i, closest));
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest, dim));
                    lower[i * k + closest] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(pointCenterDist2(i, j));
                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                if (lower[i * k + j] < upper[i]) {
                    closest = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
        if (assignment[i] != closest) {
            unsigned short oldAssignment = assignment[i];
            --clusterSize[0][assignment[i]];
            ++clusterSize[0][closest];
            assignment[i] = closest;
            double* xp = data + i * dim;

            subV(sumNewCenters + oldAssignment * dim, xp, dim);
            addV(sumNewCenters + closest * dim, xp, dim);
        }


        /*int furthestMovingCenter = 0;
        for (int j = 0; j < k; ++j) {
            centerMovement[j] = 0.0;
            int totalClusterSize = 0;
                totalClusterSize += clusterSize[0][j];

            if (totalClusterSize > 0) {
                for (int d = 0; d < dim; ++d) {
                    double z = 0.0;
                    //z += (*sumNewCenters[0])(j, d);
                    z += sumNewCenters[j * dim + d];
                    z /= totalClusterSize;
                    //centerMovement[j] += (z - (*centers)(j, d)) * (z - (*centers)(j, d));//calculate distance
                    centerMovement[j] += (z - center[j * dim + d]) * (z - center[j * dim + d]);
                    //(*centers)(j, dim) = z; //update new centers
                    center[j * dim + d] = z;
                }
            }
            centerMovement[j] = sqrt(centerMovement[j]);

            if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
                furthestMovingCenter = j;
            }
        }
        bool converged = (0.0 == centerMovement[furthestMovingCenter]);
        if (!converged) {
            updateBound(lower, upper, centerMovement, assignment, numLowerBounds, k, n);
        }*/

    }
}