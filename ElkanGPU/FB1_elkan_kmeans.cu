/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "FB1_elkan_kmeans.h"
#include "general_functions.h"
//#include "gpufunctions.h"
#include <cmath>
#include <chrono>

#define GPUA 0
#define GPUB 0
#define GPUC 0

void FB1_ElkanKmeans::update_center_dists(int threadId) {
#if GPUA
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        s[c1] = std::numeric_limits<double>::max();
    }

    int n = *centers->n * *centers->n;
    int blockSize = 3 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    innerProd << <numBlocks, blockSize >> > (centerCenterDistDiv2, s, centers->data, *centers->d, *centers->n);
    //dist2 << <1, 10 >> > (data, 0, 1, 5, 0, res);
    cudaDeviceSynchronize();
#else

    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<double>::max();

            for (int c2 = 0; c2 < k; ++c2) {
                if (c1 != c2) {
                    centerCenterDistDiv2[c1 * k + c2] = sqrt(centerCenterDist2(c1, c2)) / 2.0;
                    if (centerCenterDistDiv2[c1 * k + c2] < s[c1]) {
                        s[c1] = centerCenterDistDiv2[c1 * k + c2];
                    }
                }
            }
        }
    }
#endif
}

int FB1_ElkanKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);
    bool* converged;
    cudaMallocManaged(&converged, 1 * sizeof(bool));
    cudaMallocManaged(&ub_old, n * sizeof(double));
    std::fill(ub_old, ub_old + n, std::numeric_limits<double>::max());
    cudaMallocManaged(&lower, (n * k) * sizeof(double));
    std::fill(lower, lower + n * k, 0.0);
    cudaMallocManaged(&oldcenter2newcenterDis, (k * k) * sizeof(double));
    std::fill(oldcenter2newcenterDis, oldcenter2newcenterDis + k * k, 0.0);
    cudaMallocManaged(&oldcenters, (k * d) * sizeof(double));
    std::fill(oldcenters, oldcenters + k * d, 0.0);
    *converged = false;

    while ((iterations < maxIterations) && !(*converged)) {
        //std::cout << "hier" << std::endl;
        ++iterations;
#if Countdistance
        int numberdistances = 0;
#endif
        update_center_dists(threadId);

        synchronizeAllThreads();

#if GPUC
        int n = endNdx;
        int blockSize = 3 * 32;
        int numBlocks = (n + blockSize - 1) / blockSize;
        //elkanFun << <numBlocks, blockSize >> > (x->data, centers->data, assignment, lower, upper, s, centerCenterDistDiv2, clusterSize, sumNewCenters[threadId]->data, centerMovement, k, d, endNdx);
        elkanFunFB << <numBlocks, blockSize >> > (x->data, centers->data, assignment, lower, upper, s, centerCenterDistDiv2, oldcenter2newcenterDis, ub_old, oldcenters, clusterSize, sumNewCenters[threadId]->data, centerMovement, k, d, endNdx, converged, numLowerBounds);
        cudaDeviceSynchronize();

#else
        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;

            if (upper[i] <= s[closest]) {
                continue;
            }

            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }

                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest * k + j]) { continue; }

#if Countdistance
                numberdistances++;
#endif
                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));
                    lower[i * k + closest] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j]) || upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(pointCenterDist2(i, j));

                if (lower[i * k + j] < upper[i]) {
                    closest = j;
                    upper[i] = lower[i * k + j];
                }
            }
            if (assignment[i] != closest) {
                changeAssignment(i, closest, threadId);
            }
        }
#endif

#if Countdistance
        std::cout << numberdistances << "\n";
#endif

        //verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6

        int furthestMovingCenter = move_centers_newbound(oldcenters, oldcenter2newcenterDis);
        *converged = (0.0 == centerMovement[furthestMovingCenter]);
        
        if (!(*converged)) {
            update_bounds(startNdx, endNdx);
        }
        else {
            //std::cout << iterations << "\n";
        }



    }
    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
}

void FB1_ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    int n = endNdx;
    int blockSize = 3 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    updateBoundFB << <numBlocks, blockSize >> > (lower, upper, ub_old, centerMovement, assignment, numLowerBounds, k, endNdx);
    cudaDeviceSynchronize();
#else
    for (int i = startNdx; i < endNdx; ++i) {
        ub_old[i] = upper[i];
    }

    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
#endif

    
}

void FB1_ElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    //centerCenterDistDiv2 = new double[k * k];
    cudaMallocManaged(&centerCenterDistDiv2, (k * k) * sizeof(double));
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);

}

void FB1_ElkanKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    cudaFree(centerCenterDistDiv2);
    cudaFree(lower);
    cudaFree(ub_old);
    cudaFree(oldcenters);
    //delete[] centerCenterDistDiv2;
    //centerCenterDistDiv2 = NULL;
    //delete [] oldcenterCenterDistDiv2;
    //oldcenterCenterDistDiv2 = NULL;
    delete centers;
    centers = NULL;
}

int FB1_ElkanKmeans::move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis) {

    int furthestMovingCenter = 0;
    for (int j = 0; j < k; ++j) {
        centerMovement[j] = 0.0;
        int totalClusterSize = 0;
        double old = 0;
        for (int t = 0; t < numThreads; ++t) {
            totalClusterSize += clusterSize[t][j];
        }
        if (totalClusterSize > 0) {
            for (int dim = 0; dim < d; ++dim) {
                double z = 0.0;
                for (int t = 0; t < numThreads; ++t) {
                    z += (*sumNewCenters[t])(j, dim);
                }
                z /= totalClusterSize;
                centerMovement[j] += (z - (*centers)(j, dim)) * (z - (*centers)(j, dim));//calculate distance
                oldcenters[j * d + dim] = (*centers)(j, dim);
                (*centers)(j, dim) = z;
            }
        }
        centerMovement[j] = sqrt(centerMovement[j]);

        if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
            furthestMovingCenter = j;
        }
    }

    for (int c1 = 0; c1 < k; ++c1) {
        for (int c2 = 0; c2 < k; ++c2)
            if (c1 != c2) {
                oldcenter2newcenterDis[c1 * k + c2] = 0.0;
                for (int dim = 0; dim < d; ++dim) {
                    oldcenter2newcenterDis[c1 * k + c2] += (oldcenters[c1 * d + dim] - (*centers)(c2, dim)) * (oldcenters[c1 * d + dim] - (*centers)(c2, dim));
                }
                oldcenter2newcenterDis[c1 * k + c2] = sqrt(oldcenter2newcenterDis[c1 * k + c2]);
            }
    }
#ifdef COUNT_DISTANCES
    numDistances += k;
#endif

    return furthestMovingCenter;
}
