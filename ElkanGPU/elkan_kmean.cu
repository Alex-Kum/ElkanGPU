/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "elkan_kmean.h"
#include "gpufunctions.h"
#include "general_functions.h"
#include <cmath>
#include <chrono>
 //using namespace std::chrono;

#define Time 0
#define Countdistance 0
#define GPUA 0
#define GPUB 0
#define GPUC 0
#define GPUD 0

void ElkanKmeans::update_center_dists(int threadId) {
    
#if GPUA
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        s[c1] = std::numeric_limits<double>::max();
    }

    int n = *centers->n * *centers->n;
    int blockSize = 3*32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    innerProd<<<numBlocks, blockSize>>> (centerCenterDistDiv2, s, centers->data, *centers->d, *centers->n);
    //dist2 << <1, 10 >> > (data, 0, 1, 5, 0, res);
    cudaDeviceSynchronize();
#else

    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<double>::max();

            for (int c2 = 0; c2 < k; ++c2) {
                // we do not need to consider the case when c1 == c2 as centerCenterDistDiv2[c1*k+c1]
                // is equal to zero from initialization, also this distance should not be used for s[c1]
                if (c1 != c2) {
                    // divide by 2 here since we always use the inter-center
                    // distances divided by 2
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

int ElkanKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);
    //lower = new double[n * k];
    cudaMallocManaged(&lower, (n * k) * sizeof(double));
    std::fill(lower, lower + n * k, 0.0);
    //x->print();
    //auto start_time = std::chrono::high_resolution_clock::now();
#if Time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elkan_time{};
    std::chrono::duration<double> elapsed_seconds = end - start;
#endif
    while ((iterations < maxIterations) && !converged) {
#if Time
        start_time = std::chrono::high_resolution_clock::now();
        start = std::chrono::system_clock::now();
#endif
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
        elkanFun<<<numBlocks, blockSize>>> (x->data, centers->data, assignment, lower, upper, s, centerCenterDistDiv2, clusterSize, sumNewCenters[threadId]->data, centerMovement, k, d, endNdx);
        cudaDeviceSynchronize();
        
#else

        for (int i = startNdx; i < endNdx; ++i) {
            //std::cout << d << "\n";
            unsigned short closest = assignment[i];
            bool r = true;

            if (upper[i] <= s[closest]) {
                continue;
            }

            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest * k + j]) { continue; }
#if Countdistance
                numberdistances++;
#endif
                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));
                    lower[i * k + closest] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j])) {
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
#if Time
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "\n";
        total_elkan_time += (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time));
#endif
        //verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        synchronizeAllThreads();
//#if GPUD

//#else
    if (threadId == 0) {
        int furthestMovingCenter = move_centers();
        converged = (0.0 == centerMovement[furthestMovingCenter]);
    }
//#endif


        synchronizeAllThreads();
        //total_elkan_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time));
        if (!converged) {
            update_bounds(startNdx, endNdx);
        }
        else {
            std::cout << "Iterations: " << iterations << "\n";
//#if Time
//            std::cout << total_elkan_time.count() << "\n";
//#endif
        }
        synchronizeAllThreads();


    }

    return iterations;
}

void ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    int n = endNdx;
    int blockSize = 3 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    updateBound <<<numBlocks, blockSize>>> (lower,upper,centerMovement,assignment,numLowerBounds,k,endNdx);    
    cudaDeviceSynchronize();
#else
    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
#endif
}

void ElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    //centerCenterDistDiv2 = new double[k * k];
    cudaMallocManaged(&centerCenterDistDiv2, (k * k) * sizeof(double));
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
}

void ElkanKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    //delete[] centerCenterDistDiv2;
    cudaFree(centerCenterDistDiv2);
    cudaFree(lower);
    centerCenterDistDiv2 = NULL;
    //delete centers;
    //centers = NULL;
}

/*float* arr1;
    float* arr2;
    float* arr3;
    cudaMallocManaged(&arr1, 5 * sizeof(float));
    cudaMallocManaged(&arr2, 5 * sizeof(float));
    cudaMallocManaged(&arr3, 5 * sizeof(float));
    double* data;
    double* res;
    cudaMallocManaged(&data, 10 * sizeof(double));
    cudaMallocManaged(&res, 1 * sizeof(double));
    for (int i = 0; i < 5; i++)
        arr1[i] = i;

    for (int i = 0; i < 5; i++)
        arr2[i] = 5 - i;
    arr2[2] = 2;
    arr2[3] = 3;

    for (int i = 0; i < 5; i++) {
        data[i] = arr1[i];
    }
    for (int i = 0; i < 5; i++) {
        data[i + 5] = arr2[i];
    }
    for (int i = 0; i < 10; i++)
        std::cout << "i: " << i << " " << data[i] << std::endl;
         //dist2 << <1, 10 >> > (data, 0, 1, 5, 0, res);
    cudaDeviceSynchronize();
    std::cout << "ERGEBNIS: " << *res << std::endl;
    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(arr3);
    
    */