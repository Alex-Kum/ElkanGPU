/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

// -lineinfo  cuda c++ comand line

#include "elkan_kmean.h"
#include "gpufunctions.h"
#include "general_functions.h"
#include <cmath>
#include <chrono>
 //using namespace std::chrono;

#define Time 0
#define Countdistance 0

#define GPUALL 0
#if GPUALL
    #define GPUA 1
    #define GPUB 1
    #define GPUC 1
#else
    #define GPUA 0
    #define GPUB 0
    #define GPUC 1
#endif

#define GPUD 0

void ElkanKmeans::update_center_dists(int threadId) {
    
#if GPUA
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        s[c1] = std::numeric_limits<double>::max();
    }

    int n = centers->n * centers->n;
    int blockSize = 2*32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    int* test = new int[n];
    int* d_test;
    cudaMalloc(&d_test, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        test[i] = i % 100;
    }
    cudaMemcpy(d_test, test, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centerCenterDistDiv2, centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyHostToDevice);
    innerProd<<<numBlocks, blockSize>>> (centerCenterDistDiv2, s, centers->data, centers->d, centers->n, d_test);
    cudaDeviceSynchronize();
    cudaMemcpy(centers->data, x->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(test, d_test, n * sizeof(int), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < n; i++) {
        //count += test[i];
        std::cout << "i: " << i << " -> " << test[i] << std::endl;
    }
    //std::cout << "COUNT A: " << count << std::endl;
    //dist2 << <1, 10 >> > (data, 0, 1, 5, 0, res);
    

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
    std::cout << "run thread start" << std::endl;
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);
    std::cout << "endNdx : " << endNdx << std::endl;
   // bool* converged = new bool[1];
   /// bool* d_converged;
    //auto e = cudaMalloc(&d_converged, 1 * sizeof(bool));
    //if (e != cudaSuccess) {
   //     std::cout << "cudaMalloc failed (converged)" << std::endl;
    //}

    int* test = new int[endNdx];
    int* d_test;
    cudaMalloc(&d_test, endNdx * sizeof(int));

    unsigned short* closest2 = new unsigned short[endNdx];
    unsigned short* d_closest2;
    auto f = cudaMalloc(&d_closest2, endNdx * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (closest2)" << std::endl;
    }

    lower = new double[n * k];
    auto g = cudaMalloc(&d_lower, (n * k) * sizeof(double));
    if (g != cudaSuccess) {
        std::cout << "cudaMalloc failed (lower)" << std::endl;
    }
    std::fill(lower, lower + n * k, 0.0);
   converged = false;
    //auto start_time = std::chrono::high_resolution_clock::now();
#if Time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elkan_time{};
    std::chrono::duration<double> elapsed_seconds = end - start;
#endif
    while ((iterations < maxIterations) && !(converged)) {
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
       
        for (int i = 0; i < n; i++) {
            test[i] = i % 100;
            //closest2[i] = assignment[i];
        }
        int n = endNdx;
        int blockSize = 2 * 32;
        int numBlocks = (n + blockSize - 1) / blockSize;
        cudaMemcpy(x->d_data, x->data, (n*d) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(centers->d_data, centers->data, (k*d) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower, lower, (n * k) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, s, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centerCenterDistDiv2, centerCenterDistDiv2, (k*k) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_test, test, n * sizeof(int), cudaMemcpyHostToDevice);
                //cudaMemcpy(d_clusterSize, clusterSize, k * sizeof(int), cudaMemcpyHostToDevice);
                //cudaMemcpy(sumNewCenters[threadId]->d_data, sumNewCenters[threadId]->data, k*d * sizeof(double), cudaMemcpyHostToDevice);
               // cudaMemcpy(d_centerMovement, centerMovement, k * sizeof(double), cudaMemcpyHostToDevice);
                //cudaMemcpy(d_converged, converged, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_closest2, closest2, n * sizeof(unsigned short), cudaMemcpyHostToDevice);

        cudaMemcpy(d_closest2, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
        //elkanFun<<<numBlocks, blockSize>>> (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_clusterSize, sumNewCenters[threadId]->d_data, d_centerMovement, k, d, endNdx, numLowerBounds, d_converged, d_closest2);
        elkanFunNoMove<<<numBlocks, blockSize>>> (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, d_test);
        //wtfTest<<<numBlocks, blockSize>>> (d_test);
        //setTest << <numBlocks, blockSize >> > (d_test);
        cudaDeviceSynchronize();
        cudaMemcpy(x->data, x->d_data, (n * d) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cudaMemcpy(lower, d_lower, (n * k) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(upper, d_upper, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(test, d_test, n * sizeof(int), cudaMemcpyDeviceToHost);

                //cudaMemcpy(clusterSize, d_clusterSize, k * sizeof(int), cudaMemcpyDeviceToHost);
                //cudaMemcpy(sumNewCenters[threadId]->data, sumNewCenters[threadId]->d_data, k * d * sizeof(double), cudaMemcpyDeviceToHost);
                //cudaMemcpy(centerMovement, d_centerMovement, k * sizeof(double), cudaMemcpyDeviceToHost);
                //cudaMemcpy(converged, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(closest2, d_closest2, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        int count = 0;
        for (int i = 0; i < n; i++) {
            count += test[i];
            std::cout << "i: " << i << " -> " << test[i] << std::endl;
        }
        //std::cout << "COUNT: " << count << std::endl;

        //elkanFunNoMove<<<numBlocks, blockSize>>> (x->data, centers->data, assignment, lower, upper, s, centerCenterDistDiv2, k, d, endNdx, numLowerBounds, converged, closest2);
        //cudaDeviceSynchronize();

        //changeAss<<<numBlocks, blockSize >>>(x->data, assignment, closest2, clusterSize, sumNewCenters[threadId]->data, d, endNdx);
        //cudaDeviceSynchronize();

        for (int i = startNdx; i < endNdx; ++i) {
            if (assignment[i] != closest2[i]) {
                changeAssignment(i, closest2[i], threadId);
            }
        }
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

    if (threadId == 0) {
        int furthestMovingCenter = move_centers();
        converged = (0.0 == centerMovement[furthestMovingCenter]);
    }
        synchronizeAllThreads();
        //total_elkan_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time));
        if (!converged) {
            update_bounds(startNdx, endNdx);
        }
        else {
            std::cout << "Iterations: " << iterations << "\n";
        }
    }
    //for (int i = 0; i < 10500; i++)
    //    std::cout << "i= " << i << ": " <<  test[i] << std::endl;

    //cudaFree(d_converged);
    cudaFree(d_closest2);
    cudaFree(d_test);
    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
}

void ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    int n = endNdx;
    int blockSize = 2 * 32;
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
    std::cout << "ElkanKmeans init" << std::endl;
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    std::cout << "ElkanKmeans init end" << std::endl;
    centerCenterDistDiv2 = new double[k * k];
    auto h = cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double));
    if (h != cudaSuccess) {
        std::cout << "cudaMalloc failed (centercenterdistdiv2)" << std::endl;
    }
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
}

void ElkanKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    //delete[] centerCenterDistDiv2;
    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    delete centerCenterDistDiv2;
    delete lower;
    
    centerCenterDistDiv2 = NULL;
    //delete centers;
    //centers = NULL;
}


/*CPU:

for (int i = 0; i < n; i++) {
    test[i] = i % 100;
    //closest2[i] = assignment[i];
}

cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
cudaMemcpy(d_closest2, closest2, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
cudaMemcpy(d_test, test, n * sizeof(int), cudaMemcpyHostToDevice);

elkanFunNoMove << <numBlocks, blockSize >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, 
    k, d, endNdx, numLowerBounds, d_closest2, d_test);
cudaDeviceSynchronize();

cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
cudaMemcpy(closest2, d_closest2, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
cudaMemcpy(test, d_test, n * sizeof(int), cudaMemcpyDeviceToHost);

for (int i = 0; i < n; i++) {
    std::cout << "i: " << i << " -> " << test[i] << std::endl;
}

GPU:
__global__ void elkanFunNoMove(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, int numlower, unsigned short* closest2, int* test) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        test[i] = 5;        
        //closest2[i] = closest2[i] - 1;           1
        //closest2[i] = 1;                         2
        //closest2[i] = assignment[i];             3
        ... 
*/