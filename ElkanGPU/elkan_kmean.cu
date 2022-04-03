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
#define GPUA 1
#define GPUB 1
#define GPUC 1
#endif

#define GPUD 0

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void ElkanKmeans::update_center_dists(int threadId) {

#if GPUA
    // find the inter-center distances
    /*for (int c1 = 0; c1 < k; ++c1) {
        s[c1] = std::numeric_limits<double>::max();
    }*/

    int n = centers->n * centers->n;
    int blockSize = 2 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

   // cudaMemcpy(d_s, s, k * sizeof(double), cudaMemcpyHostToDevice);
    
    innerProd << <numBlocks, blockSize >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);
    cudaDeviceSynchronize();
    //cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);

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
    /*const int streamSize = 99840;
    const int nStreams = 5;*/
    const int streamSize = 249600;
    const int nStreams = 2;
    cudaStream_t stream[nStreams];

    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&stream[i]);

    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);
    std::cout << "endNdx : " << endNdx << std::endl;

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

    bool* convergedd = new bool;
    bool* d_converged;
    f = cudaMalloc(&d_converged, 1 * sizeof(bool));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (converged)" << std::endl;
    }

    converged = false;
    *convergedd = false;
#if GPUC

    gpuErrchk(cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lower, lower, (n * k) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice));
#endif
    
    while ((iterations < maxIterations) && !(*convergedd)) {
        *convergedd = true,
        ++iterations;

        update_center_dists(threadId);
        synchronizeAllThreads();

#if GPUC

        int n = endNdx;
        int blockSize = 2 * 32;        int numBlocks = (n + blockSize - 1) / blockSize;
        
        //gpuErrchk(cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
        //cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice);

        elkanFunNoMove << <numBlocks, blockSize >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, d_clusterSize, sumNewCenters[threadId]->d_data, 0);
        cudaDeviceSynchronize();

        //cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(sumNewCenters[0]->data, sumNewCenters[0]->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(clusterSize[0], d_clusterSize, k * sizeof(int), cudaMemcpyDeviceToHost);

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


        //verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6

        synchronizeAllThreads();


        if (threadId == 0) {
            //cudaMemcpy(centers->d_data, centers->data, (k* d) * sizeof(double), cudaMemcpyHostToDevice);
            //cudaMemcpy(d_centerMovement, centerMovement, k * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_converged, convergedd, 1 * sizeof(bool), cudaMemcpyHostToDevice);

            int nM = centers->n;
            int blockSizeM = 1 * 32;
            int numBlocksM = (nM + blockSizeM - 1) / blockSizeM;

            elkanMoveCenter << <numBlocks, blockSize >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[threadId]->d_data, d_converged, k, d, nM);
            cudaDeviceSynchronize();

            /*int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);*/
            //cudaMemcpy(centerMovement, d_centerMovement, k * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(convergedd, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            //cudaMemcpy(centers->data, centers->d_data, (k* d) * sizeof(double), cudaMemcpyDeviceToHost);
            //std::cout << "center after: " << centers->data[10] << std::endl;
            //std::cout << "Furthest Movement: " << centerMovement[furthestMovingCenter] << " (center " << furthestMovingCenter << ")" <<  std::endl;
        }


        synchronizeAllThreads();
        //total_elkan_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time));
        if (!(*convergedd)) {
            update_bounds(startNdx, endNdx);
            //std::cout << "Iterations: " << iterations << "\n";
        }
        else {
            std::cout << "Iterations: " << iterations << "\n";
        }
    }

    //cudaFree(d_converged);
    cudaFree(d_closest2);
    cudaFree(d_converged);
    delete convergedd;
    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(stream[i]);

    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
}

void ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    const int n = endNdx;
    const int blockSize = 2 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    //cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_centerMovement, centerMovement, k * sizeof(double), cudaMemcpyHostToDevice);

    updateBound << <numBlocks, blockSize >> > (d_lower, d_upper, d_centerMovement, d_assignment, numLowerBounds, k, endNdx);
    cudaDeviceSynchronize();

    //cudaMemcpy(centerMovement, d_centerMovement, k * sizeof(double), cudaMemcpyDeviceToHost);
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