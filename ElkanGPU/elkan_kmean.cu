/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

 // -lineinfo  cuda c++ comand line

#include "elkan_kmean.h"
//#include "gpufunctions.h"
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
#define GPUC 0
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
    for (int c1 = 0; c1 < k; ++c1) {
        s[c1] = std::numeric_limits<double>::max();
    }

    int n = centers->n * centers->n;
    int blockSize = 2 * 32;
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
    innerProd << <numBlocks, blockSize >> > (centerCenterDistDiv2, s, centers->data, centers->d, centers->n, d_test);
    cudaDeviceSynchronize();
    cudaMemcpy(centers->data, x->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(test, d_test, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
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
    const int streamSize = 99840;
    const int nStreams = 5;
    cudaStream_t stream[nStreams];
    cudaStream_t stream1;
    cudaStream_t stream2;
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&stream[i]);
    // gpuErrchk(cudaStreamCreate(&stream1));
     //gpuErrchk(cudaStreamCreate(&stream2));
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
#if GPUC
    //gpuErrchk(cudaHostRegister(x->data, (n * d) * sizeof(double), cudaHostRegisterDefault));
    //gpuErrchk(cudaHostRegister(lower, (n * k) * sizeof(double), cudaHostRegisterDefault));
    //gpuErrchk(cudaHostRegister(upper, n * sizeof(double), cudaHostRegisterDefault));
    //gpuErrchk(cudaHostRegister(assignment, n * sizeof(unsigned short), cudaHostRegisterDefault));
    //for (int i = 0; i < nStreams; i++) {
    //           int offset = i * streamSize;
    //           //std::cout << "Offset: " << offset << std::endl;
    //           //std::cout << "Number upperbounds per stream : " << (n / nStreams) << std::endl;
    //           //std::cout << (n * d) * sizeof(double) / nStreams << std::endl;
    //           //cudaMemcpyAsync(&x->d_data[offset], &x->data[offset], (n * d) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
    //           gpuErrchk(cudaMemcpyAsync(&d_lower[offset], &lower[offset], (n * k) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]));
    //           gpuErrchk(cudaMemcpyAsync(&d_upper[offset], &upper[offset], n * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]));
    //           gpuErrchk(cudaMemcpyAsync(&d_assignment[offset], &assignment[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyHostToDevice, stream[i]));
    //       }

    gpuErrchk(cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lower, lower, (n * k) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice));

    //gpuErrchk(cudaMemcpyAsync(&x->d_data[0], &x->data[0], (250000 * d) * sizeof(double), cudaMemcpyHostToDevice,stream1));
   /* res = cudaMemcpyAsync(d_lower, lower, (0.5 * n * k) * sizeof(double), cudaMemcpyHostToDevice, stream1);
    if (res != cudaSuccess)
        std::cout << "help" << std::endl;
    res = cudaMemcpyAsync(d_upper, upper, 0.5 * n * sizeof(double), cudaMemcpyHostToDevice, stream1);
    if (res != cudaSuccess)
        std::cout << "help" << std::endl;*/
        //cudaMemcpyAsync(d_assignment, assignment, 0.5 * n * sizeof(unsigned short), cudaMemcpyHostToDevice, stream1);


       // gpuErrchk(cudaMemcpyAsync(x->d_data + 250000, x->data + 250000, (250000 * d) * sizeof(double), cudaMemcpyHostToDevice, stream1));
       // gpuErrchk(cudaDeviceSynchronize());
        //gpuErrchk(cudaHostUnregister(x->data));
        /*if (res != cudaSuccess)
            std::cout << "help" << std::endl;
        res = cudaMemcpyAsync(d_lower + 250000, lower + 250000, (0.5 * n * k) * sizeof(double), cudaMemcpyHostToDevice, stream2);
        if (res != cudaSuccess)
            std::cout << "help" << std::endl;
        res = cudaMemcpyAsync(d_upper + 250000, upper + 250000, 0.5 * n * sizeof(double), cudaMemcpyHostToDevice, stream2);
        if (res != cudaSuccess)
            std::cout << "help" << std::endl;*/
            //cudaMemcpyAsync(d_assignment + 250000, assignment + 250000, 0.5 * n * sizeof(unsigned short), cudaMemcpyHostToDevice, stream2);


           // std::cout << "start trans" << std::endl;
            //for (int i = 0; i < nStreams; i++) {
            //    int offset = i * streamSize;
            //    //std::cout << "Offset: " << offset << std::endl;
            //    //std::cout << "Number upperbounds per stream : " << (n / nStreams) << std::endl;
            //    //std::cout << (n * d) * sizeof(double) / nStreams << std::endl;
            //    //cudaMemcpyAsync(&x->d_data[offset], &x->data[offset], (n * d) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
            //    gpuErrchk(cudaMemcpyAsync(&d_lower[offset], &lower[offset], (n * k) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]));
            //    gpuErrchk(cudaMemcpyAsync(&d_upper[offset], &upper[offset], n * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]));
            //    gpuErrchk(cudaMemcpyAsync(&d_assignment[offset], &assignment[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyHostToDevice, stream[i]));
            //}
            //std::cout << "end trans" << std::endl;
            //cudaDeviceSynchronize();
            //cudaHostUnregister(x->data);
            /*

            /* gpuErrchk(cudaHostRegister(centers->data, (k * d) * sizeof(double), cudaHostRegisterDefault));
             gpuErrchk(cudaHostRegister(s, k * sizeof(double), cudaHostRegisterDefault));
             gpuErrchk(cudaHostRegister(centerCenterDistDiv2, (k * k) * sizeof(double), cudaHostRegisterDefault));
             gpuErrchk(cudaHostRegister(closest2, n * sizeof(unsigned short), cudaHostRegisterDefault));*/
#endif

    while ((iterations < maxIterations) && !(converged)) {
        //std::cout << "start iter" << iterations << std::endl;
        ++iterations;

        update_center_dists(threadId);
        synchronizeAllThreads();

#if GPUC
        //cudaDeviceSynchronize();
        //cudaHostUnregister(x->data);
        //cudaHostUnregister(lower);
        //cudaHostUnregister(upper);
        //cudaHostUnregister(assignment);
        int n = endNdx;
        int blockSize = 2 * 32;
        int numBlocks = (n + blockSize - 1) / blockSize;

        //cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_s, s, k * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_centerCenterDistDiv2, centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_closest2, closest2, n * sizeof(unsigned short), cudaMemcpyHostToDevice);

        ////cudaMemcpy(d_closest2, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
        ////elkanFun<<<numBlocks, blockSize>>> (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_clusterSize, sumNewCenters[threadId]->d_data, d_centerMovement, k, d, endNdx, numLowerBounds, d_converged, d_closest2);
        //elkanFunNoMove << <numBlocks, blockSize >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, 0);
        //cudaDeviceSynchronize();
        //cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        //cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(closest2, d_closest2, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);

        cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
       /* cudaMemcpy(d_s, s, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centerCenterDistDiv2, centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_closest2, closest2, n * sizeof(unsigned short), cudaMemcpyHostToDevice);*/

        //for (int i = 0; i < nStreams; i++) {
        //    int offset = i * streamSize;
        //    //cudaMemcpyAsync(&centers->d_data[offset], &centers->data[offset], (k * d) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        //    cudaMemcpyAsync(&d_s[offset], &d_s[offset], k * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        //    cudaMemcpyAsync(&d_centerCenterDistDiv2[offset], &centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        //    cudaMemcpyAsync(&d_closest2[offset], &closest2[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyHostToDevice, stream[i]);

        //    elkanFunNoMove << <streamSize / blockSize, blockSize, 0, stream[i] >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, offset);

        //    cudaMemcpyAsync(&s[offset], &d_s[offset], k * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //    cudaMemcpyAsync(&centerCenterDistDiv2[offset], &d_centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //    cudaMemcpyAsync(&closest2[offset], &d_closest2[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //    //cudaMemcpyAsync(&centerCenterDistDiv2[offset], &d_centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //    cudaMemcpyAsync(&assignment[offset], &d_assignment[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //    //cudaMemcpyAsync(&centers->data[offset], &centers->d_data[offset], (k * d) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        //}

        for (int i = 0; i < nStreams; i++) {
            int offset = i * streamSize;
            //cudaMemcpyAsync(&centers->d_data[offset], &centers->data[offset], (k * d) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&d_s[offset], &d_s[offset], k * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&d_centerCenterDistDiv2[offset], &centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&d_closest2[offset], &closest2[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        }

        for (int i = 0; i < nStreams; i++) {
            int offset = i * streamSize;
            elkanFunNoMove << <streamSize / blockSize, blockSize, 0, stream[i] >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, offset);
        }

        for (int i = 0; i < nStreams; i++) {
            int offset = i * streamSize;
            cudaMemcpyAsync(&s[offset], &d_s[offset], k * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
            cudaMemcpyAsync(&centerCenterDistDiv2[offset], &d_centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
            cudaMemcpyAsync(&closest2[offset], &d_closest2[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
            //cudaMemcpyAsync(&centerCenterDistDiv2[offset], &d_centerCenterDistDiv2[offset], (k * k) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
            cudaMemcpyAsync(&assignment[offset], &d_assignment[offset], n * sizeof(unsigned short) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
            //cudaMemcpyAsync(&centers->data[offset], &centers->d_data[offset], (k * d) * sizeof(double) / nStreams, cudaMemcpyDeviceToHost, stream[i]);
        }



        //elkanFunNoMove << <numBlocks, blockSize >> > (x->d_data, centers->d_data, d_assignment, d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, numLowerBounds, d_closest2, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        /*cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);

        cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(closest2, d_closest2, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);*/

        /* int count = 0;
         for (int i = 0; i < n; i++) {
             count += test[i];
             std::cout << "i: " << i << " -> " << test[i] << std::endl;
         }*/
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
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
            //std::cout << "Furthest Movement: " << centerMovement[furthestMovingCenter] << " (center " << furthestMovingCenter << ")" <<  std::endl;
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

    //cudaFree(d_converged);
    cudaFree(d_closest2);
    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(stream[i]);
    //cudaStreamDestroy(stream1);
    //cudaStreamDestroy(stream2);

    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
}

void ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    const int n = endNdx;
    const int blockSize = 2 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centerMovement, centerMovement, k * sizeof(double), cudaMemcpyHostToDevice);

    updateBound << <numBlocks, blockSize >> > (d_lower, d_upper, d_centerMovement, d_assignment, numLowerBounds, k, endNdx);
    cudaDeviceSynchronize();

    cudaMemcpy(centerMovement, d_centerMovement, k * sizeof(double), cudaMemcpyDeviceToHost);
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
