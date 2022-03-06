/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cmath>

Kmeans::Kmeans() : x(NULL), n(0), k(0), d(0), numThreads(0), converged(false), counter(0),
clusterSize(NULL), centerMovement(NULL), assignment(NULL) {
    //std::cout << "kmeans konstructor" << std::endl;
#ifdef COUNT_DISTANCES
    numDistances = 0;
#endif
}

void Kmeans::free() {
    if (counter == 0) {
        //std::cout << "kmeans free" << std::endl;
        delete centerMovement;
        cudaFree(d_centerMovement);
        for (int t = 0; t < numThreads; ++t) {
            //if (d_clusterSize[t] != nullptr)
            //    cudaFree(d_clusterSize[t]);
            if (clusterSize[t] != nullptr)
                delete clusterSize[t];
        }
        cudaFree(d_clusterSize);
        delete[] clusterSize;

        cublasDestroy(cublas_handle);
        centerMovement = NULL;
        clusterSize = NULL;
        assignment = NULL;
        n = k = d = numThreads = 0;
        counter++;
    }
}



void Kmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    //free();
    std::cout << "kmeans init" << std::endl;
    converged = false;
    x = aX;
    n = x->n;
    d = x->d;
    k = aK;
    stat = cublasCreate(&cublas_handle);
    //std::cout << n << "\n";
#ifdef USE_THREADS
    numThreads = aNumThreads;
    pthread_barrier_init(&barrier, NULL, numThreads);
#else
    numThreads = 1;
#endif

    assignment = initialAssignment;
    auto r = cudaMalloc(&d_assignment, n * sizeof(unsigned short));
    if (r != cudaSuccess) {
        std::cout << "cudaMalloc failed (centerMovement)" << std::endl;
    }
    centerMovement = new double[k];
    auto a = cudaMalloc(&d_centerMovement, k * sizeof(double));
    if (a != cudaSuccess) {
        std::cout << "cudaMalloc failed (centerMovement)" << std::endl;
    }
    clusterSize = new int* [numThreads];
    auto c = cudaMalloc(&d_clusterSize, k * sizeof(int));
    if (c != cudaSuccess) {
        std::cout << "cudaMalloc failed (clusterSize[])" << std::endl;
    }
    for (int t = 0; t < numThreads; ++t) {
        clusterSize[t] = new int[k];

        std::fill(clusterSize[t], clusterSize[t] + k, 0);
        for (int i = start(t); i < end(t); ++i) {
            assert(assignment[i] < k);
            ++clusterSize[t][assignment[i]];
        }
    }


#ifdef COUNT_DISTANCES
    numDistances = 0;
#endif
}

void Kmeans::changeAssignment(int xIndex, int closestCluster, int threadId) {
    --clusterSize[threadId][assignment[xIndex]];
    ++clusterSize[threadId][closestCluster];

    assignment[xIndex] = closestCluster;

}
#ifdef USE_THREADS
struct ThreadInfo {
public:
    ThreadInfo() : km(NULL), threadId(0), pthread_id(0) {}
    Kmeans* km;
    int threadId;
    pthread_t pthread_id;
    int numIterations;
    int maxIterations;
};
#endif

void* Kmeans::runner(void* args) {
#ifdef USE_THREADS
    ThreadInfo* ti = (ThreadInfo*)args;
    ti->numIterations = ti->km->runThread(ti->threadId, ti->maxIterations);
#endif
    return NULL;
}

int Kmeans::run(int maxIterations) {
    int iterations = 0;
#ifdef USE_THREADS
    {
        ThreadInfo* info = new ThreadInfo[numThreads];
        for (int t = 0; t < numThreads; ++t) {
            info[t].km = this;
            info[t].threadId = t;
            info[t].maxIterations = maxIterations;
            pthread_create(&info[t].pthread_id, NULL, Kmeans::runner, &info[t]);
        }
        // wait for everything to finish...
        for (int t = 0; t < numThreads; ++t) {
            pthread_join(info[t].pthread_id, NULL);
        }
        iterations = info[0].numIterations;
        delete[] info;
    }
#else
    {
        //x->print();
        iterations = runThread(0, maxIterations);
    }
#endif

    return iterations;
}

double Kmeans::getSSE() const {
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        sse += pointCenterDist2(i, assignment[i]);
    }
    return sse;
}

void Kmeans::verifyAssignment(int iteration, int startNdx, int endNdx) const {
#ifdef VERIFY_ASSIGNMENTS
    for (int i = startNdx; i < endNdx; ++i) {
        // keep track of the squared distance and identity of the closest-seen
        // cluster (so far)
        int closest = assignment[i];
        double closest_dist2 = pointCenterDist2(i, closest);
        double original_closest_dist2 = closest_dist2;
        // look at all centers
        for (int j = 0; j < k; ++j) {
            if (j == closest) {
                continue;
            }
            double d2 = pointCenterDist2(i, j);

            // determine if we found a closer center
            if (d2 < closest_dist2) {
                closest = j;
                closest_dist2 = d2;
            }
        }

        // if we have found a discrepancy, then print out information and crash
        // the program
        if (closest != assignment[i]) {
            std::cerr << "assignment error:" << std::endl;
            std::cerr << "iteration             = " << iteration << std::endl;
            std::cerr << "point index           = " << i << std::endl;
            std::cerr << "closest center        = " << closest << std::endl;
            std::cerr << "closest center dist2  = " << closest_dist2 << std::endl;
            std::cerr << "assigned center       = " << assignment[i] << std::endl;
            std::cerr << "assigned center dist2 = " << original_closest_dist2 << std::endl;
            assert(false);
        }
    }
#endif
}