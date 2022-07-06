/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "ham_elkanFB.h"
#include "general_functions.h"
//#include "gpufunctions.h"
#include <cmath>
#include <chrono>

#define GPUA 0
#define GPUB 0
#define GPUC 0

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void HamElkanFB::update_center_dists(int threadId) {
#if GPUA
    const int n = centers->n * centers->n;
    const int blockSize = 1 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    /* cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);*/
    //innerProdFBHam << <numBlocks, blockSize >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);
    cudaMemset(d_s, std::numeric_limits<double>::max(), k * sizeof(double));
    innerProd << <numBlocks, blockSize >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);
    /* cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
     cudaMemcpy(centerCenterDistDiv2, d_centerCenterDistDiv2, (k * k) * sizeof(double), cudaMemcpyDeviceToHost);
     cudaMemcpy(s, d_s, k * sizeof(double), cudaMemcpyDeviceToHost);*/

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

int HamElkanFB::runThread(int threadId, int maxIterations) {
    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);

    unsigned short* closest2 = new unsigned short[endNdx];
    unsigned short* d_closest2;
    auto f = cudaMalloc(&d_closest2, endNdx * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (closest2)" << std::endl;
    }

    bool* convergedd = new bool;
    bool* d_converged;
    f = cudaMalloc(&d_converged, 1 * sizeof(bool));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (converged)" << std::endl;
    }

    double* distances = new double[n * k];
    double* distances2 = new double[n * k];
    cudaMalloc(&d_calculated, n * sizeof(bool));
    cudaMalloc(&d_distances, (n * k) * sizeof(double));
    //cudaMalloc(&d_distances2, (n * k) * sizeof(double));

    converged = false;
    *convergedd = false;

#if GPUC
    cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lower, lower, (n * k) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub_old, ub_old, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldcenter2newcenterDis, oldcenter2newcenterDis, (k * k) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice));

    const int nC = endNdx * k;
    std::cout << "nc: " << nC << std::endl;
    const int blockSizeC = 3 * 32;
    const int numBlocksC = (nC + blockSizeC - 1) / blockSizeC;

    const int nD = endNdx;
    const int blockSizeD = 3 * 32;
    const int numBlocksD = (nD + blockSizeD - 1) / blockSizeD;

    const int nM = centers->n;
    const int blockSizeM = 1 * 32;
    const int numBlocksM = (nM + blockSizeM - 1) / blockSizeM;

    unsigned long long int* d_countDistances;
    gpuErrchk(cudaMalloc(&d_countDistances, 1 * sizeof(unsigned long long int)));
    cudaMemset(d_countDistances, 0, 1 * sizeof(unsigned long long int));

#endif
#if GPUC
    while ((iterations < maxIterations) && !(*convergedd)) {
#else
    while ((iterations < maxIterations) && !converged) {
#endif 
        ++iterations;
        *convergedd = true;

        update_center_dists(threadId);


#if GPUC  
        /*calculateFilter << <numBlocksD, blockSizeD >> > (d_assignment, d_lower, d_upper, d_s, d_maxoldcenter2newcenterDis, d_ub_old, d_calculated, n, d_closest2);
        elkanFunFBHam2TT << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_distances, d_calculated, k, d, endNdx);
        elkanFunFBHamTT << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_maxoldcenter2newcenterDis, d_ub_old, k, d, endNdx, d_closest2, d_calculated, d_distances);*/

        //calculateFilter2 << <numBlocksD, blockSizeD >> > (d_assignment, d_lower, d_upper, d_s, d_maxoldcenter2newcenterDis, d_ub_old, d_calculated, n, d_closest2, x->d_data, centers->d_data, d);
       // /*elkanFunFBHam2 << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_distances, d_calculated, k, d, endNdx*10);
        //elkanFunFBHamBounds << <numBlocksD, blockSizeD >> > (x->d_data, d_lower, d_upper, d_distances, d_calculated, k, d, n, d_closest2);*/

       //// elkanFunFBHam2 << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_distances2, d_calculated, k, d, endNdx);


       // //cudaMemcpy(distances, d_distances, (n * k) * sizeof(double), cudaMemcpyDeviceToHost);
       // //cudaMemcpy(distances2, d_distances2, (n * k) * sizeof(double), cudaMemcpyDeviceToHost);
       // //for (int i = 0; i < n; i++) {
       // //    for (int j = 0; j < k; j++) {
       // //        if (distances[i * k + j] != distances2[i * k + j]) {
       // //            std::cout << "UNGLEICH -- i: " << i << " j: " << j << " " << distances[i * k + j] << " != " << distances2[i * k + j] << std::endl;
       // //        }                
       // //    }
       // //}
       // //cudaMemcpy(d_distances, distances, (n * k) * sizeof(double), cudaMemcpyHostToDevice);
       // //cudaMemcpy(d_distances2, distances2, (n * k) * sizeof(double), cudaMemcpyHostToDevice);        

        elkanFunFBHam << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, d_maxoldcenter2newcenterDis, d_ub_old, k, d, endNdx, d_closest2, d_countDistances);

            //elkanFunLloyd << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_assignment, k, d, endNdx, d_closest2);

        changeAss << <numBlocksD, blockSizeD >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[threadId]->d_data, d, nC, 0);
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


        //verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
#if GPUC 
        cudaMemcpy(d_converged, convergedd, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        elkanMoveCenterFB << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[threadId]->d_data, d_oldcenters, d_converged, k, d, nM);
        cudaMemcpy(convergedd, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

        const int n = centers->n * centers->n;
        const int blockSize = 1 * 32;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double));
        elkanFBMoveAddition << <numBlocks, blockSize >> > (d_oldcenters, d_oldcenter2newcenterDis, centers->d_data, d, k, centers->n);
        elkanFBMoveAdditionHam << <centers->n, 1 >> > (d_oldcenters, d_oldcenter2newcenterDis, d_maxoldcenter2newcenterDis, k, centers->n);
#else
        int furthestMovingCenter = move_centers_newbound(oldcenters, oldcenter2newcenterDis);
        converged = (0.0 == centerMovement[furthestMovingCenter]);

#endif
#if GPUC
        if (!(*convergedd)) {
#else
        if (!converged) {
#endif
            update_bounds(startNdx, endNdx);
        }
        }
    /* cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
     for (int i = 0; i < 20; i++) {
         std::cout << "assignment: " << assignment[i] << std::endl;
     }*/

    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
    }

void HamElkanFB::update_bounds(int startNdx, int endNdx) {
#if GPUB
    int n = endNdx;
    int blockSize = 3 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    /*cudaMemcpy(d_lower, lower, (n * k) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub_old, ub_old, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centerMovement, centerMovement, k * sizeof(double), cudaMemcpyHostToDevice);*/

    updateBoundFBHam << <numBlocks, blockSize >> > (d_lower, d_upper, d_ub_old, d_centerMovement, d_assignment, numLowerBounds, k, endNdx);

    /*cudaMemcpy(lower, d_lower, (n * k) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(upper, d_upper, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ub_old, d_ub_old, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    cudaMemcpy(centerMovement, d_centerMovement, k * sizeof(double), cudaMemcpyDeviceToHost);*/

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

void HamElkanFB::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    centerCenterDistDiv2 = new double[k * k];
    cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double));
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
    oldcenter2newcenterDis = new double[k * k];
    cudaMalloc(&d_oldcenter2newcenterDis, (k * k) * sizeof(double));
    std::fill(oldcenter2newcenterDis, oldcenter2newcenterDis + k * k, 0.0);
    cudaMalloc(&d_maxoldcenter2newcenterDis, k * sizeof(double));
    ub_old = new double[n];
    cudaMalloc(&d_ub_old, n * sizeof(double));
    std::fill(ub_old, ub_old + n, std::numeric_limits<double>::max());
    lower = new double[n];
    cudaMalloc(&d_lower, n * sizeof(double));
    std::fill(lower, lower + n, 0.0);
    oldcenters = new double[k * d];
    cudaMalloc(&d_oldcenters, (k * d) * sizeof(double));
    std::fill(oldcenters, oldcenters + k * d, 0.0);
}

void HamElkanFB::free() {
    TriangleInequalityBaseKmeans::free();
    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    cudaFree(d_ub_old);
    cudaFree(d_oldcenters);
    cudaFree(d_oldcenter2newcenterDis);
    cudaFree(d_maxoldcenter2newcenterDis);
    delete centerCenterDistDiv2;
    delete lower;
    delete ub_old;
    delete oldcenters;
    //delete[] centerCenterDistDiv2;
    //centerCenterDistDiv2 = NULL;
    //delete [] oldcenterCenterDistDiv2;
    //oldcenterCenterDistDiv2 = NULL;
    delete centers;
    centers = NULL;
}

int HamElkanFB::move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis) {

    int furthestMovingCenter = 0;
    for (int j = 0; j < k; ++j) {
        centerMovement[j] = 0.0;
        int totalClusterSize = 0;
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
    //return 0;
    return furthestMovingCenter;
}
