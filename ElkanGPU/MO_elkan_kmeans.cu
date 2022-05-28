/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "MO_elkan_kmeans.h"
#include "general_functions.h"
//#include "gpufunctions.h"
#include <cmath>
#include <chrono>
 //using namespace std::chrono;

#define Time 0
#define Countdistance 0
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

void MO_ElkanKmeans::update_center_dists(int threadId) {
#if GPUA
    const int n = centers->n * centers->n;
    const int blockSize = 3 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMemset(d_s, std::numeric_limits<double>::max(), k * sizeof(double));
    innerProdMO << <numBlocks, blockSize >> > (d_centerCenterDistDiv2, d_oldcenterCenterDistDiv2, d_s, centers->d_data, centers->d, k, centers->n);
#else
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<double>::max();

            for (int c2 = 0; c2 < k; ++c2) {
                // we do not need to consider the case when c1 == c2 as centerCenterDistDiv2[c1*k+c1]
                // is equal to zero from initialization, also this distance should not be used for s[c1]
                if (c1 != c2) {
                    // divide by 2 here since we always use the inter-center
                    // distances divided by 2
                    //std::cout <<sqrt(centerCenterDist2(c1, c2))<< "\n";
                    oldcenterCenterDistDiv2[c1 * k + c2] = centerCenterDistDiv2[c1 * k + c2];
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

int MO_ElkanKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);
    //x->print();
    //auto start_time = std::chrono::high_resolution_clock::now();

    unsigned short* closest2 = new unsigned short[endNdx];
    unsigned short* d_closest2;
    auto f = cudaMalloc(&d_closest2, endNdx * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (closest2)" << std::endl;
    }

    ub_old = new double[n];
    cudaMalloc(&d_ub_old, n * sizeof(double));
    std::fill(ub_old, ub_old + n, std::numeric_limits<double>::max());

    oldcenterCenterDistDiv2 = new double[k * k];
    cudaMalloc(&d_oldcenterCenterDistDiv2, (k*k) * sizeof(double));
    std::fill(oldcenterCenterDistDiv2, oldcenterCenterDistDiv2 + k * k, 0.0);

    oldcenter2newcenterDis = new double[k * k];
    cudaMalloc(&d_oldcenter2newcenterDis, (k * k) * sizeof(double));
    std::fill(oldcenter2newcenterDis, oldcenter2newcenterDis + k * k, 0.0);

    oldcenters = new double[k * d];
    cudaMalloc(&d_oldcenters, (k * d) * sizeof(double));
    //oldcenters->fill(0.0);
    std::fill(oldcenters, oldcenters + k * d, 0.0);

    bool* convergedd = new bool;
    bool* d_converged;
    f = cudaMalloc(&d_converged, 1 * sizeof(bool));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (converged)" << std::endl;
    }

#if GPUC
    const int nC = endNdx;
    const int blockSizeC = 3 * 32;
    const int numBlocksC = (n + blockSizeC - 1) / blockSizeC;

    const int nM = centers->n;
    const int blockSizeM = 1 * 32;
    const int numBlocksM = (nM + blockSizeM - 1) / blockSizeM;

    cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub_old, ub_old, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldcenter2newcenterDis, oldcenter2newcenterDis, (k * k) * sizeof(double), cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice));
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
        int nC = endNdx;
        int blockSizeC = 3 * 32;
        int numBlocksC = (nC + blockSizeC - 1) / blockSizeC;   
        
        elkanFunMO << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment, 
            d_upper, d_s, d_centerCenterDistDiv2, d_oldcenter2newcenterDis, d_oldcenterCenterDistDiv2, d_ub_old, d_centerMovement, k, d, endNdx, d_closest2);

        changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[threadId]->d_data, d, nC, 0);

#else
        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;

            if (upper[i] <= s[closest]) {
                continue;
            }

            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                if (upper[i] <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }  //upper[i] <= lower[i * k + j] ||
                if (upper[i] <= centerCenterDistDiv2[closest * k + j]) { continue; }
#if Countdistance
                numberdistances++;
#endif
                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));
                    //lower[i * k + closest] = upper[i];
                    //lower2[i * k + closest] = upper[i];
                    r = false;
                    //if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j])) {
                        //continue;
                    //}
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(pointCenterDist2(i, j));

                if (sqrt(pointCenterDist2(i, j)) < upper[i]) {
                    closest = j;
                    upper[i] = sqrt(pointCenterDist2(i, j));
                }
            }
            if (assignment[i] != closest) {
                changeAssignment(i, closest, threadId);
            }


        }

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
#endif
#if GPUC 
        /*cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cudaMemcpy(centers->data, centers->d_data, (k * d) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(x->data, x->d_data, (n * d) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        verifyAssignment(iterations, startNdx, endNdx);
        cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();*/

        cudaMemcpy(d_converged, convergedd, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        elkanMoveCenterFB << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[threadId]->d_data, d_oldcenters, d_converged, k, d, nM);
        cudaMemcpy(convergedd, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

        const int n = centers->n * centers->n;
        const int blockSize = 1 * 32;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        cudaMemset(d_oldcenter2newcenterDis, 0.0, (k * k) * sizeof(double));
        elkanFBMoveAddition << <numBlocks, blockSize >> > (d_oldcenters, d_oldcenter2newcenterDis, centers->d_data, d, k, centers->n);
#else
        if (threadId == 0) {
            int furthestMovingCenter = move_centers_newbound(oldcenters, oldcenter2newcenterDis);
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }
#endif          
#if GPUC
        if (!(*convergedd)) {
#else
        if (!converged) {
#endif
            update_bounds(startNdx, endNdx);
            //std::cout << "iter: " << iterations << std::endl;
        }
        synchronizeAllThreads();

    }
    cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++) {
        std::cout << "assignment: " << assignment[i] << std::endl;
    }
    delete convergedd;
    cudaFree(d_converged);
    std::cout << "ITERATIONEN: " << iterations << std::endl;
    return iterations;
}

void MO_ElkanKmeans::update_bounds(int startNdx, int endNdx) {
#if GPUB
    int n = endNdx;
    int blockSize = 3 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    updateBoundMO << <numBlocks, blockSize >> > (d_upper, d_ub_old, d_centerMovement, d_assignment, endNdx);

#else
    for (int i = startNdx; i < endNdx; ++i) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
    }   
    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
    }
#endif
}

void MO_ElkanKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    centerCenterDistDiv2 = new double[k * k];
    cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double));
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);

}

void MO_ElkanKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_oldcenterCenterDistDiv2);
    cudaFree(d_oldcenter2newcenterDis);
    cudaFree(d_oldcenters);
    
    delete centerCenterDistDiv2;
    centerCenterDistDiv2 = NULL;
    cudaFree(d_ub_old);

    
    delete oldcenterCenterDistDiv2;
    delete oldcenter2newcenterDis;
    delete oldcenters;
    delete ub_old;
    //delete [] oldcenterCenterDistDiv2;
    //oldcenterCenterDistDiv2 = NULL;
    delete centers;
    centers = NULL;
}
int MO_ElkanKmeans::move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis) {
    int furthestMovingCenter = 0;
    /*
    for (int j = 0; j < k; ++j) {

        //std::cout << oldcenters[ j]<< "\n";
        for (int dim = 0; dim < d; ++dim) {
            std::cout << oldcenters[j] << "\n";
        }
    }
    */
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
                //std::cout << z << "\n";
                //std::cout << (*centers)(j, dim) << "\n";
                centerMovement[j] += (z - (*centers)(j, dim)) * (z - (*centers)(j, dim));//calculate distance
                //std::cout << (*centers)(j, dim) << "\n";
                old = (*centers)(j, dim);
                //std::cout << (*oldcenters)(j, dim) << "\n";
                oldcenters[j * d + dim] = old;
                //std::cout << (*centers)(j, dim) << "\n";
                (*centers)(j, dim) = z; //update new centers
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
/*
int ElkanKmeans_newbound_opt::move_centers2(int *sortindex,bool sorting, bool *nochanged) {
    int furthestMovingCenter = 0;
    for (int j = 0; j < k; ++j) {
        if (sorting == true){
        //j=sortindex[j];
            auto itr = std::find(sortindex,sortindex + k,j);


            centerMovement[j] = 0.0;
            int totalClusterSize = 0;
            for (int t = 0; t < numThreads; ++t) {
                totalClusterSize += clusterSize[t][j];
            }
            if (totalClusterSize > 0) {
                if (nochanged[j]==false){


                for (int dim = 0; dim < d; ++dim) {
                    double z = 0.0;
                    for (int t = 0; t < numThreads; ++t) {
                        z += (*sumNewCenters[t])(j,dim);
                    }
                    z /= totalClusterSize;
                    //std::cout << z << "\n";

                    centerMovement[j] += (z - (*centers)(std::distance(sortindex,itr), dim)) * (z - (*centers)(std::distance(sortindex,itr), dim));//calculate distance
                    (*centers)(std::distance(sortindex,itr), dim) = z; //update new centers
                }

            centerMovement[j] = sqrt(centerMovement[j]);
                }
            if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
                furthestMovingCenter = j;
            }
            }
        }
        else{
        centerMovement[j] = 0.0;
        int totalClusterSize = 0;
        for (int t = 0; t < numThreads; ++t) {
            totalClusterSize += clusterSize[t][j];
        }
        if (totalClusterSize > 0) {
            if (nochanged[j]==false){
            for (int dim = 0; dim < d; ++dim) {
                double z = 0.0;
                for (int t = 0; t < numThreads; ++t) {
                    z += (*sumNewCenters[t])(j,dim);
                }
                z /= totalClusterSize;
                //std::cout << z << "\n";
                centerMovement[j] += (z - (*centers)(j, dim)) * (z - (*centers)(j, dim));//calculate distance
                (*centers)(j, dim) = z; //update new centers
            }

        centerMovement[j] = sqrt(centerMovement[j]);
            }
        if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
            furthestMovingCenter = j;
        }
        }

        }
    }

    #ifdef COUNT_DISTANCES
    numDistances += k;
    #endif

    return furthestMovingCenter;
}
*/
