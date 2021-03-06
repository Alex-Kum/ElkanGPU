#include "ham_elkan.h"
//#include "gpufunctions.h"
#include "general_functions.h"
#include <cmath>
#include <chrono>
#include <fstream>

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

void HamElkan::update_center_dists(int threadId) {

#if GPUA
    const int n = centers->n * centers->n;
    const int blockSize = 1 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    cudaMemset(d_s, std::numeric_limits<double>::max(), k * sizeof(double));
    innerProd << <numBlocks, blockSize >> > (d_centerCenterDistDiv2, d_s, centers->d_data, centers->d, centers->n);

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

int HamElkan::runThread(int threadId, int maxIterations) {
    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);

    unsigned short* closest2 = new unsigned short[endNdx];
    unsigned short* d_closest2;
    auto f = cudaMalloc(&d_closest2, endNdx * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (closest2)" << std::endl;
    }

    
  /*  auto g = cudaMalloc(&d_lower, (n) * sizeof(double));
    if (g != cudaSuccess) {
        std::cout << "cudaMalloc failed (lower)" << std::endl;
    }*/

   // std::fill(lower, lower + n, 0);

 /*   bool* d_check;
    g = cudaMalloc(&d_check, (k * n) * sizeof(bool));
    if (g != cudaSuccess) {
        std::cout << "cudaMalloc failed (check)" << std::endl;
    }*/

    bool* convergedd = new bool;
    bool* d_converged;
    f = cudaMalloc(&d_converged, 1 * sizeof(bool));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (converged)" << std::endl;
    }

   /* double* d_distances = new double[n * k];
    bool* d_calculated = new bool[n];
    cudaMalloc(&d_calculated, n * sizeof(bool));
    cudaMalloc(&d_distances, (n*k) * sizeof(double));*/
    //cudaMalloc(&d_distances2, (n * k) * sizeof(double));

    converged = false;
    *convergedd = false;

#if GPUC
    gpuErrchk(cudaMemcpy(x->d_data, x->data, (n * d) * sizeof(double), cudaMemcpyHostToDevice));
   /*LL gpuErrchk(cudaMemcpy(d_lower, lower, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_upper, upper, n * sizeof(double), cudaMemcpyHostToDevice));*/
    gpuErrchk(cudaMemcpy(d_assignment, assignment, n * sizeof(unsigned short), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centers->d_data, centers->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(sumNewCenters[0]->d_data, sumNewCenters[0]->data, (k * d) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusterSize, clusterSize[0], k * sizeof(int), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_lastExactCentroid, lastExactCentroid, (n * d) * sizeof(int), cudaMemcpyHostToDevice));

    /*std::cout << "Uppper: " << upper[0] << std::endl;
    std::cout << "Uppper: " << lower[0] << std::endl;*/
    const int nC = endNdx;
    const int blockSizeC = 3 * 32;
    const int numBlocksC = (nC + blockSizeC - 1) / blockSizeC;

    const int nD = endNdx * k;
    const int blockSizeD = 3 * 32;
    const int numBlocksD = (nD + blockSizeD - 1) / blockSizeD;

    const int nM = centers->n;
    const int blockSizeM = 1 * 32;
    const int numBlocksM = (nM + blockSizeM - 1) / blockSizeM;
    unsigned long long int* d_countDistances;
    cudaMalloc(&d_countDistances, 1 * sizeof(unsigned long long int));
    cudaMemset(d_countDistances, 0, 1 * sizeof(unsigned long long int));
#endif
#if GPUC  
    while ((iterations < maxIterations) && !(*convergedd)) {
        
        *convergedd = true,
#else
    while ((iterations < maxIterations) && !converged) {
#endif
        ++iterations;

        update_center_dists(threadId);

#if GPUC                          //!!!!!!!!!!!!!!!
        /*elkanFunHam << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, d_closest2, d_countDistances);*/
        elkanFunLloyd << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment, k, d, endNdx, d_closest2, d_countDistances);

       /* elkanFunFBHamK << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, n, d_closest2, d_distances, d_calculated);
        elkanFunFBHamTT << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, d_closest2, d_calculated, d_distances);*/

       /* calculateFilterH << <numBlocksC, blockSizeC >> > (d_assignment, d_lower, d_upper, d_s, d_calculated, nC, d_closest2);
        elkanFunFBHam2TTLoop << <numBlocksD, blockSizeD >> > (x->d_data, centers->d_data, d_distances, d_calculated, k, d, nC);
        elkanFunFBHamTT << <numBlocksC, blockSizeC >> > (d_lower, d_upper, k, nC, d_closest2, d_calculated, d_distances);*/

        /*elkanFunHamShared << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, d_closest2);*/
        /*elkanFunHamSharedK << <numBlocksC, blockSizeC*k >> > (x->d_data, centers->d_data, d_assignment,
            d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, d_closest2);*/
       // elkanFunHamFewerRules << <numBlocksC, blockSizeC >> > (x->d_data, centers->d_data, d_assignment,
       //     d_lower, d_upper, d_s, d_centerCenterDistDiv2, k, d, endNdx, d_closest2);

        changeAss << <numBlocksC, blockSizeC >> > (x->d_data, d_assignment, d_closest2, d_clusterSize, sumNewCenters[threadId]->d_data, d, nC, 0);

#else
        for (int i = startNdx; i < endNdx; ++i) {
            closest2[i] = assignment[i];
            
            if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
                double closestDistance = INFINITY;
                double secondClosestDist = INFINITY;
                for (int j = 0; j < k; ++j) {
                    //double curDistance = sqrt(dist22(data, center, i, j, dim));
                    double curDistance = sqrt(pointCenterDist2(i, j));
#if DISTANCES
                    unsigned long long int c;
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    if (curDistance < closestDistance) {
                        secondClosestDist = closestDistance;
                        closestDistance = curDistance;
                        closest2[i] = j;
                    }
                    else if (curDistance < secondClosestDist) {
                        secondClosestDist = curDistance;
                    }
                }
                upper[i] = closestDistance;
                lower[i] = secondClosestDist;
            }
            if (assignment[i] != closest2[i]) {
                changeAssignment(i, closest2[i], threadId);
            }
        }
#endif

#if GPUC
        cudaMemcpy(d_converged, convergedd, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        elkanMoveCenter << <numBlocksM, blockSizeM >> > (d_centerMovement, d_clusterSize, centers->d_data, sumNewCenters[threadId]->d_data, d_converged, k, d, nM);
        cudaMemcpy(convergedd, d_converged, 1 * sizeof(bool), cudaMemcpyDeviceToHost);

#else
        //verifyAssignment(iterations, startNdx, endNdx);
        int furthestMovingCenter = move_centers();
        converged = (0.0 == centerMovement[furthestMovingCenter]);
#endif
        // ELKAN 4, 5, AND 6
        // 
        //total_elkan_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time));

        //if (!converged){
        //std::cout << "iteration: " << iterations << std::endl;
#if GPUC
        if (!(*convergedd)) {
#else
        if (!converged) {
#endif
            update_bounds(startNdx, endNdx);
        }
    }

    /*cudaMemcpy(assignment, d_assignment, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++)
        std::cout << "assignment: " << assignment[i] << std::endl;*/
#if DISTANCES
    unsigned long long int cDist;
    //cudaMemcpy(&cDist, d_countDistances, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    std::cout << "distance calculations: " << cDist << std::endl;
#endif
    //std::ofstream myfile;
   // myfile.open("example2.txt");
   // 
   // for (int i = 0; i < n; i++)
   //     myfile << "i: " << i << " assignment: " << assignment[i] << "\n";

   ///* for (int i = 0; i < 20; i++)
   //     std::cout << "assignment: " << assignment[i] << std::endl;*/

   // myfile.close();

    cudaFree(d_closest2);
    cudaFree(d_converged);
    //cudaFree(d_check);
    /*cudaFree(d_distances);
    cudaFree(d_calculated);*/
    //cudaFree(d_lastExactCentroid);
    delete convergedd;
    /* for (int i = 0; i < nStreams; i++)
         cudaStreamDestroy(stream[i]);*/

    return iterations;
}

void HamElkan::update_bounds(int startNdx, int endNdx) {
#if GPUB
    const int n = endNdx;
    const int blockSize = 1 * 32;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    updateBoundHamShared << <numBlocks, 258 >> > (x->d_data, d_lower, d_upper, d_centerMovement, d_assignment, numLowerBounds, d, k, endNdx);
    updateBoundHam << <numBlocks, blockSize >> > (x->d_data, d_lower, d_upper, d_centerMovement, d_assignment, numLowerBounds, d, k, endNdx);
#else
    for (int i = startNdx; i < endNdx; ++i) {
        double maxMovement = 0;
        upper[i] += centerMovement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (j == assignment[i])
                continue;
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }
        lower[i] -= maxMovement;
    }
#endif
}

void HamElkan::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    std::cout << "ElkanKmeans init" << std::endl;
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    std::cout << "ElkanKmeans init end" << std::endl;
    centerCenterDistDiv2 = new double[k * k];
    auto h = cudaMalloc(&d_centerCenterDistDiv2, (k * k) * sizeof(double));
    if (h != cudaSuccess) {
        std::cout << "cudaMalloc failed (centercenterdistdiv2)" << std::endl;
    }
    //std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
    cudaMemset(d_centerCenterDistDiv2, 0.0, (k * k) * sizeof(double));
   // cudaMemset(d_lower, 0.0, (n) * sizeof(double));
    cudaMemset(d_upper, std::numeric_limits<double>::max(), (n) * sizeof(double));
    lower = new double[n];
}

void HamElkan::free() {
    TriangleInequalityBaseKmeans::free();
    cudaFree(d_centerCenterDistDiv2);
    cudaFree(d_lower);
    
    delete centerCenterDistDiv2;
    delete lower;

    centerCenterDistDiv2 = NULL;
}





//
//
//__global__ void elkanFunFB(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
//    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2) {
//
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n) {
//        //unsigned short closest = assignment[i];
//        closest2[i] = assignment[i];
//        bool r = true;
//
//        if (upper[i] > s[closest2[i]]) {
//            for (int j = 0; j < k; ++j) {
//                if (j == closest2[i]) { continue; }
//                if (upper[i] <= lower[i * k + j]) { continue; }
//                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
//
//                // ELKAN 3(a)
//                if (r) {
//                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
//                    lower[i * k + closest2[i]] = upper[i];
//                    r = false;
//
//                }
//
//                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
//                if (lower[i * k + j] < upper[i]) {
//                    closest2[i] = j;
//                    upper[i] = lower[i * k + j];
//                }
//            }
//        }
//        
//
//    }
//}