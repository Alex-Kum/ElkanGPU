#pragma once
#define DTYPE double
#define dtype double
#define BLOCKSIZE 256
#define _HUGE_ENUF  1e+300
#define INFINITY   ((float)(_HUGE_ENUF * _HUGE_ENUF))

//const int BLOCKSIZE = 256;

typedef struct PointInfo
{
    //Indices of old and new assigned centroids
    int centroidIndex;
    int oldCentroid;

    //The current upper bound
    DTYPE uprBound;
    //DTYPE oldUprBound;
}point;

typedef struct CentInfo {
    //Centroid's group index
    int groupNum;

    //Centroid's drift after updating
    DTYPE drift;

    //number of data points assigned to centroid
    int count;
} cent;


//__global__ void elkanFunFBHam(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
//    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2) {
//
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n) {
//        closest2[i] = assignment[i];
//        if (upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i]) {
//            double closestDistance = INFINITY;
//            double secondClosestDist = INFINITY;
//
//            for (int j = 0; j < k; ++j) {
//                double curDistance = sqrt(innerProdp2c(data, center, i, j, dim));
//                if (curDistance < closestDistance) {
//                    secondClosestDist = closestDistance;
//                    closestDistance = curDistance;
//                    closest2[i] = j;
//                }
//                else if (curDistance < secondClosestDist) {
//                    secondClosestDist = curDistance;
//                }
//            }
//            upper[i] = closestDistance;
//            lower[i] = secondClosestDist;
//        }
//    }
//}

//int c1 = i / k;
//int j = i % k;