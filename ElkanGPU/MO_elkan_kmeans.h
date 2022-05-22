#ifndef MO_ELKAN_KMEANS_H
#define MO_ELKAN_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations.
 */


#include "triangle_inequality_base_kmeans.h"

class MO_ElkanKmeans : public TriangleInequalityBaseKmeans {
public:
    MO_ElkanKmeans() : centerCenterDistDiv2(NULL) {}
    //ElkanKmeans_newbound() : oldcenterCenterDistDiv2(NULL) {}
    virtual ~MO_ElkanKmeans() { free(); }
    virtual void free();
    virtual void initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "MO_elkan"; }

protected:
    virtual int runThread(int threadId, int maxIterations);

    // Update the distances between each pair of centers.
    void update_center_dists(int threadId);

    // Update the upper and lower bounds for the range of points given.
    void update_bounds(int startNdx, int endNdx);

    // Keep track of the distance (divided by 2) between each pair of
    // points.
    double* centerCenterDistDiv2;
    double* d_centerCenterDistDiv2;
    double* oldcenter2newcenterDis;
    double* d_oldcenter2newcenterDis;
    double* oldcenters;
    double* d_oldcenters;
    double* ub_old;
    double* d_ub_old;
    double* oldcenterCenterDistDiv2;
    double* d_oldcenterCenterDistDiv2;
    int move_centers2(int* sortindex, bool sorting, bool* nochanged);
    int move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis);

};

#endif



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