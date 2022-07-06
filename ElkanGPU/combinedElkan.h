#ifndef CO_ELKAN_KMEANS_H
#define CO_ELKAN_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations.
 */

#include <functional>
#include "triangle_inequality_base_kmeans.h"

class CO_ElkanKmeans : public TriangleInequalityBaseKmeans {
public:
    CO_ElkanKmeans() : centerCenterDistDiv2(NULL) {}
    //ElkanKmeans_newbound() : oldcenterCenterDistDiv2(NULL) {}
    virtual ~CO_ElkanKmeans() { free(); }
    virtual void free();
    virtual void initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "CO_elkan"; }

protected:
    virtual int runThread(int threadId, int maxIterations);

    // Update the distances between each pair of centers.
    void update_center_dists(int threadId);

    // Update the upper and lower bounds for the range of points given.
    void update_bounds(int startNdx, int endNdx);
    //void updateBoundsMO();
    //void updae

    // Keep track of the distance (divided by 2) between each pair of
    // points.
    const int changeIter = 35;
    int iterations;
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
    double* lower;
    double* d_lower;
    // std::function<void()> centerDists;
    // std::function<void()> updateBound;
    // std::function<void()> loopFun;
    int move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis);
};

#endif