#ifndef HAM_ELKAN_H
#define HAM_ELKAN_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations.
 */


#include "triangle_inequality_base_kmeans.h"

class HamElkan : public TriangleInequalityBaseKmeans {
public:
    HamElkan() : centerCenterDistDiv2(NULL) {}
    virtual ~HamElkan() { free(); }
    virtual void free();
    virtual void initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, double* low, int aNumThreads);
    virtual std::string getName() const { return "HamElkan"; }

protected:
    virtual int runThread(int threadId, int maxIterations);

    // Update the distances between each pair of centers.
    void update_center_dists(int threadId);

    // Update the upper and lower bounds for the range of points given.
    void update_bounds(int startNdx, int endNdx);

    // Keep track of the distance (divided by 2) between each pair of
    // points.
    double* centerCenterDistDiv2;
    double* lower;
    double* d_centerCenterDistDiv2;
    double* d_lower;
};

#endif

