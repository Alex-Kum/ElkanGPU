/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 * 
 * https://github.com/kuanhsunchen/ElkanOpt
 * 
 * I added GPU Code
 */

#include "triangle_inequality_base_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <limits>
#include <cmath>

void TriangleInequalityBaseKmeans::free() {
    OriginalSpaceKmeans::free();
    cudaFree(d_s);
    cudaFree(d_upper);
    delete s;
    delete upper;
    //delete [] lower;
    //delete [] lower2;
    //delete [] lower3;
    s = NULL;
    upper = NULL;
    //lower = NULL;
    //lower2 = NULL;
    //lower3 = NULL;
}

/* This function computes the inter-center distances, keeping only the closest
 * distances, and updates "s". After this, s[j] will contain the distance
 * between center j and its closest other center, divided by two. The division
 * here saves repeated work later, since we always will need the distance / 2.
 *
 * Parameters: none
 *
 * Return value: none
 */
 // TODO: parallelize this
void TriangleInequalityBaseKmeans::update_s(int threadId) {
    // initialize
    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<double>::max();
        }
    }
    // compute inter-center squared distances between all pairs
    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            for (int c2 = 0; c2 < k; ++c2) {
                if (c2 == c1) {
                    continue;
                }
                double d2 = centerCenterDist2(c1, c2);
                if (d2 < s[c1]) { s[c1] = d2; }
            }
            // take the root and divide by two
            s[c1] = sqrt(s[c1]) / 2.0;
        }
    }
}


/* This function initializes the upper/lower bounds, assignment, centerCounts,
 * and sumNewCenters. It sets the bounds to invalid values which will force the
 * first iteration of k-means to set them correctly.  NB: subclasses should set
 * numLowerBounds appropriately before entering this function.
 *
 * Parameters: none
 *
 * Return value: none
 */
void TriangleInequalityBaseKmeans::initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads) {
    //std::cout << "triangleineququalitybasekmeans init" << std::endl;
    OriginalSpaceKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    s = new double[k];
    auto j = cudaMalloc(&d_s, k * sizeof(double));
    if (j != cudaSuccess) {
        std::cout << "cudaMalloc failed (s)" << std::endl;
    }
    upper = new double[n];
    auto k = cudaMalloc(&d_upper, n * sizeof(double));
    if (k != cudaSuccess) {
        std::cout << "cudaMalloc failed (upper)" << std::endl;
    }
    //cudaMemset(d_upper, std::numeric_limits<double>::max(), n * sizeof(double));

    //lower = new double[n * numLowerBounds];
    //lower2 = new double[n * numLowerBounds];
    //lower3 = new double[n * numLowerBounds];

    // start with invalid bounds and assignments which will force the first
    // iteration of k-means to do all its standard work 
    std::fill(s, s + k, 0.0);
    std::fill(upper, upper + n, std::numeric_limits<double>::max());
    //std::fill(lower, lower + n * numLowerBounds, 0.0);
    //std::fill(lower2, lower2 + n * numLowerBounds, 0.0);
    //std::fill(lower3, lower3 + n * numLowerBounds, 0.0);
}