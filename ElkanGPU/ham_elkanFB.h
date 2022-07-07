#ifndef FB1_ELKAN_KMEANSHAM_H
#define FB1_ELKAN_KMEANSHAM_H


#include "triangle_inequality_base_kmeans.h"

class HamElkanFB : public TriangleInequalityBaseKmeans {
public:
    HamElkanFB() : centerCenterDistDiv2(NULL) {}
    //ElkanKmeans_newbound() : oldcenterCenterDistDiv2(NULL) {}
    virtual ~HamElkanFB() { free(); }
    virtual void free();
    virtual void initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "fbelkanham"; }

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
    double* d_maxoldcenter2newcenterDis;
    double* oldcenters;
    double* d_oldcenters;
    //double *lower2;
    double* d_distances;
    double* d_distances2;
    bool* d_calculated;
    double* lower;
    double* d_lower;
    double* ub_old;
    double* d_ub_old;
    int move_centers_newbound(double* oldcenters, double* oldcenter2newcenterDis);

};

#endif

