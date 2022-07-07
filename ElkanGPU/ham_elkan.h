#ifndef HAM_ELKAN_H
#define HAM_ELKAN_H

#include "triangle_inequality_base_kmeans.h"

class HamElkan : public TriangleInequalityBaseKmeans {
public:
    HamElkan() : centerCenterDistDiv2(NULL) {}
    virtual ~HamElkan() { free(); }
    virtual void free();
    virtual void initialize(Dataset const* aX, unsigned short aK, unsigned short* initialAssignment, int aNumThreads);
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

