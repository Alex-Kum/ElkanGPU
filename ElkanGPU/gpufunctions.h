#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "yyheader.h"
#include <stdio.h>

#define DTYPE double
#define BLOCKSIZE 256
#define DISTANCES 0

__global__ void setTestL(int* test) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid %i\n", i);

    /*if (i < 10000) {
        test[i] = 2;
    }*/
    if (i == 2)
        test[i] = 999;

}

__device__ void atomicMax(double* const address,
    const double value)
{
    //printf("genutzt");
    if (*address >= value)
        return;

    unsigned long long int* const address_as_i = (unsigned long long int*)address;
    unsigned long long int old = *address_as_i, assumed;

    do
    {
        assumed = old;
        if (__longlong_as_double(assumed) >= value)
            break;

        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

/*#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));

         Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif*/

__global__ void clearCentCalcDataLloyd2(DTYPE* newCentSum,
    unsigned int* newCentCount,
    const int numCent,
    const int numDim, int* test)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    printf("tid ");
}



__global__ void clearCentCalcDataLloyd(DTYPE* newCentSum,
    unsigned int* newCentCount,
    const int numCent,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);

    //printf("tid ");

    if (tid >= numCent)
        return;

    //if (tid < 10000) {
    //    printf("test %i", tid);
    //    test[tid] = 5;
    //}

    unsigned int dimIndex;

    for (dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
        newCentSum[(tid * numDim) + dimIndex] = 0.0;
    }
    newCentCount[tid] = 0;
}

__global__ void clearCentCalcData(DTYPE* newCentSum,
    DTYPE* oldCentSum,
    unsigned int* newCentCount,
    unsigned int* oldCentCount,
    const int numCent,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numCent)
        return;

    unsigned int dimIndex;

    for (dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
        newCentSum[(tid * numDim) + dimIndex] = 0.0;
        oldCentSum[(tid * numDim) + dimIndex] = 0.0;
    }
    newCentCount[tid] = 0;
    oldCentCount[tid] = 0;
}

__global__ void clearDriftArr(DTYPE* maxDriftArr,
    const int numGrp)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numGrp)
        return;

    maxDriftArr[tid] = 0.0;
}

__device__ DTYPE calcDis(DTYPE* vec1, DTYPE* vec2, const int numDim)
{
    int index;
    DTYPE total = 0;
    DTYPE square;

    for (index = 0; index < numDim; index++)
    {
        square = (vec1[index] - vec2[index]);
        total += square * square;
    }

    return sqrt(total);
}

__global__ void initRunKernel(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* pointLwrs,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    unsigned int centIndex;

    DTYPE currDistance;
    pointInfo[tid].uprBound = INFINITY;

    for (centIndex = 0; centIndex < numCent; centIndex++)
    {
        // calculate euclidean distance between point and centroid
        currDistance = calcDis(&pointData[tid * numDim],
            &centData[centIndex * numDim],
            numDim);
        if (currDistance < pointInfo[tid].uprBound)
        {
            // make the former current min the new
            // lower bound for it's group
            if (pointInfo[tid].uprBound != INFINITY)
                pointLwrs[(tid * numGrp) + centInfo[pointInfo[tid].centroidIndex].groupNum] =
                pointInfo[tid].uprBound;

            // update assignment and upper bound
            pointInfo[tid].centroidIndex = centIndex;
            pointInfo[tid].uprBound = currDistance;
        }
        else if (currDistance < pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum])
        {
            pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum] = currDistance;
        }
    }
}

__global__ void checkConverge(PointInfo* pointInfo,
    unsigned int* conFlag,
    const int numPnt)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    /*if (pointInfo[tid].oldCentroid != pointInfo[tid].centroidIndex) {
        unsigned int old = *conFlag;
        if (old == 0) {
            *conFlag = 1;
        }
    }*/
    if (pointInfo[tid].oldCentroid != pointInfo[tid].centroidIndex) {
        atomicCAS(conFlag, 0, 1);
    }

}

__device__ void pointCalcsFull(PointInfo* pointInfoPtr,
    CentInfo* centInfo,
    DTYPE* pointDataPtr,
    DTYPE* pointLwrPtr,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    unsigned int* groupArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{


    unsigned int grpIndex, centIndex;

    DTYPE compDistance;
    DTYPE oldLwr = INFINITY;
    DTYPE oldCentUpr = pointInfoPtr->uprBound;
    DTYPE oldCentLwr = pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum];

    // loop through all the groups
    for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
    {
        // if the group is marked as going through the group filter
        if (groupArr[grpIndex])
        {
            // save the former lower bound pre-update
            if (grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum)
                oldLwr = oldCentLwr + maxDriftArr[grpIndex];
            else
                oldLwr = pointLwrPtr[grpIndex] + maxDriftArr[grpIndex];

            // reset the group's lower bound in order to find the new lower bound
            pointLwrPtr[grpIndex] = INFINITY;

            if (grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum &&
                pointInfoPtr->oldCentroid != pointInfoPtr->centroidIndex)
                pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum] = oldCentUpr;

            // loop through all the group's centroids
            for (centIndex = 0; centIndex < numCent; centIndex++)
            {
                // if the cluster is the cluster already assigned
                // at the start of this iteration
                if (centIndex == pointInfoPtr->oldCentroid)
                    continue;

                // if the cluster is a part of the group being checked now
                if (grpIndex == centInfo[centIndex].groupNum)
                {
                    // local filtering condition
                    if (pointLwrPtr[grpIndex] < oldLwr - centInfo[centIndex].drift)
                        continue;

                    // perform distance calculation
                    compDistance = calcDis(pointDataPtr, &centData[centIndex * numDim], numDim);

                    if (compDistance < pointInfoPtr->uprBound)
                    {
                        pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] = pointInfoPtr->uprBound;
                        pointInfoPtr->centroidIndex = centIndex;
                        pointInfoPtr->uprBound = compDistance;
                    }
                    else if (compDistance < pointLwrPtr[grpIndex])
                    {
                        pointLwrPtr[grpIndex] = compDistance;
                    }
                }
            }
        }
    }
}

__global__ void assignPointsFull(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* pointLwrs,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    DTYPE tmpGlobLwr = INFINITY;

    int btid = threadIdx.x;
    unsigned int index;

    extern __shared__ unsigned int groupLclArr[];

    // reassign point's former centroid before finding new centroid
    pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

    // update points upper bound ub = ub + drift(b(x))
    pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for (index = 0; index < numGrp; index++)
    {
        // subtract lowerbound by group's drift
        pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

        // if the lowerbound is less than the temp global lower,
        if (pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
        }
    }

    // if the global lower bound is less than the upper bound
    if (tmpGlobLwr < pointInfo[tid].uprBound)
    {
        // tighten upper bound ub = d(x, b(x))
        pointInfo[tid].uprBound =
            calcDis(&pointData[tid * numDim],
                &centData[pointInfo[tid].centroidIndex * numDim], numDim);

        // if the lower bound is less than the upper bound
        if (tmpGlobLwr < pointInfo[tid].uprBound)
        {
            // loop through groups
            for (index = 0; index < numGrp; index++)
            {
                // if the lower bound is less than the upper bound
                // mark the group to go through the group filter
                if (pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
                    groupLclArr[index + (btid * numGrp)] = 1;
                else
                    groupLclArr[index + (btid * numGrp)] = 0;
            }

            // execute point calcs given the groups
            pointCalcsFull(&pointInfo[tid], centInfo, &pointData[tid * numDim],
                &pointLwrs[tid * numGrp], centData, maxDriftArr,
                &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim);
        }
    }
}

void calcWeightedMeansLloyd(CentInfo* newCentInfo,
    CentInfo** allCentInfo,
    DTYPE* newCentData,
    DTYPE* oldCentData,
    DTYPE** allCentData,
    const int numCent,
    const int numDim,
    const int numGPU)
{
    DTYPE numerator = 0;
    DTYPE denominator = 0;
    DTYPE zeroNumerator = 0;
    int zeroCount = 0;

    for (int i = 0; i < numCent; i++)
    {
        for (int j = 0; j < numDim; j++)
        {
            oldCentData[(i * numDim) + j] = newCentData[(i * numDim) + j];
        }
    }

    for (int i = 0; i < numGPU; i++)
    {
        for (int j = 0; j < numCent; j++)
        {
            newCentInfo[j].count += allCentInfo[i][j].count;

            newCentInfo[j].groupNum = allCentInfo[0][j].groupNum;
        }
    }

    for (int j = 0; j < numCent; j++)
    {
        for (int k = 0; k < numDim; k++)
        {
            for (int l = 0; l < numGPU; l++)
            {
                if (allCentInfo[l][j].count == 0)
                {
                    zeroCount++;
                    zeroNumerator += allCentData[l][(j * numDim) + k];
                }

                numerator +=
                    allCentData[l][(j * numDim) + k] * allCentInfo[l][j].count;

                denominator += allCentInfo[l][j].count;
            }

            if (denominator != 0)
            {
                newCentData[(j * numDim) + k] = numerator / denominator;
            }

            else
            {
                newCentData[(j * numDim) + k] = zeroNumerator / zeroCount;
            }

            zeroCount = 0;
            zeroNumerator = 0;
            numerator = 0;
            denominator = 0;
        }

        /*
  newCentInfo[j].drift = calcDisCPU(&newCentData[j*numDim],
                                       &oldCentData[j*numDim],
                                       numDim);
        */
    }
}

__device__ void pointCalcsSimple(PointInfo* pointInfoPtr,
    CentInfo* centInfo,
    DTYPE* pointDataPtr,
    DTYPE* pointLwrPtr,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    unsigned int* groupArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{

    unsigned int index;
    DTYPE compDistance;

    for (index = 0; index < numGrp; index++)
    {
        if (groupArr[index])
        {
            pointLwrPtr[index] = INFINITY;
        }
    }

    for (index = 0; index < numCent; index++)
    {
        if (groupArr[centInfo[index].groupNum])
        {
            if (index == pointInfoPtr->oldCentroid)
                continue;

            compDistance = calcDis(pointDataPtr,
                &centData[index * numDim],
                numDim);

            if (compDistance < pointInfoPtr->uprBound)
            {
                pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] =
                    pointInfoPtr->uprBound;
                pointInfoPtr->centroidIndex = index;
                pointInfoPtr->uprBound = compDistance;
            }
            else if (compDistance < pointLwrPtr[centInfo[index].groupNum])
            {
                pointLwrPtr[centInfo[index].groupNum] = compDistance;
            }
        }
    }
}

__global__ void assignPointsSimple(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* pointLwrs,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    DTYPE tmpGlobLwr = INFINITY;
   // DTYPE tmpGlobLwr = 0;
    unsigned int btid = threadIdx.x;
    unsigned int index;


    pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

    extern __shared__ unsigned int groupLclArr[];

    // update points upper bound
    //pointInfo[tid].oldUprBound = pointInfo[tid].uprBound;
    pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for (index = 0; index < numGrp; index++)
    {
        // subtract lowerbound by group's drift
        pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

        // if the lowerbound is less than the temp global lower,
        if (pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
        }
    }

    // if the global lower bound is less than the upper bound
    if (tmpGlobLwr < pointInfo[tid].uprBound)
    //vielleicht auch zweites?
    //if (tmpGlobLwr < pointInfo[tid].uprBound && pointInfo[tid].uprBound >= oldcenter2newcenterDis[assignment[i] * k + j] - pointInfo[tid].oldUprBound)
    {
        // tighten upper bound ub = d(x, b(x))
        pointInfo[tid].uprBound =
            calcDis(&pointData[tid * numDim],
                &centData[pointInfo[tid].centroidIndex * numDim],
                numDim);

        // if the lower bound is less than the upper bound
        if (tmpGlobLwr < pointInfo[tid].uprBound)
        {
            // loop through groups
            for (index = 0; index < numGrp; index++)
            {
                // if the lower bound is less than the upper bound
                // mark the group to go through the group filter
                if (pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
                    groupLclArr[index + (btid * numGrp)] = 1;
                else
                    groupLclArr[index + (btid * numGrp)] = 0;
            }

            // execute point calcs given the groups
            pointCalcsSimple(&pointInfo[tid], centInfo, &pointData[tid * numDim],
                &pointLwrs[tid * numGrp], centData, maxDriftArr,
                &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim);
        }

    }
}

__global__ void calcNewCentroids(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* centData,
    DTYPE* oldCentData,
    DTYPE* oldSums,
    DTYPE* newSums,
    DTYPE* maxDriftArr,
    unsigned int* oldCounts,
    unsigned int* newCounts,
    const int numCent,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);

    if (tid >= numCent)
        return;

    DTYPE oldFeature, oldSumFeat, newSumFeat, compDrift;

    unsigned int dimIndex;

    // create the new centroid vector
    for (dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
        if (newCounts[tid] > 0)
        {
            oldCentData[(tid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];

            oldFeature = centData[(tid * numDim) + dimIndex];
            oldSumFeat = oldSums[(tid * numDim) + dimIndex];
            newSumFeat = newSums[(tid * numDim) + dimIndex];

            centData[(tid * numDim) + dimIndex] =
                (oldFeature * oldCounts[tid] - oldSumFeat + newSumFeat) / newCounts[tid];
        }
        else
        {
            // no change to centroid
            oldCentData[(tid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];
        }
        newSums[(tid * numDim) + dimIndex] = 0.0;
        oldSums[(tid * numDim) + dimIndex] = 0.0;
    }


    // calculate the centroid's drift
    compDrift = calcDis(&oldCentData[tid * numDim],
        &centData[tid * numDim],
        numDim);

    atomicMax(&maxDriftArr[centInfo[tid].groupNum], compDrift);
    /*if (compDrift > maxDriftArr[centInfo[tid].groupNum]) {
        maxDriftArr[centInfo[tid].groupNum] = compDrift;
    }*/

    // set the centroid's vector to the new vector
    centInfo[tid].drift = compDrift;
    centInfo[tid].count = newCounts[tid];

    // clear the count and the sum arrays
    oldCounts[tid] = 0;
    newCounts[tid] = 0;

}


__global__ void calcNewCentroidsLloyd(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* centData,
    DTYPE* newSums,
    unsigned int* newCounts,
    const int numCent,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numCent)
        return;

    unsigned int dimIndex;

    for (dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
        if (newCounts[tid] > 0)
        {
            centData[(tid * numDim) + dimIndex] =
                newSums[(tid * numDim) + dimIndex] / newCounts[tid];
        }
        // otherwise, no change
        newSums[(tid * numDim) + dimIndex] = 0.0;
    }

    centInfo[tid].count = newCounts[tid];

    newCounts[tid] = 0;
}

__global__ void calcCentDataLloyd(PointInfo* pointInfo,
    DTYPE* pointData,
    DTYPE* newSums,
    unsigned int* newCounts,
    const int numPnt,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);

    if (tid >= numPnt)
        return;

    unsigned int dimIndex;

    // atomicAdd 1 to new counts corresponding
    atomicAdd(&newCounts[pointInfo[tid].centroidIndex], 1);

    // for all values in the vector
    for (dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
        atomicAdd(&newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex],
            pointData[(tid * numDim) + dimIndex]);
    }
}

__global__ void assignPointsLloyd(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    DTYPE currMin = INFINITY;
    DTYPE currDis;

    unsigned int index;

    // reassign point's former centroid before finding new centroid
    pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

    for (index = 0; index < numCent; index++)
    {
        currDis = calcDis(&pointData[tid * numDim],
            &centData[index * numDim],
            numDim);
        if (currDis < currMin)
        {
            pointInfo[tid].centroidIndex = index;
            currMin = currDis;
        }
    }
}

__global__ void calcCentData(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* oldSums,
    DTYPE* newSums,
    unsigned int* oldCounts,
    unsigned int* newCounts,
    const int numPnt,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    unsigned int dimIndex;

    // atomicAdd 1 to old and new counts corresponding
    if (pointInfo[tid].oldCentroid >= 0)
        atomicAdd(&oldCounts[pointInfo[tid].oldCentroid], 1);
    //    oldCounts[pointInfo[tid].oldCentroid] += 1;


    //newCounts[pointInfo[tid].centroidIndex] += 1;
    atomicAdd(&newCounts[pointInfo[tid].centroidIndex], 1);

    // if old assignment and new assignment are not equal
    if (pointInfo[tid].oldCentroid != pointInfo[tid].centroidIndex)
    {
        // for all values in the vector
        for (dimIndex = 0; dimIndex < numDim; dimIndex++)
        {
            // atomic add the point's vector to the sum count
            if (pointInfo[tid].oldCentroid >= 0)
            {
                //REMOVEDATOMIC
                atomicAdd(&oldSums[(pointInfo[tid].oldCentroid * numDim) + dimIndex],
                    pointData[(tid * numDim) + dimIndex]);
                //oldSums[(pointInfo[tid].oldCentroid * numDim) + dimIndex] += pointData[(tid * numDim) + dimIndex];

            }
            //REMOVEDATOMIC
            //double* one = &newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex];
            //double two = pointData[(tid * numDim) + dimIndex];
            //atomicAdd(one,two);
            atomicAdd(&newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex],
                pointData[(tid * numDim) + dimIndex]);
            //newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex] += pointData[(tid * numDim) + dimIndex];
        }
    }

}


__global__ void addKer(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void multip(const int* a, const int* b, int* c) {
    int i = threadIdx.x;
    if (i < 5) {
        c[i] = a[i] * b[i];
    }
}

/*__global__ void zweiD(const int* a, const int* b, int* c) {
    int i = threadIdx.x;
    if (i < 4) {
        c[i] = a[i] * b[i];
    }
}*/

__device__ void addV(double* x, double* y, int dim) {
    double const* end = x + dim;
    while (x < end) {
        *(x++) += *(y++);
    }
}

__device__ void subV(double* x, double* y, int dim) {
    double const* end = x + dim;
    while (x < end) {
        *(x++) -= *(y++);
    }
}

__device__ void addVectorss(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        *(a++) += *(b++);
    }
}

//__device__ void subVectorss(double* a, double const* b, int d) {
//    double const* end = a + d;
//    while (a < end) {
//        //*(a++) -= *(b++);
//        double bVal = *(b++);
//        atomicSub(a, bVal);
//        a++;
//    }
//}

__device__ void addVectorsAtomic(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        //*(a++) += *(b++);
        double bVal = *(b++);
        atomicAdd(a, bVal);
        a++;
    }
}

__device__ void subVectorsAtomic(double* a, double const* b, int d) {
    double const* end = a + d;
    while (a < end) {
        double bVal = *(b++);
        atomicAdd(a, -bVal);
        a++;
    }
}

__device__ double dist(const double* data, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * data[y * dim + i];
    }
    return result;
}

__device__ double distp2c(const double* data, const double* center, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * center[y * dim + i];
    }
    return result;
}

__device__ double dist22(const double* data, const double* center, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        double diff = data[x * dim + i] - center[y * dim + i];
        result += diff * diff;
    }
    return result;
}

__device__ double dist33(const double* data, int x, int y, int dim) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        double diff = data[x * dim + i] - data[y * dim + i];
        result += diff * diff;
    }
    return result;
}

/*__global__ void dist2(const double* data, int x, int y, int dim, int n, double* res) {
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        result += data[x * dim + i] * data[y * dim + i];
    }
    *res = result;
}*/

__global__ void assignPointsSuper(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* pointLwrs,
    DTYPE* centData,
    DTYPE* maxDrift,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * BLOCKSIZE);
    if (tid >= numPnt)
        return;

    // point calc variables
    int centIndex;
    DTYPE compDistance;

    // set centroid's old centroid to be current assignment
    pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

    // update bounds
    pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;
    pointLwrs[tid * numGrp] -= *maxDrift;

    if (pointLwrs[tid * numGrp] < pointInfo[tid].uprBound)
    {
        // tighten upper bound
        pointInfo[tid].uprBound =
            calcDis(&pointData[tid * numDim],
                &centData[pointInfo[tid].centroidIndex * numDim], numDim);


        if (pointLwrs[(tid * numGrp)] < pointInfo[tid].uprBound)
        {
            // to get a new lower bound
            pointLwrs[tid * numGrp] = INFINITY;

            for (centIndex = 0; centIndex < numCent; centIndex++)
            {
                // do not calculate for the already assigned cluster
                if (centIndex == pointInfo[tid].oldCentroid)
                    continue;

                compDistance = calcDis(&pointData[tid * numDim],
                    &centData[centIndex * numDim],
                    numDim);

                if (compDistance < pointInfo[tid].uprBound)
                {
                    pointLwrs[tid * numGrp] = pointInfo[tid].uprBound;
                    pointInfo[tid].centroidIndex = centIndex;
                    pointInfo[tid].uprBound = compDistance;
                }
                else if (compDistance < pointLwrs[tid * numGrp])
                {
                    pointLwrs[tid * numGrp] = compDistance;
                }
            }
        }
    }
}

//__global__ void innerProdFBHam(double* centerCenterDist, double* s, const double* data, int dim, int n) {
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int c1 = index / n;
//    int c2 = index % n;
//
//    if (c1 != c2 && index < n * n) {
//        double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
//        centerCenterDist[index] = sqrt(distance) / 2.0;
//        if (centerCenterDist[index] < s[c1]) {
//            s[c1] = centerCenterDist[index];
//        }
//    }
//}

__global__ void innerProdMOHam(double* centerCenterDist, double* oldcenterCenterDistDiv2, double* maxoldcenterCenterDistDiv2, double* s, const double* data, int dim, int k, int n, double* centerMovement) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        //double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
        double distance = dist33(data, c1, c2, dim);
        oldcenterCenterDistDiv2[c1 * k + c2] = centerCenterDist[c1 * k + c2];
        maxoldcenterCenterDistDiv2[c1] = s[c1] - centerMovement[c1];
        centerCenterDist[index] = sqrt(distance) / 2.0;
        if (centerCenterDist[index] < s[c1]) {
            s[c1] = centerCenterDist[index];
        }
    }
}


__global__ void innerProdMO(double* centerCenterDist, double* oldcenterCenterDistDiv2, double* s, const double* data, int dim, int k, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        //double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
        double distance = dist33(data, c1, c2, dim);
        oldcenterCenterDistDiv2[c1 * k + c2] = centerCenterDist[c1 * k + c2];
        centerCenterDist[index] = sqrt(distance) / 2.0;
        if (centerCenterDist[index] < s[c1]) {
            s[c1] = centerCenterDist[index];
        }
    }
}

__global__ void innerProdFBHam(double* centerCenterDist, double* s, const double* data, int dim, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;
   
    if (c1 != c2 && index < n * n) {
        double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
        centerCenterDist[index] = sqrt(distance) / 2.0;
        if (centerCenterDist[index] < s[c1]) {
            s[c1] = centerCenterDist[index];
        }
    }
}

__global__ void innerProd(double* centerCenterDist, double* s, const double* data, int dim, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = index / n;
    int c2 = index % n;

    if (c1 != c2 && index < n * n) {
        //double distance = dist(data, c1, c1, dim) - 2 * dist(data, c1, c2, dim) + dist(data, c2, c2, dim);
        double distance = dist33(data, c1, c2, dim);
        centerCenterDist[index] = sqrt(distance) / 2.0;
        if (centerCenterDist[index] < s[c1]) {
            s[c1] = centerCenterDist[index];
        }
    }
}

__device__ double innerProdp2c(double* data, double* center, int x, int y, int dim) {
    return dist(data, x, x, dim) - 2 * distp2c(data, center, x, y, dim) + dist(center, y, y, dim);
}

//__device__ void updateBound( double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n) {
//        upper[i] += centerMovement[assignment[i]];
//        for (int j = 0; j < k; ++j) {
//            lower[i * numLowerBounds + j] -= centerMovement[j];
//        }
//    }
//}

__global__ void elkanFBMoveAddition(double* oldcenters, double* oldcenter2newcenterDis, double* center, int d, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / n;
    int c2 = i % n;

    if (c1 != c2 && i < n * n) {
        //oldcenter2newcenterDis[c1 * k + c2] = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            oldcenter2newcenterDis[c1 * k + c2] += (oldcenters[c1 * d + dim] - center[c2 * d + dim])  * (oldcenters[c1 * d + dim] - center[c2 * d + dim]);
        }
        oldcenter2newcenterDis[c1 * k + c2] = sqrt(oldcenter2newcenterDis[c1 * k + c2]);
    }    
}

__global__ void elkanFBMoveAdditionHam(double* oldcenters, double* oldcenter2newcenterDis, double* maxoldcenter2newcenterDis, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double maxCenterDis = INFINITY;
        for (int j = 0; j < k; j++) {
            if (oldcenter2newcenterDis[i * k + j] < maxCenterDis) {
                maxCenterDis = oldcenter2newcenterDis[i * k + j];
            }
        }              
        maxoldcenter2newcenterDis[i] = maxCenterDis;        
    }    
}

__global__ void updateBoundHamShared(double* data, double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int dim, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        double maxMovement = 0;
        upper[i] += movement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            /*if (j == assignment[i])
                continue;*/
            if (movement[j] > maxMovement)
                maxMovement = movement[j];
        }
        lower[i] -= maxMovement;
    }
}

__global__ void updateBoundHam(double* data, double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int dim, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {        
        double maxMovement = 0;
        upper[i] += centerMovement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            /*if (j == assignment[i])
                continue;*/
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }
        lower[i] -= maxMovement;
    }
}



__global__ void updateBound(double* data, double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int dim, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        
        /*double dist = distp2c(data, lastExactCentroid, i, i, dim);                     
        upper[i] = dist;*/
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
}

__global__ void updateBoundShared(double* data, double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int dim, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {

        /*double dist = distp2c(data, lastExactCentroid, i, i, dim);
        upper[i] = dist;*/
        upper[i] += movement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= movement[j];
        }
    }
}

__global__ void updateBoundFBShared(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double movement[256];
    if (threadIdx.x < 256) {
        movement[threadIdx.x] = centerMovement[threadIdx.x];
    }
    __syncthreads();

    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += movement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= movement[j];
        }
    }
}

__global__ void updateBoundFB(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
}

__global__ void updateBoundFBHam(double* lower, double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int numLowerBounds, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double maxMovement = 0;
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];

        for (int j = 0; j < k; ++j) {
            if (j == assignment[i])
                continue;
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }
        lower[i] -= maxMovement;
    }
}


__global__ void updateBoundMO(double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
    }
}

__global__ void calculateFilterMO(unsigned short* assignment, double* upper,
    double* s, double* maxoldcenter2newcenterDis, double* ub_old, bool* calculated, int n, unsigned short* closest2, double* maxoldcenterCenterDistDiv2, double* maxcenterMovement) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i] && upper[i] >= 2.0 * (maxoldcenterCenterDistDiv2[assignment[i]]) - ub_old[i] - *maxcenterMovement;
    }
}

__global__ void elkanFunMOHamKCalc(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        if (calculated[c1]) {
            distances[c1 * k + c2] = sqrt(dist22(data, center, c1, c2, dim));
        }
    }
}

__global__ void elkanFunMOHamBounds(double* upper, double* distances, bool* calculated, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (calculated[i]) {
            double closestDistance = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = distances[i * k + j];
                if (curDistance < closestDistance) {
                    closestDistance = curDistance;
                    closest2[i] = j;
                }
            }
            upper[i] = closestDistance;
        }
    }
}

__global__ void elkanFunMOHam(double* data, double* center, unsigned short* assignment, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* maxoldcenterCenterDistDiv2, double* ub_old, double* maxcenterMovement, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        unsigned long long int c;
        //if (upper[i] > s[closest2[i]] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i] && upper[i] >= 2.0 * (maxoldcenterCenterDistDiv2[assignment[i]]) - ub_old[i] - *maxcenterMovement) {
        if (upper[i] > s[closest2[i]] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i] && upper[i] >= 2.0 * (maxoldcenterCenterDistDiv2[assignment[i]]) - ub_old[i]) {
            double closestDistance = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist22(data, center, i, j, dim));
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
                if (curDistance < closestDistance) {
                    closestDistance = curDistance;
                    closest2[i] = j;
                }
            }
            upper[i] = closestDistance;
        }
    }
}

__global__ void elkanFunMOShared(double* data, double* center, unsigned short* assignment, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* oldcenterCenterDistDiv2, double* ub_old, double* centerMovement, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int blockSize = 4 * 32;

    __shared__ int counter;
    __shared__ int calculate[blockSize];

    counter = 0;
    calculate[threadIdx.x] = -1;
    __syncthreads();

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= s[closest2[i]]) {
            index = atomicAdd(&counter, 1);
            calculate[index] = i;
        }

    }
    __syncthreads();

    if (i < n) {
        closest2[calculate[threadIdx.x]] = assignment[calculate[threadIdx.x]];
        bool r = true;

        if (calculate[threadIdx.x] >= 0) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[calculate[threadIdx.x]]) { continue; }
                if (upper[calculate[threadIdx.x]] <= 2.0 * (oldcenterCenterDistDiv2[assignment[calculate[threadIdx.x]] * k + j]) - ub_old[calculate[threadIdx.x]] - centerMovement[j]) { continue; }
                if (upper[calculate[threadIdx.x]] <= oldcenter2newcenterDis[assignment[calculate[threadIdx.x]] * k + j] - ub_old[calculate[threadIdx.x]]) { continue; }
                if (upper[calculate[threadIdx.x]] <= centerCenterDistDiv2[closest2[calculate[threadIdx.x]] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[calculate[threadIdx.x]] = sqrt(dist22(data, center, calculate[threadIdx.x], closest2[calculate[threadIdx.x]], dim));
                    //lower[calculate[threadIdx.x] * k + closest2[calculate[threadIdx.x]]] = upper[calculate[threadIdx.x]];
                    r = false;
                    if ((upper[calculate[threadIdx.x]] <= 2.0 * (oldcenterCenterDistDiv2[assignment[calculate[threadIdx.x]] * k + j]) - ub_old[calculate[threadIdx.x]] - centerMovement[j])
                        || (upper[calculate[threadIdx.x]] <= centerCenterDistDiv2[closest2[calculate[threadIdx.x]] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                double dist = sqrt(dist22(data, center, calculate[threadIdx.x], j, dim));
                if (dist < upper[calculate[threadIdx.x]]) {
                    closest2[calculate[threadIdx.x]] = j;
                    upper[calculate[threadIdx.x]] = dist;
                }
            }
        }
    }
}

//    if (i < n) {
//        //unsigned short closest = assignment[i];
//        closest2[i] = assignment[i];
//        double localUpper = upper[i];
//        bool r = true;
//        unsigned long long int c;
//
//        if (localUpper > s[closest2[i]]) {
//            //upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
//            for (int j = 0; j < k; ++j) {
//                if (j == closest2[i]) { continue; }
//                if (localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) { continue; }
//                if (localUpper <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }  //upper[i] <= lower[i * k + j] ||
//                if (localUpper <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }
//
//                // ELKAN 3(a)
//                if (r) {
//                    //localUpper = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
//                    localUpper = sqrt(dist22(data, center, i, closest2[i], dim));
//#if DISTANCES
//                    c = atomicAdd(countDistances, 1);
//                    if (c == 18446744073709551615) {
//                        printf("OVERFLOW");
//                    }
//#endif
//                    r = false;
//                    if ((localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) || (localUpper <= centerCenterDistDiv2[closest2[i] * k + j])) {
//                        continue;
//                    }
//                }
//
//                // ELKAN 3(b)
//                //lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
//                //double inner2 = sqrt(innerProdp2c(data, center, i, j, dim));
//                double inner = sqrt(dist22(data, center, i, j, dim));
//#if DISTANCES
//                c = atomicAdd(countDistances, 1);
//                if (c == 18446744073709551615) {
//                    printf("OVERFLOW");
//                }
//#endif
//                if (inner < localUpper) {
//                    closest2[i] = j;
//                    localUpper = inner;
//                }
//            }
//        }
//        upper[i] = localUpper;
//    }
//}

__global__ void elkanFunMO(double* data, double* center, unsigned short* assignment, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* oldcenterCenterDistDiv2, double* ub_old, double* centerMovement, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //unsigned short closest = assignment[i];
        closest2[i] = assignment[i];
        double localUpper = upper[i];
        bool r = true;
        unsigned long long int c;
        
        if (localUpper > s[closest2[i]]) {
            //upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) { continue; }
                if (localUpper <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }  //upper[i] <= lower[i * k + j] ||
                if (localUpper <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    //localUpper = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    localUpper = sqrt(dist22(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    r = false;
                    if ((localUpper <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) || (localUpper <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                //double inner2 = sqrt(innerProdp2c(data, center, i, j, dim));
                double inner = sqrt(dist22(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (inner < localUpper) {
                    closest2[i] = j;
                    localUpper = inner;
                }
            }
        }
        upper[i] = localUpper;
    }
}

__global__ void changeAssFirst(double* data, unsigned short* assignment, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {

        if (assignment[i] != closest2[i]) {
            //unsigned short oldAssignment = assignment[i];
           // --clusterSize[assignment[i]];
            ++clusterSize[closest2[i]];
            //assignment[i] = closest2[i];
            //atomicSub(&clusterSize[assignment[i]], 1);
            //atomicAdd(&clusterSize[closest2[i]], 1);            
        }
    }
}

__global__ void changeAss(double* data, unsigned short* assignment, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int dim, int n, int offset) {
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /*if (i == 24) {
            printf("clustersize cluster von 24 vorher %i\n", clusterSize[assignment[24]]);
        }*/
        if (assignment[i] != closest2[i]) {
            unsigned short oldAssignment = assignment[i];
           /* if (i == 24) {
                printf("sumnewcenters oldAssign 24 vorher %f\n", sumNewCenters[oldAssignment * dim]);
            }*/
            atomicSub(&clusterSize[assignment[i]], 1);
            atomicAdd(&clusterSize[closest2[i]], 1);
            double* xp = data + i * dim;
            assignment[i] = closest2[i];

            subVectorsAtomic(sumNewCenters + oldAssignment * dim, xp, dim);
            addVectorsAtomic(sumNewCenters + closest2[i] * dim, xp, dim);
            /*if (i == 24) {
                printf("sumnewcenters oldAssign 24 nachher %f\n", sumNewCenters[oldAssignment * dim]);
            }*/
        }
       /* if (i == 24) {
            printf("clustersize cluster von 24 nachher %i\n", clusterSize[assignment[24]]);            
            printf("sumnewcenters closest 24 nachher %f\n", sumNewCenters[closest2[24] * dim]);
        }*/
    }
}

__global__ void elkanFun(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int* clusterSize, double* sumNewCenters, double* centerMovement, int k, int dim, int n, int numlower, bool* converged, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //unsigned short closest = assignment[i];
        closest2[i] = assignment[i];
        bool r = true;

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    //upper[i] = sqrt(pointCenterDist2(i, closest));
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(pointCenterDist2(i, j));
                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
        //if (i == 500) {
        //    printf("Hallo Welt!\n assignment: %i, closest: %i", assignment[i], closest2[i]);
        //}
       /* if (i == 0) {
            printf("sizeAss: %i, sizeClos: %i\n", clusterSize[0][assignment[i]], clusterSize[0][closest2[i]]);
        }
        if (assignment[i] != closest2[i]) {
            unsigned short oldAssignment = assignment[i];
            --clusterSize[0][assignment[i]];
            ++clusterSize[0][closest2[i]];
            double* xp = data + i * dim;
            //if (i == 500) {
           //     printf("oldAssignment: %i, assignment: %i, cSize ass: %i, cSize clos: %i, xp %p\n", oldAssignment, closest2[i], clusterSize[0][assignment[i]], clusterSize[0][closest2[i]], xp);
           // }
            assignment[i] = closest2[i];

            subVectorss(sumNewCenters + oldAssignment * dim, xp, dim);
            addVectorss(sumNewCenters + closest2[i] * dim, xp, dim);

        }*/


        /*int furthestMovingCenter = 0;
        for (int j = 0; j < k; ++j) {
            centerMovement[j] = 0.0;
            int totalClusterSize = 0;
                totalClusterSize += clusterSize[0][j];

            if (totalClusterSize > 0) {
                for (int d = 0; d < dim; ++d) {
                    double z = 0.0;
                    //z += (*sumNewCenters[0])(j, d);
                    z += sumNewCenters[j * dim + d];
                    z /= totalClusterSize;
                    //centerMovement[j] += (z - (*centers)(j, d)) * (z - (*centers)(j, d));//calculate distance
                    centerMovement[j] += (z - center[j * dim + d]) * (z - center[j * dim + d]);
                    //(*centers)(j, dim) = z; //update new centers
                    center[j * dim + d] = z;
                }
            }
            centerMovement[j] = sqrt(centerMovement[j]);

            if (centerMovement[furthestMovingCenter] < centerMovement[j]) {
                furthestMovingCenter = j;
            }
        }
        *converged = (0.0 == centerMovement[furthestMovingCenter]);
        if (!(*converged)) {
            upper[i] += centerMovement[assignment[i]];
            for (int j = 0; j < k; ++j) {
                lower[i * numlower + j] -= centerMovement[j];
            }
        }*/

    }
}


__global__ void wtfTest(bool* test) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("test i = %i wird auf true gesetzt\n", i);
    test[i] = true;
}

__global__ void setTest(int* test, unsigned short* assignment, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000) {
        // unsigned short t = 2;
         //closest2[i] = 2;
        closest2[i] = assignment[i];
        test[i] = 2;
    }
}

__global__ void setTesttt(int* test, unsigned short* arr1, unsigned short* arr2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000) {
        arr2[i] = arr1[i];
        test[i] = 5;
    }
}

__global__ void elkanMoveCenterFB(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, double* oldcenters, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) {
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                oldcenters[i * dim + d] = center[i * dim + d];
                center[i * dim + d] = z;
            }
        }
        centerMovement[i] = sqrt(centerMovement[i]);

        if (centerMovement[i] > 0)
            *converged = false;
    }
}

__global__ void elkanMoveCenterMOHamMax(double* centerMovement, double* maxcenterMovement, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        *maxcenterMovement = 0.0;
        for (int j = 0; j < k; j++) {
            if (centerMovement[j] > *maxcenterMovement)
                *maxcenterMovement = centerMovement[j];
        }
    }
}

__global__ void elkanMoveCenterMOHam(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, double* oldcenters, double* maxcenterMovement, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) {
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                oldcenters[i * dim + d] = center[i * dim + d];
                center[i * dim + d] = z;
            }
        }
        centerMovement[i] = sqrt(centerMovement[i]);
        if (centerMovement[i] > *maxcenterMovement) {    //race
            *maxcenterMovement = centerMovement[i];
        }
        if (centerMovement[i] > 0)
            *converged = false;
    }
}



__global__ void elkanMoveCenter(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) { 
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                center[i * dim + d] = z;
            }
        }
        
        centerMovement[i] = sqrt(centerMovement[i]);
        if (centerMovement[i] > 0)
            *converged = false;
    }
}

__global__ void elkanFunHamPRINT(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        //&& upper[i] > lower[i]
        if (upper[i] > s[closest2[i]]) {
            //if (upper[i] > lower[i]) {
            //    //printf("ERROR UPPER>LOWER");
            //    return;
            //}
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(innerProdp2c(data, center, i, j, dim));
                if (i == 1) {
                    printf("Distance: %f, to center: %i\n", curDistance, j);
                }
                if (curDistance < closestDistance) {
                    if (i == 1) {
                        printf("NEW LOWEST\n");
                        printf("VORHER: clos: %f, secClos: %f\n", closestDistance, secondClosestDist);
                    }
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[i] = j;
                    if (i == 1) {
                        printf("NACHHER: clos: %f, secClos: %f\n", closestDistance, secondClosestDist);
                    }
                }
                if (closestDistance > secondClosestDist)
                    printf("ERROR closestDistance> secondClosestDist");
            }
            upper[i] = closestDistance;
            lower[i] = secondClosestDist;

            if (i == 1) {
                printf("closest: %f to %i\n", closestDistance, closest2[i]);
                printf("secondclosest: %f\n\n", secondClosestDist);
            }
            if (upper[i] > lower[i])
                printf("ERROR UPPER>LOWER\n");
        }
    }
}

__global__ void elkanFunHam(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        unsigned long long int c;
        if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {    
                double curDistance = sqrt(dist22(data, center, i, j, dim));
#if DISTANCES
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
    }
}

__global__ void elkanFunHamSharedDaFuck(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 3 * 32;

    //__shared__ int counter;
    //__shared__ int calculate[blockSize];
    int calc = -1;
    //counter = 0;
    //calculate[threadIdx.x] = -1;
    __syncthreads();

    //__shared__ double sharedData[blockSize * dimension];

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
            //index = atomicAdd(&counter, 1);
            //for (int j = 0; j < dim; j++) {
            //    sharedData[index * dim + j] = data[i * dim + j];                
            //}
            //calculate[threadIdx.x] = i;
            calc = i;
        }

    }
    __syncthreads();

    if (i < n) {

        if (calc >= 0) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(innerProdp2c(data, center, calc, j, dim));
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[calc] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[calc] = closestDistance;
            lower[calc] = secondClosestDist;
        }
    }
}

__global__ void elkanFunHamShared(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 3 * 32;

    __shared__ int counter;
    __shared__ int calculate[blockSize];

    counter = 0;
    calculate[threadIdx.x] = -1;
    __syncthreads();

    //__shared__ double sharedData[blockSize * dimension];

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
            index = atomicAdd(&counter, 1);
            //for (int j = 0; j < dim; j++) {
            //    sharedData[index * dim + j] = data[i * dim + j];                
            //}
            calculate[index] = i;
        }

    }
    __syncthreads();

    //calculate[threadIdx.x]--;

    if (i < n) {
        
        if (calculate[threadIdx.x]>=0) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist22(data, center, calculate[threadIdx.x], j, dim));
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[calculate[threadIdx.x]] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[calculate[threadIdx.x]] = closestDistance;
            lower[calculate[threadIdx.x]] = secondClosestDist;
        }
    }
}

//__global__ void elkanFunHamSharedK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
//    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int point = threadIdx.x / k;
//    int cluster = threadIdx.x % k;
//    const int blockSize = 3 * 32;
//
//    __shared__ int counter;
//    __shared__ int calculate[blockSize];
//
//    counter = 0;
//    calculate[point] = -1;
//    __syncthreads();
//
//    //__shared__ double sharedData[blockSize * dimension];
//
//    int index;
//
//    if (i < n && cluster==0) {
//        closest2[i] = assignment[i];
//        if (upper[i] >= s[closest2[i]] && upper[i] >= lower[i]) {
//            index = atomicAdd(&counter, 1);
//            //for (int j = 0; j < dim; j++) {
//            //    sharedData[index * dim + j] = data[i * dim + j];                
//            //}
//            calculate[index] = i;
//        }
//
//    }
//    __syncthreads();
//
//    //calculate[threadIdx.x]--;
//
//    if (i < n) {
//        if (calculate[point] >= 0) {
//            double closestDistance = INFINITY;
//            double secondClosestDist = INFINITY;
//
//                lower[] = sqrt(dist22(data, center, calculate[point], cluster, dim));
//                if (curDistance < closestDistance) {
//                    secondClosestDist = closestDistance;
//                    closestDistance = curDistance;
//                    closest2[calculate[threadIdx.x]] = cluster;
//                }
//                else if (curDistance < secondClosestDist) {
//                    secondClosestDist = curDistance;
//                }
//            
//            upper[calculate[threadIdx.x]] = closestDistance;
//            lower[calculate[threadIdx.x]] = secondClosestDist;
//        }
//    }
//}


__global__ void elkanFunHamFewerRules(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= lower[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {                           
                double curDistance = sqrt(innerProdp2c(data, center, i, j, dim));
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
    }
}

__global__ void elkanFunNoMoveKcombine(double* lower, double* upper,int k, int n, unsigned short* closest2, bool* calculated, double* dist) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < k; j++) {
            if (calculated[i+k+j]) {
                lower[i * k + j] = dist[i * k + j];
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }
    
}

__global__ void elkanFunNoMoveK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset, bool* calculated, double* dist) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (i < n) {
        closest2[c1] = assignment[c1];
        double localUpper = upper[c1];
        //upper[i] > s[closest2[i]] && upper[i] >= lower[i * k + j] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i];
        calculated[c1 * k + j] = localUpper > s[closest2[c1]] && localUpper >= lower[c1 * k + j];

       /* if (localUpper > s[closest2[c1]]) {
            if (j == closest2[c1]) { return; }
            if (localUpper <= lower[c1 * k + j]) { return; }
            if (localUpper <= centerCenterDistDiv2[closest2[c1] * k + j]) { return; }*/
        if (calculated[c1 * k + j]){
            dist[c1 * k + j] = sqrt(dist22(data, center, c1, j, dim));           
        }
    }
}

__global__ void elkanFunNoMoveKKcombine(double* lower, double* upper, int k, int n, unsigned short* closest2, bool* calculated) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < k; j++) {
            if (lower[i * k + j] < upper[i]) {
                closest2[i] = j;
                upper[i] = lower[i * k + j];
            }
            
        }
    }
 }

__global__ void elkanFunNoMoveKK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (c1 < n) {
        closest2[c1] = assignment[c1];
        //double localUpper = ;

        if (upper[c1] > s[closest2[c1]]) {
            //if (j == closest2[c1]) { return; }
            if (upper[c1] <= lower[c1 * k + j]) { return; }
            if (upper[c1] <= centerCenterDistDiv2[closest2[c1] * k + j]) { return; }

            lower[c1 * k + j] = sqrt(dist22(data, center, c1, j, dim));
        }
    }
}

__global__ void elkanFunNoMoveShared(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 4 * 32;

    __shared__ int counter;
    __shared__ int calculate[blockSize];

    counter = 0;
    calculate[threadIdx.x] = -1;
    __syncthreads();

    //__shared__ double sharedData[blockSize * dimension];

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= s[closest2[i]]) {
            index = atomicAdd(&counter, 1);
            calculate[index] = i;
        }

    }
    __syncthreads();

    if (i < n) {
        closest2[calculate[threadIdx.x]] = assignment[calculate[threadIdx.x]];
        bool r = true;

        if (calculate[threadIdx.x]>=0) {
                for (int j = 0; j < k; ++j) {
                    if (j == closest2[calculate[threadIdx.x]]) { continue; }
                    if (upper[calculate[threadIdx.x]] <= lower[calculate[threadIdx.x] * k + j]) { continue; }
                    if (upper[calculate[threadIdx.x]] <= centerCenterDistDiv2[closest2[calculate[threadIdx.x]] * k + j]) { continue; }
                    if (r) {
                        upper[calculate[threadIdx.x]] = sqrt(dist22(data, center, calculate[threadIdx.x], closest2[calculate[threadIdx.x]], dim));

                        lower[calculate[threadIdx.x] * k + closest2[calculate[threadIdx.x]]] = upper[calculate[threadIdx.x]];
                        r = false;
                        if ((upper[calculate[threadIdx.x]] <= lower[calculate[threadIdx.x] * k + j]) || (upper[calculate[threadIdx.x]] <= centerCenterDistDiv2[closest2[calculate[threadIdx.x]] * k + j])) {
                            continue;
                        }
                    }
                    lower[calculate[threadIdx.x] * k + j] = sqrt(dist22(data, center, calculate[threadIdx.x], j, dim));
                    if (lower[calculate[threadIdx.x] * k + j] < upper[calculate[threadIdx.x]]) {
                        closest2[calculate[threadIdx.x]] = j;
                        upper[calculate[threadIdx.x]] = lower[calculate[threadIdx.x] * k + j];
                    }
                }                             
        }
    }
}

////without shared
//int i = blockIdx.x * blockDim.x + threadIdx.x;
//if (i < n) {
//    if (upper[i] > s[closest2[i]]){
//        ...
//
////with shared
//int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
//const int blockSize = 5 * 32;
//
//__shared__ int counter;
//__shared__ int calculate[blockSize];
//
//counter = 0;
//calculate[threadIdx.x] = -1;
//__syncthreads();
//
//int index;
//if (i < n) {
//    closest2[i] = assignment[i];
//    if (upper[i] >= s[closest2[i]]) {
//        index = atomicAdd(&counter, 1);
//        calculate[index] = i;
//    }
//
//}
//__syncthreads();        
//if (i < n) {
//    //use calculate[threadIdx.x] instead of i after this
//    closest2[calculate[threadIdx.x]] = assignment[calculate[threadIdx.x]];
//    bool r = true;
//    if (calculate[threadIdx.x] >= 0) {
//        ...


__global__ void elkanFunNoMove(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset, unsigned long long int* countDistances) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
   /* if (i == 0)
        printf("Closestt start\n");*/
    if (i < n) {
        closest2[i] = assignment[i];
        /*if (i == 0)
            printf("Closestt bevor i=0:  %i\n", closest2[i]);*/
        bool r = true;
        double localUpper = upper[i];
        unsigned long long int c;
        if (localUpper > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (localUpper <= lower[i * k + j]) { continue; }
                if (localUpper <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    //localUpper = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    localUpper = sqrt(dist22(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    /* for (int j = 0; j < dim; j++) {
                         lastExactCentroid[i * dim + j] = center[assignment[i] * dim + j];
                     }*/
                    lower[i * k + closest2[i]] = localUpper;
                    r = false;
                     if ((localUpper <= lower[i * k + j]) || (localUpper <= centerCenterDistDiv2[closest2[i] * k + j])) {
                         continue;
                     }
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                lower[i * k + j] = sqrt(dist22(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
               /* if (i == 0)
                    printf("Closest i=0:  %i\n", closest2[i]);*/
                if (lower[i * k + j] < localUpper){
                    closest2[i] = j;
                   /* if (i == 0)
                        printf("Closestt i=0:  %i\n", closest2[i]);*/
                    localUpper = lower[i * k + j];
                    /*   for (int j = 0; j < dim; j++) {
                           lastExactCentroid[i * dim + j] = center[assignment[i] * dim + j];
                       }*/
                }
            }
            upper[i] = localUpper;
        }
    }
}


__global__ void elkanParallelCheck(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int offset, bool* check) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int j = i % k;

    if (c1 < n) {
        closest2[c1] = assignment[c1];
        check[c1*k+j] = true;
        

        if (upper[c1] > s[closest2[c1]]) {
            if (j == closest2[c1]) { return; }
            if (upper[c1] <= lower[c1 * k + j]) { return; }
            check[c1 * k + j] = false;
        }
    }
}

__global__ void elkanFunNoMoveAfterCheck(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int offset, bool* check) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];

        for (int j = 0; j < k; ++j) {
            if (check[i * k + j]) 
                continue;

            upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
            //lower[i * k + closest2[i]] = upper[i];
            lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));

            if (lower[i * k + j] < upper[i]) {
                closest2[i] = j;
                upper[i] = lower[i * k + j];
            }
        }           
        
    }
}

__global__ void elkanFunNoMoveFewer(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int offset) {

    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
               
                //upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));

                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }
}

__global__ void elkanFunNoMoveWTF(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, int numlower, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        //unsigned short closest = assignment[i];

        closest2[i] = assignment[i];
        bool r = true;

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }
}

__global__ void elkanFunFBTest(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, int numlower, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        //unsigned short closest = assignment[i];

        closest2[i] = assignment[i];
        bool r = true;

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }
    }
}

__global__ void elkanFunMOHamTTLoop(double* upper,
    int k, int n, unsigned short* closest2, bool* calculated, double* distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double closestDistance = INFINITY;
        for (int j = 0; j < k; ++j) {            
            if (calculated[i * k + j]) {
                if (distances[i * k + j] < closestDistance) {
                    closestDistance = distances[i * k + j];
                    closest2[i] = j;
                }
            }
                      
        }
        upper[i] = closestDistance;
    }
}

__global__ void elkanFunFBHamTTLoop(double* lower, double* upper,
    int k, int n, unsigned short* closest2, bool* calculated, double* distances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //closest2[i] = assignment[i];
        //double curDistance;
        for (int j = 0; j < k; ++j) {
            if (calculated[i * k + j]) {
                //lower[i * k + j] = distances[i * k + j];
                //lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }      
    }
}

__global__ void elkanFunFBHamTT(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, bool* calculated, double* distances) {
   /* __global__ void elkanFunFBHamTT(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
        double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, bool* calculated, double* distances) {*/
//__global__ void elkanFunFBHamTT(double* lower, double* upper, int k, int n, unsigned short* closest2, bool* calculated, double* distances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //closest2[i] = assignment[i];
        if (calculated[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = distances[i * k + j];
               // double curDistance2 = sqrt(innerProdp2c(data, center, i, j, dim));
               /* if (curDistance != curDistance2) {
                    printf("UNGLEICH %f, %f\n", curDistance, curDistance2);
                }*/
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
    }
}

__global__ void elkanFunFBHamK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, double*distances, bool*calculated) {
   /* __global__ void elkanFunFBHamK(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
        double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, double* distances, bool* calculated) {*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        calculated[c1] = upper[c1] > s[assignment[c1]] && upper[c1] >= lower[c1]; //&& upper[c1] >= maxoldcenter2newcenterDis[assignment[c1]] - ub_old[c1];    
        if (calculated[c1])
            distances[i] = sqrt(dist22(data, center, c1, c2, dim));
    }
}

__global__ void elkanFunFBHam(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        unsigned long long int c;
        if (upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i]) {
        //if (upper[i] > s[closest2[i]]) {
        //if (upper[i] > s[closest2[i]] && upper[i] >= lower[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist22(data, center, i, j, dim));
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
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
    }    
}

__global__ void elkanFunFBHamShared(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* maxoldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 20 * 32;

    __shared__ int counter;
    __shared__ int calculate[blockSize];

    counter = 0;
    calculate[threadIdx.x] = -1;
    __syncthreads();

    //__shared__ double sharedData[blockSize * dimension];

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i]) {
            index = atomicAdd(&counter, 1);
            //for (int j = 0; j < dim; j++) {
            //    sharedData[index * dim + j] = data[i * dim + j];                
            //}
            calculate[index] = i;
        }

    }
    __syncthreads();

    //calculate[threadIdx.x]--;

    if (i < n) {
        if (calculate[threadIdx.x] >= 0) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = sqrt(dist22(data, center, calculate[threadIdx.x], j, dim));
                
                if (curDistance < closestDistance) {
                    secondClosestDist = closestDistance;
                    closestDistance = curDistance;
                    closest2[calculate[threadIdx.x]] = j;
                }
                else if (curDistance < secondClosestDist) {
                    secondClosestDist = curDistance;
                }
            }
            upper[calculate[threadIdx.x]] = closestDistance;
            lower[calculate[threadIdx.x]] = secondClosestDist;
        }
    }
}

__global__ void elkanFunLloyd(double* data, double* center, unsigned short* assignment, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        double closestDistance = INFINITY;
        for (int j = 0; j < k; ++j) {            
            double curDistance = sqrt(dist22(data, center, i, j, dim));
           // atomicAdd(countDistances, 1);
            if (curDistance < closestDistance) {
                closestDistance = curDistance;
                closest2[i] = j;
            }
        }        
    }
}

__global__ void calculateFilterLoopMO(unsigned short* assignment, double* distances, double* upper,
    double* s, double* oldcenter2newcenterDis, double* ub_old, bool* calculated, int n, int k, unsigned short* closest2, double* centerCenterDistDiv2, double* oldcenterCenterDistDiv2, double* centerMovement) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        for (int j = 0; j < k; j++) {
            calculated[i * k + j] = upper[i] > s[closest2[i]] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i] && 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j];
        }
    }
}

__global__ void calculateFilterLoop(unsigned short* assignment, double* lower, double* upper,
    double* s, double* oldcenter2newcenterDis, double* ub_old, bool* calculated, int n, int k, unsigned short* closest2, double* centerCenterDistDiv2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i]; 
        for (int j = 0; j < k; j++){
            //calculated[i*k+j] = upper[i] > s[closest2[i]] && upper[i] >= lower[i * k + j] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i] && upper[i] >= centerCenterDistDiv2[closest2[i] * k + j];
            
            //correct
            calculated[i * k + j] = upper[i] > s[closest2[i]] && upper[i] >= lower[i * k + j] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i];

            //calculated[i * k + j] = upper[i] > s[closest2[i]] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i];
        }
    }
}

__global__ void calculateFilterLoopE(unsigned short* assignment, double* lower, double* upper,
    double* s, bool* calculated, int n, int k, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        for (int j = 0; j < k; j++) {
            //calculated[i*k+j] = upper[i] > s[closest2[i]] && upper[i] >= lower[i * k + j] && upper[i] >= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i] && upper[i] >= centerCenterDistDiv2[closest2[i] * k + j];
            calculated[i * k + j] = upper[i] > s[closest2[i]] && upper[i] >= lower[i * k + j];
        }
    }
}

__global__ void calculateFilterH(unsigned short* assignment, double* lower, double* upper,
    double* s, bool* calculated, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= lower[i];
    }
}

__global__ void calculateFilter(unsigned short* assignment, double* lower, double* upper,
    double* s, double* maxoldcenter2newcenterDis, double* ub_old, bool* calculated, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i];
    }
}

__global__ void calculateFilter2(unsigned short* assignment, double* lower, double* upper,
    double* s, double* maxoldcenter2newcenterDis, double* ub_old, bool* calculated, int n, unsigned short* closest2, double* data, double* center, int dim) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i];
        if (calculated[i]) {
            upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
            calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i];
        }
    }
}

__global__ void calculateFilterMore(unsigned short* assignment, double* lower, double* upper,
    double* s, double* maxoldcenter2newcenterDis, double* ub_old, bool* calculated, int n, unsigned short* closest2, double* centerCenterDistDiv2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];
        //calculated[i] = upper[i] > s[closest2[i]] && upper[i] >= lower[i] && upper[i] >= maxoldcenter2newcenterDis[assignment[i]] - ub_old[i] && upper[i] >= centerCenterDistDiv2[closest2[i] * k + j];
    }
}

__global__ void elkanFunFBHam2(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /*int c1 = i / k;
    int c2 = i % k;*/

    if (i < n) {
        if (calculated[i]) {
            for (int j = 0; j < k; ++j) {
                distances[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
            }
        }
    }
}

__global__ void elkanFunFBHam2TT(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        if (calculated[c1]) {
            //distances[i] = sqrt(innerProdp2c(data, center, c1, c2, dim));
            distances[i] = sqrt(dist22(data, center, c1, c2, dim));
        }
    }
}

__global__ void elkanFunFBHam2TTDouble(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        if (calculated[c1]) {
            distances[c1 * k + c2] = sqrt(innerProdp2c(data, center, c1, c2, dim));
        }
    }
}


__global__ void elkanFunMOHam2TTLoop(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n, unsigned short* assignment,
        double* upper, double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* oldcenterCenterDistDiv2, double* ub_old, double* centerMovement, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        closest2[i] = assignment[i];
        calculated[c1 * k + c2] = upper[c1] > s[closest2[c1]] && upper[c1] >= distances[c1 * k + c2] && upper[c1] >= oldcenter2newcenterDis[assignment[c1] * k + c2] - ub_old[c1];
        if (calculated[c1 * k + c2]) {
            distances[c1 * k + c2] = sqrt(dist22(data, center, c1, c2, dim));
            //lower[c1*k+c2] = sqrt(dist22(data, center, c1, c2, dim));
        }
    }
}

__global__ void elkanFunFBHam2TTLoop(double* data, double* center, double* distances, bool* calculated, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c1 = i / k;
    int c2 = i % k;

    if (c1 < n) {
        if (calculated[c1 * k + c2]) {
            distances[c1 * k + c2] = sqrt(dist22(data, center, c1, c2, dim));
            //lower[c1*k+c2] = sqrt(dist22(data, center, c1, c2, dim));
        }
    }
}

__global__ void elkanFunFBHamBounds(double* data, double* lower, double* upper,
    double* distances, bool* calculated, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (calculated[i]) {
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;

            for (int j = 0; j < k; ++j) {
                double curDistance = distances[i * k + j];
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
    }
}

__global__ void elkanFunFB(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2, unsigned long long int* countDistances) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        closest2[i] = assignment[i];
        bool r = true;
        unsigned long long int c;
        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(dist22(data, center, i, closest2[i], dim));
#if DISTANCES
                    c = atomicAdd(countDistances, 1);
                    if (c == 18446744073709551615) {
                        printf("OVERFLOW");
                    }
#endif
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) || upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(dist22(data, center, i, j, dim));
#if DISTANCES
                c = atomicAdd(countDistances, 1);
                if (c == 18446744073709551615) {
                    printf("OVERFLOW");
                }
#endif
                if (lower[i * k + j] < upper[i]) {
                    closest2[i] = j;
                    upper[i] = lower[i * k + j];
                }
            }
        }      
    }
}

__global__ void elkanFunFBShared(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 4 * 32;

    __shared__ int counter;
    __shared__ int calculate[blockSize];

    counter = 0;
    calculate[threadIdx.x] = -1;
    __syncthreads();

    int index;

    if (i < n) {
        closest2[i] = assignment[i];
        if (upper[i] >= s[closest2[i]]) {
            index = atomicAdd(&counter, 1);
            calculate[index] = i;
        }

    }
    __syncthreads();

    if (i < n) {
        closest2[calculate[threadIdx.x]] = assignment[calculate[threadIdx.x]];
        bool r = true;

        if (calculate[threadIdx.x]>=0) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[calculate[threadIdx.x]]) { continue; }
                if (upper[calculate[threadIdx.x]] <= lower[calculate[threadIdx.x] * k + j]) { continue; }
                if (upper[calculate[threadIdx.x]] <= oldcenter2newcenterDis[assignment[calculate[threadIdx.x]] * k + j] - ub_old[calculate[threadIdx.x]]) { continue; }
                if (upper[calculate[threadIdx.x]] <= centerCenterDistDiv2[closest2[calculate[threadIdx.x]] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[calculate[threadIdx.x]] = sqrt(dist22(data, center, calculate[threadIdx.x], closest2[calculate[threadIdx.x]], dim));
                    lower[calculate[threadIdx.x] * k + closest2[calculate[threadIdx.x]]] = upper[calculate[threadIdx.x]];
                    r = false;
                }

                // ELKAN 3(b)
                lower[calculate[threadIdx.x] * k + j] = sqrt(dist22(data, center, calculate[threadIdx.x], j, dim));
                if (lower[calculate[threadIdx.x] * k + j] < upper[calculate[threadIdx.x]]) {
                    closest2[calculate[threadIdx.x]] = j;
                    upper[calculate[threadIdx.x]] = lower[calculate[threadIdx.x] * k + j];
                }
            }
        }
    }
}
