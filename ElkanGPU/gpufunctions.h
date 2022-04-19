#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "yyheader.h"
#include <stdio.h>

#define DTYPE double
#define BLOCKSIZE 256

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

    unsigned int btid = threadIdx.x;
    unsigned int index;


    pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

    extern __shared__ unsigned int groupLclArr[];

    // update points upper bound
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


__global__ void innerProd(double* centerCenterDist, double* s, const double* data, int dim, int n) {
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
        oldcenter2newcenterDis[c1 * k + c2] = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            oldcenter2newcenterDis[c1 * k + c2] += (oldcenters[c1 * d + dim] - center[c2 * d + dim])  * (oldcenters[c1 * d + dim] - center[c2 * d + dim]);
        }
        oldcenter2newcenterDis[c1 * k + c2] = sqrt(oldcenter2newcenterDis[c1 * k + c2]);
    }
    
}

__global__ void updateBoundHam(double* data, double* lower, double* upper, double* centerMovement, unsigned short* assignment, int numLowerBounds, int dim, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {        
        double maxMovement = 0;
       /* if (i == 24) {
            printf("upper: %f\n", upper[i]);
            printf("%f + %f = ", upper[i], centerMovement[assignment[i]]);
        }*/
        upper[i] += centerMovement[assignment[i]];
       /* if (i == 24) {
            printf("%f (newupper)\n", upper[i]);
        }*/

        for (int j = 0; j < k; ++j) {
            /*if (i == 24) {
                printf("centerMovement: %f of center %i\n", centerMovement[j], j);
            }*/
            if (j == assignment[i])
                continue;
            if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }
        /*if (i == 24) {
            printf("MaxMovement: %f\n", maxMovement);
            printf("%f - %f = ", lower[i], maxMovement);
        }*/
        lower[i] -= maxMovement;
        /*if (i == 24) {
            printf("%f (newlower)\n", lower[i]);
        }*/
        //--------------------------------------------------------------------------------------------------
       /* double maxMovement = 0;
        for (int j = 0; j < k; ++j) {
            if (j == assignment[i])
                upper[i] += centerMovement[j];
            else if (centerMovement[j] > maxMovement)
                maxMovement = centerMovement[j];
        }        
        lower[i] -= maxMovement;*/
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

__global__ void updateBoundMO(double* upper, double* ub_old, double* centerMovement, unsigned short* assignment, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ub_old[i] = upper[i];
        upper[i] += centerMovement[assignment[i]];
    }
}



__global__ void elkanFunMO(double* data, double* center, unsigned short* assignment, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* oldcenterCenterDistDiv2, double* ub_old, double* centerMovement, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //unsigned short closest = assignment[i];
        closest2[i] = assignment[i];
        bool r = true;

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= 2.0 * (oldcenterCenterDistDiv2[assignment[i] * k + j]) - ub_old[i] - centerMovement[j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }  //upper[i] <= lower[i * k + j] ||
                if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    r = false;
                }

                // ELKAN 3(b)
                //lower[i * k + j] = sqrt(innerProdp2c(data, center, i, j, dim));
                double inner = sqrt(innerProdp2c(data, center, i, j, dim));
                if (inner < upper[i]) {
                    closest2[i] = j;
                    upper[i] = inner;
                }
            }
        }
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


__global__ void elkanMoveCenter(double* centerMovement, int* clusterSize, double* center, double* sumNewCenters, bool* converged, int k, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        centerMovement[i] = 0.0;
        int totalClusterSize = clusterSize[i];

        if (totalClusterSize > 0) { 
           /* if (1==i)
                printf("CENTERPOS 1:\n");*/
            for (int d = 0; d < dim; ++d) {
                double z = 0.0;
                z += sumNewCenters[i * dim + d];
                z /= totalClusterSize;
                centerMovement[i] += (z - center[i * dim + d]) * (z - center[i * dim + d]);
                center[i * dim + d] = z;
                /*if (1 == i)
                printf("%f ", center[i * dim + d]);*/
            }
           /* if (1 == i)
            printf("\n");*/
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
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        bool notIf = false;

        closest2[i] = assignment[i];
        //&& upper[i] > lower[i]
        //upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
       /* if (i == 2115) {
            printf("upper: %f assigned to %u\n", upper[i], closest2[i]);
            printf("lower: %f\n", lower[i]);
        }*/
        if (upper[i] > s[closest2[i]] && upper[i] >= lower[i]) {
            //if (upper[i] <= lower[i]) {
            //    //printf("SONST NICHT IM IF %i\n", i);
            //    notIf = true;
            //}
            double closestDistance = INFINITY;
            double secondClosestDist = INFINITY;
            //if (i == 24 && upper[i] <= lower[i]) {
          /*  if (i == 2115) {
                printf("IN IF upper: %f assigned to %u\n", upper[i], closest2[i]);
                printf("IN IF lower: %f\n", lower[i]);
            }*/
            for (int j = 0; j < k; ++j) {                
                double curDistance = sqrt(innerProdp2c(data, center, i, j, dim));
                //if (i == 24 && upper[i] <= lower[i]) {
              /*  if (i == 2115) {
                    printf("Distance: %f, to center: %i\n", curDistance, j);
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

            //if (i == 24 && upper[i] <= lower[i]) {
           /* if (closest2[i] != assignment[i] && notIf)
                printf("#####%i########\n", i);*/
           /* if (i == 2115) {
                printf("AFTER: upper: %f assigned to %u\n", closestDistance, closest2[i]);
                printf("AFTER: lower: %f\n\n", secondClosestDist);
            }*/

            upper[i] = closestDistance;
            lower[i] = secondClosestDist;

            
        }
    }
}


__global__ void elkanFunNoMove(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int offset) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
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
                    /* for (int j = 0; j < dim; j++) {
                         lastExactCentroid[i * dim + j] = center[assignment[i] * dim + j];
                     }*/
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
                    /*   for (int j = 0; j < dim; j++) {
                           lastExactCentroid[i * dim + j] = center[assignment[i] * dim + j];
                       }*/
                }
            }
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
    double* s, double* centerCenterDistDiv2, int k, int dim, int n, unsigned short* closest2, int* clusterSize, double* sumNewCenters, int offset) {

    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        closest2[i] = assignment[i];

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
               
                upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
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

__global__ void elkanFunFB(double* data, double* center, unsigned short* assignment, double* lower, double* upper,
    double* s, double* centerCenterDistDiv2, double* oldcenter2newcenterDis, double* ub_old, int k, int dim, int n, unsigned short* closest2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //unsigned short closest = assignment[i];
        closest2[i] = assignment[i];
        bool r = true;

        if (upper[i] > s[closest2[i]]) {
            for (int j = 0; j < k; ++j) {
                if (j == closest2[i]) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) { continue; }
                //if (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    //upper[i] = sqrt(pointCenterDist2(i, closest));
                    upper[i] = sqrt(innerProdp2c(data, center, i, closest2[i], dim));
                    lower[i * k + closest2[i]] = upper[i];
                    r = false;
                    //if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j]) || upper[i] <= oldcenter2newcenterDis[assignment[i] * k + j] - ub_old[i]) {
                    //    //if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest2[i] * k + j])) {
                    //    continue;
                    //}
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

        /*int furthestMovingCenter = 0;
        for (int j = 0; j < k; ++j) {
            centerMovement[j] = 0.0;
            int totalClusterSize = 0;
            double old = 0;
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
                    oldcenters[j * d + dim] = center[j * dim + d];
                    center[j * dim + d] = z;
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
                    for (int d = 0; d < dim; ++d) {
                        oldcenter2newcenterDis[c1 * k + c2] += (oldcenters[c1 * dim + d] - center[c2 * dim + d]) * (oldcenters[c1 * dim + d] - center[c2 * dim + d]);
                    }
                    oldcenter2newcenterDis[c1 * k + c2] = sqrt(oldcenter2newcenterDis[c1 * k + c2]);
                }
        }

        *converged = (0.0 == centerMovement[furthestMovingCenter]);
        if (!(*converged)) {
            ub_old[i] = upper[i];
            upper[i] += centerMovement[assignment[i]];
            for (int j = 0; j < k; ++j) {
                lower[i * numLowerBounds + j] -= centerMovement[j];
            }
        }*/

    }
}
