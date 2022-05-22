#pragma once

#include "Dataset.h"
#include "yyheader.h"

int generateCentWithDataSame(CentInfo* centInfo,
    DTYPE* centData,
    DTYPE* copyData,
    const int numCent,
    const int numCopy,
    const int numDim);

int generateCentWithData(CentInfo* centInfo,
    DTYPE* centData,
    DTYPE* copyData,
    const int numCent,
    const int numCopy,
    const int numDim);

double startSuperOnGPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numDim,
    const int maxIter,
    const int numGPUU,
    unsigned int* ranIter);

void initPoints(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* pointLwrs,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int numThread);

void updateCentroids(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int numThread);

void pointCalcsSimpleCPU(PointInfo* pointInfoPtr,
    CentInfo* centInfo,
    DTYPE* pointDataPtr,
    DTYPE* pointLwrPtr,
    DTYPE* centData,
    DTYPE* maxDriftArr,
    unsigned int* groupArr,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim);

double startSimpleOnCPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int numThread,
    const int maxIter,
    unsigned int* ranIter);

double startFullOnGPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int maxIter,
    const int numGPU,
    unsigned int* ranIter);

double startSimpleOnGPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int maxIter,
    const int numGPU,
    unsigned int* ranIter);

DTYPE* storeDataOnGPU(DTYPE* data,
    const int numVec,
    const int numFeat);

PointInfo* storePointInfoOnGPU(PointInfo* pointInfo,
    const int numPnt);

CentInfo* storeCentInfoOnGPU(CentInfo* centInfo,
    const int numCent);

void warmupGPU(const int numGPU);

DTYPE calcDisCPU(DTYPE* vec1,
    DTYPE* vec2,
    const int numDim);

unsigned int checkConverge(PointInfo* pointInfo,
    const int numPnt);

double warumGehtNichts();

double startLloydOnCPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numDim,
    const int numThread,
    const int maxIter,
    unsigned int* ranIter);

double startLloydOnGPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numDim,
    const int maxIter,
    const int numGPU,
    unsigned int* ranIter);

/*double startSimpleOnGPU(PointInfo* pointInfo,
    CentInfo* centInfo,
    DTYPE* pointData,
    DTYPE* centData,
    const int numPnt,
    const int numCent,
    const int numGrp,
    const int numDim,
    const int maxIter,
    const int numGPU,
    unsigned int* ranIter,
    unsigned long long int* countPtr);*/