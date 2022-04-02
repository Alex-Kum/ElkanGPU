//#pragma once
//
//#include "Dataset.h"
//#include "yyheader.h"
//
//int generateCentWithData(CentInfo* centInfo,
//    DTYPE* centData,
//    DTYPE* copyData,
//    const int numCent,
//    const int numCopy,
//    const int numDim);
//
//double startFullOnGPU(PointInfo* pointInfo,
//    CentInfo* centInfo,
//    DTYPE* pointData,
//    DTYPE* centData,
//    const int numPnt,
//    const int numCent,
//    const int numGrp,
//    const int numDim,
//    const int maxIter,
//    const int numGPU,
//    unsigned int* ranIter);
//
//double startSimpleOnGPU(PointInfo* pointInfo,
//    CentInfo* centInfo,
//    DTYPE* pointData,
//    DTYPE* centData,
//    const int numPnt,
//    const int numCent,
//    const int numGrp,
//    const int numDim,
//    const int maxIter,
//    const int numGPU,
//    unsigned int* ranIter);
//
//DTYPE* storeDataOnGPU(DTYPE* data,
//    const int numVec,
//    const int numFeat);
//
//PointInfo* storePointInfoOnGPU(PointInfo* pointInfo,
//    const int numPnt);
//
//CentInfo* storeCentInfoOnGPU(CentInfo* centInfo,
//    const int numCent);
//
//void warmupGPU(const int numGPU);
//
//DTYPE calcDisCPU(DTYPE* vec1,
//    DTYPE* vec2,
//    const int numDim);
//
//unsigned int checkConverge(PointInfo* pointInfo,
//    const int numPnt);
//
//double warumGehtNichts();
//
//double startLloydOnCPU(PointInfo* pointInfo,
//    CentInfo* centInfo,
//    DTYPE* pointData,
//    DTYPE* centData,
//    const int numPnt,
//    const int numCent,
//    const int numDim,
//    const int numThread,
//    const int maxIter,
//    unsigned int* ranIter);
//
//double startLloydOnGPU(PointInfo* pointInfo,
//    CentInfo* centInfo,
//    DTYPE* pointData,
//    DTYPE* centData,
//    const int numPnt,
//    const int numCent,
//    const int numDim,
//    const int maxIter,
//    const int numGPU,
//    unsigned int* ranIter);
//
///*double startSimpleOnGPU(PointInfo* pointInfo,
//    CentInfo* centInfo,
//    DTYPE* pointData,
//    DTYPE* centData,
//    const int numPnt,
//    const int numCent,
//    const int numGrp,
//    const int numDim,
//    const int maxIter,
//    const int numGPU,
//    unsigned int* ranIter,
//    unsigned long long int* countPtr);*/