//https://github.com/ctaylor389/k_means_yinyang_gpu


//#pragma once
//
//#include "Dataset.h"
//#include "yyheader.h"
//
//
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