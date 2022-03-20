//#include "yy_kmean.h"
//#include "gpufunctions.h"
//#include "omp.h"
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//    if (code != cudaSuccess)
//    {
//        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//        if (abort) exit(code);
//    }
//}
//
//int generateCentWithData(CentInfo* centInfo,
//    DTYPE* centData,
//    DTYPE* copyData,
//    const int numCent,
//    const int numCopy,
//    const int numDim)
//{
//    srand(90);
//    int i;
//    int j;
//    int randomMax = numCopy / numCent;
//    for (i = 0; i < numCent; i++)
//    {
//        for (j = 0; j < numDim; j++)
//        {
//            centData[(i * numDim) + j] =
//                copyData[((i * randomMax) +
//                    (rand() % randomMax)) * numDim + j];
//        }
//        centInfo[i].groupNum = -1;
//        centInfo[i].drift = 0.0;
//        centInfo[i].count = 0;
//    }
//    return 0;
//}
//
//DTYPE calcDisCPU(DTYPE* vec1,
//    DTYPE* vec2,
//    const int numDim)
//{
//    unsigned int index;
//    DTYPE total = 0.0;
//    DTYPE square;
//
//    for (index = 0; index < numDim; index++)
//    {
//        square = (vec1[index] - vec2[index]);
//        total += square * square;
//    }
//
//    return sqrt(total);
//}
//
//int groupCent(CentInfo* centInfo,
//    DTYPE* centData,
//    const int numCent,
//    const int numGrp,
//    const int numDim)
//{
//    CentInfo* overInfo = (CentInfo*)malloc(sizeof(CentInfo) * numGrp);
//    DTYPE* overData = (DTYPE*)malloc(sizeof(DTYPE) * numGrp * numDim);
//    generateCentWithData(overInfo, overData, centData, numGrp, numCent, numDim);
//
//    unsigned int iterIndex, centIndex, grpIndex, dimIndex, assignment;
//
//    const int numDimm = 10;
//    const int numGrpp = 20;
//    DTYPE currMin = INFINITY;
//    DTYPE currDistance = INFINITY;
//    DTYPE origVec[numGrpp][numDimm];
//
//    for (iterIndex = 0; iterIndex < 5; iterIndex++)
//    {
//        // assignment
//        for (centIndex = 0; centIndex < numCent; centIndex++)
//        {
//            for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//            {
//                currDistance = calcDisCPU(&centData[centIndex * numDim],
//                    &overData[grpIndex * numDim],
//                    numDim);
//                if (currDistance < currMin)
//                {
//                    centInfo[centIndex].groupNum = grpIndex;
//                    currMin = currDistance;
//                }
//            }
//            currMin = INFINITY;
//        }
//        // update over centroids
//        for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//        {
//            for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//            {
//                origVec[grpIndex][dimIndex] =
//                    overData[(grpIndex * numDim) + dimIndex];
//                overData[(grpIndex * numDim) + dimIndex] = 0.0;
//            }
//            overInfo[grpIndex].count = 0;
//        }
//
//        // update over centroids to be average of group
//        for (centIndex = 0; centIndex < numCent; centIndex++)
//        {
//            assignment = centInfo[centIndex].groupNum;
//            overInfo[assignment].count += 1;
//            for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//            {
//                overData[(assignment * numDim) + dimIndex] +=
//                    centData[(centIndex * numDim) + dimIndex];
//            }
//        }
//
//
//        for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//        {
//            if (overInfo[grpIndex].count > 0)
//            {
//                for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//                {
//                    overData[(grpIndex * numDim) + dimIndex] /=
//                        overInfo[grpIndex].count;
//                }
//            }
//            else
//            {
//                for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//                {
//                    overData[(grpIndex * numDim) + dimIndex] =
//                        origVec[grpIndex][dimIndex];
//                }
//            }
//        }
//    }
//    free(overData);
//    free(overInfo);
//    return 0;
//}
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
//    const int numGPUU,
//    unsigned int* ranIter)
//{
//    // start timer
//    double startTime, endTime;
//    startTime = omp_get_wtime();
//
//    // variable initialization
//    int gpuIter;
//    const int numGPU = 1;
//    int numPnts[numGPU];
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//        {
//            numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//        }
//
//        else
//        {
//            numPnts[gpuIter] = numPnt / numGPU;
//        }
//    }
//
//    unsigned int hostConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        hostConFlagArr[gpuIter] = 1;
//    }
//
//    unsigned int* hostConFlagPtrArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//    }
//
//    int grpLclSize = sizeof(unsigned int) * numGrp * BLOCKSIZE;
//
//    int index = 1;
//
//    unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//    // group centroids
//    groupCent(centInfo, centData, numCent, numGrp, numDim);
//
//    // create lower bound data on host
//    DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
//    for (int i = 0; i < numPnt * numGrp; i++)
//    {
//        pointLwrs[i] = INFINITY;
//    }
//
//    // store dataset on device
//    PointInfo* devPointInfo[numGPU];
//    DTYPE* devPointData[numGPU];
//    DTYPE* devPointLwrs[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        cudaSetDevice(gpuIter);
//
//        // alloc dataset to GPU
//        gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//        // copy input data to GPU
//        gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//            pointInfo + (gpuIter * numPnt / numGPU),
//            (numPnts[gpuIter]) * sizeof(PointInfo),
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//        gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//            pointData + ((gpuIter * numPnt / numGPU) * numDim),
//            sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
//            numGrp));
//
//        gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
//            pointLwrs + ((gpuIter * numPnt / numGPU) * numGrp),
//            sizeof(DTYPE) * numPnts[gpuIter] * numGrp,
//            cudaMemcpyHostToDevice));
//    }
//
//    // store centroids on device
//    CentInfo* devCentInfo[numGPU];
//    DTYPE* devCentData[numGPU];
//    DTYPE* devOldCentData[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//
//        // alloc dataset and drift array to GPU
//        gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//        // alloc the old position data structure
//        gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));
//
//        // copy input data to GPU
//        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//            centInfo, sizeof(CentInfo) * numCent,
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//            centData, sizeof(DTYPE) * numCent * numDim,
//            cudaMemcpyHostToDevice));
//    }
//
//    DTYPE* devMaxDriftArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
//    }
//
//    // centroid calculation data
//    DTYPE* devNewCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//    }
//
//    DTYPE* devOldCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//    }
//
//    unsigned int* devNewCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//    }
//
//    unsigned int* devOldCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
//    }
//
//    unsigned int* devConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//        gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//            hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//            cudaMemcpyHostToDevice));
//    }
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        clearCentCalcData << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//            devOldCentSum[gpuIter],
//            devNewCentCount[gpuIter],
//            devOldCentCount[gpuIter],
//            numCent,
//            numDim);
//
//    }
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//    }
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        // do single run of naive kmeans for initial centroid assignments
//        initRunKernel << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//            devCentInfo[gpuIter],
//            devPointData[gpuIter],
//            devPointLwrs[gpuIter],
//            devCentData[gpuIter],
//            numPnts[gpuIter],
//            numCent,
//            numGrp,
//            numDim);
//    }
//
//    CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//    }
//
//    DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//    }
//
//    CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//    DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//    for (int i = 0; i < numCent; i++)
//    {
//        for (int j = 0; j < numDim; j++)
//        {
//            newCentData[(i * numDim) + j] = 0;
//        }
//    }
//
//    DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//    DTYPE* newMaxDriftArr;
//    newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
//    for (int i = 0; i < numGrp; i++)
//    {
//        newMaxDriftArr[i] = 0.0;
//    }   
//
//    unsigned int doesNotConverge = 1;
//
//    // loop until convergence
//    while (doesNotConverge && index < maxIter)
//    {
//        doesNotConverge = 0;
//
//        for (int i = 0; i < numCent; i++)
//        {
//            newCentInfo[i].count = 0;
//        }
//
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            hostConFlagArr[gpuIter] = 0;
//        }
//
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//                cudaMemcpyHostToDevice));
//        }
//
//        // clear maintained data on device
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//
//        }
//
//
//        // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//                devPointData[gpuIter], devOldCentSum[gpuIter],
//                devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//                devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//
//        }
//
//        // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//                devCentData[gpuIter], devOldCentData[gpuIter],
//                devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//                devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//                devNewCentCount[gpuIter], numCent, numDim);
//
//        }        
//
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            assignPointsFull << <NBLOCKS, BLOCKSIZE, grpLclSize >> > (devPointInfo[gpuIter],
//                devCentInfo[gpuIter],
//                devPointData[gpuIter],
//                devPointLwrs[gpuIter],
//                devCentData[gpuIter],
//                devMaxDriftArr[gpuIter],
//                numPnts[gpuIter], numCent,
//                numGrp, numDim);
//
//        }
//
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//                devConFlagArr[gpuIter],
//                numPnts[gpuIter]);
//
//        }
//
//        index++;
//
//#pragma omp parallel for num_threads(numGPU)
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//                devConFlagArr[gpuIter], sizeof(unsigned int),
//                cudaMemcpyDeviceToHost));
//        }
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            if (hostConFlagArr[gpuIter])
//            {
//                doesNotConverge = 1;
//            }
//        }
//    }
//
//    // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//            devPointData[gpuIter], devOldCentSum[gpuIter],
//            devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//            devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//    }
//
//    // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//            devCentData[gpuIter], devOldCentData[gpuIter],
//            devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//            devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//            devNewCentCount[gpuIter], numCent, numDim);
//    } 
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaDeviceSynchronize();
//    }
//
//#pragma omp parallel for num_threads(numGPU)
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//
//        // copy finished clusters and points from device to host
//        gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//            devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//    }
//
//    // and the final centroid positions
//    gpuErrchk(cudaMemcpy(centData, devCentData[0],
//        sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//    *ranIter = index;
//
//    // clean up, return
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        cudaFree(devPointInfo[gpuIter]);
//        cudaFree(devPointData[gpuIter]);
//        cudaFree(devPointLwrs[gpuIter]);
//        cudaFree(devCentInfo[gpuIter]);
//        cudaFree(devCentData[gpuIter]);
//        cudaFree(devMaxDriftArr[gpuIter]);
//        cudaFree(devNewCentSum[gpuIter]);
//        cudaFree(devOldCentSum[gpuIter]);
//        cudaFree(devNewCentCount[gpuIter]);
//        cudaFree(devOldCentCount[gpuIter]);
//        cudaFree(devConFlagArr[gpuIter]);
//    }
//
//    free(allCentInfo);
//    free(allCentData);
//    free(newCentInfo);
//    free(newCentData);
//    free(oldCentData);
//    free(pointLwrs);
//
//    endTime = omp_get_wtime();
//    return endTime - startTime;
//}
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
//    unsigned int* ranIter)
//{
//
//    // variable initialization
//    int gpuIter;
//    const int numGPUU = 1;
//    int numPnts[numGPUU];
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//        {
//            numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//        }
//
//        else
//        {
//            numPnts[gpuIter] = numPnt / numGPU;
//        }
//    }
//
//    unsigned int hostConFlagArr[numGPUU];
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        hostConFlagArr[gpuIter] = 1;
//    }
//
//    unsigned int* hostConFlagPtrArr[numGPUU];
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//    }
//
//    int grpLclSize = sizeof(unsigned int) * numGrp * BLOCKSIZE;
//
//    int index = 1;
//
//    unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//    // group centroids
//    groupCent(centInfo, centData, numCent, numGrp, numDim);
//
//    // create lower bound data on host
//    DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
//    for (int i = 0; i < numPnt * numGrp; i++)
//    {
//        pointLwrs[i] = INFINITY;
//    }
//
//    // store dataset on device
//    PointInfo* devPointInfo[numGPUU];
//    DTYPE* devPointData[numGPUU];
//    DTYPE* devPointLwrs[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        cudaSetDevice(gpuIter);
//
//        // alloc dataset to GPU
//        gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//        // copy input data to GPU
//        gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//            pointInfo + (gpuIter * numPnt / numGPU),
//            (numPnts[gpuIter]) * sizeof(PointInfo),
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//        gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//            pointData + ((gpuIter * numPnt / numGPU) * numDim),
//            sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
//            numGrp));
//
//        gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
//            pointLwrs + ((gpuIter * numPnt / numGPU) * numGrp),
//            sizeof(DTYPE) * numPnts[gpuIter] * numGrp,
//            cudaMemcpyHostToDevice));
//    }
//
//    // store centroids on device
//    CentInfo* devCentInfo[numGPUU];
//    DTYPE* devCentData[numGPUU];
//    DTYPE* devOldCentData[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//
//        // alloc dataset and drift array to GPU
//        gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//        // alloc the old position data structure
//        gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));
//
//        // copy input data to GPU
//        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//            centInfo, sizeof(CentInfo) * numCent,
//            cudaMemcpyHostToDevice));
//
//        gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//            centData, sizeof(DTYPE) * numCent * numDim,
//            cudaMemcpyHostToDevice));
//    }
//
//    DTYPE* devMaxDriftArr[numGPUU];
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
//    }
//
//    // centroid calculation data
//    DTYPE* devNewCentSum[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//    }
//
//    DTYPE* devOldCentSum[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//    }
//
//    unsigned int* devNewCentCount[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//    }
//
//    unsigned int* devOldCentCount[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
//    }
//
//    unsigned int* devConFlagArr[numGPUU];
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//        gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//            hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//            cudaMemcpyHostToDevice));
//    }
//
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        clearCentCalcData << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//            devOldCentSum[gpuIter],
//            devNewCentCount[gpuIter],
//            devOldCentCount[gpuIter],
//            numCent,
//            numDim);
//
//    }
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//    }
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        // do single run of naive kmeans for initial centroid assignments
//        initRunKernel << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//            devCentInfo[gpuIter],
//            devPointData[gpuIter],
//            devPointLwrs[gpuIter],
//            devCentData[gpuIter],
//            numPnts[gpuIter],
//            numCent,
//            numGrp,
//            numDim);
//    }
//
//    CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//    }
//
//    DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//    }
//
//    CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//    DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//    for (int i = 0; i < numCent; i++)
//    {
//        for (int j = 0; j < numDim; j++)
//        {
//            newCentData[(i * numDim) + j] = 0;
//        }
//    }
//
//    DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//    DTYPE* newMaxDriftArr;
//    newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
//    for (int i = 0; i < numGrp; i++)
//    {
//        newMaxDriftArr[i] = 0.0;
//    }
//
//
//
//    unsigned int doesNotConverge = 1;
//
//    // loop until convergence
//    while (doesNotConverge && index < maxIter)
//    {
//        std::cout << "ITERATION" << std::endl;
//        doesNotConverge = 0;
//
//        for (int i = 0; i < numCent; i++)
//        {
//            newCentInfo[i].count = 0;
//        }
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            hostConFlagArr[gpuIter] = 0;
//        }
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//                cudaMemcpyHostToDevice));
//        }
//
//        // clear maintained data on device
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//
//        }
//
//
//        // calculate data necessary to make new centroids
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//                devPointData[gpuIter], devOldCentSum[gpuIter],
//                devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//                devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//
//        }
//
//        // make new centroids
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//                devCentData[gpuIter], devOldCentData[gpuIter],
//                devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//                devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//                devNewCentCount[gpuIter], numCent, numDim);
//
//        }
//
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            assignPointsSimple << <NBLOCKS, BLOCKSIZE, grpLclSize >> > (devPointInfo[gpuIter],
//                devCentInfo[gpuIter],
//                devPointData[gpuIter],
//                devPointLwrs[gpuIter],
//                devCentData[gpuIter],
//                devMaxDriftArr[gpuIter],
//                numPnts[gpuIter], numCent,
//                numGrp, numDim);
//
//        }
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//                devConFlagArr[gpuIter],
//                numPnts[gpuIter]);
//
//        }
//
//        index++;
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            gpuErrchk(cudaSetDevice(gpuIter));
//            gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//                devConFlagArr[gpuIter], sizeof(unsigned int),
//                cudaMemcpyDeviceToHost));
//        }
//
//        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//        {
//            if (hostConFlagArr[gpuIter])
//            {
//                doesNotConverge = 1;
//            }
//        }
//    }
//
//    // calculate data necessary to make new centroids
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//            devPointData[gpuIter], devOldCentSum[gpuIter],
//            devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//            devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//    }
//
//    // make new centroids
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//            devCentData[gpuIter], devOldCentData[gpuIter],
//            devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//            devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//            devNewCentCount[gpuIter], numCent, numDim);
//    }
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//        cudaDeviceSynchronize();
//    }
//
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        gpuErrchk(cudaSetDevice(gpuIter));
//
//        // copy finished clusters and points from device to host
//        gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//            devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//    }
//
//    // and the final centroid positions
//    gpuErrchk(cudaMemcpy(centData, devCentData[0],
//        sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//    *ranIter = index;
//
//    // clean up, return
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        cudaFree(devPointInfo[gpuIter]);
//        cudaFree(devPointData[gpuIter]);
//        cudaFree(devPointLwrs[gpuIter]);
//        cudaFree(devCentInfo[gpuIter]);
//        cudaFree(devCentData[gpuIter]);
//        cudaFree(devMaxDriftArr[gpuIter]);
//        cudaFree(devNewCentSum[gpuIter]);
//        cudaFree(devOldCentSum[gpuIter]);
//        cudaFree(devNewCentCount[gpuIter]);
//        cudaFree(devOldCentCount[gpuIter]);
//        cudaFree(devConFlagArr[gpuIter]);
//    }
//
//    free(allCentInfo);
//    free(allCentData);
//    free(newCentInfo);
//    free(newCentData);
//    free(oldCentData);
//    free(pointLwrs);
//
//    return 0.0;
//}
//
//DTYPE* storeDataOnGPU(DTYPE* data,
//    const int numVec,
//    const int numFeat)
//{
//    DTYPE* devData = NULL;
//    gpuErrchk(cudaMalloc(&devData, sizeof(DTYPE) * numVec * numFeat));
//    gpuErrchk(cudaMemcpy(devData, data,
//        sizeof(DTYPE) * numVec * numFeat, cudaMemcpyHostToDevice));
//    return devData;
//}
//
//PointInfo* storePointInfoOnGPU(PointInfo* pointInfo,
//    const int numPnt)
//{
//    PointInfo* devPointInfo = NULL;
//    gpuErrchk(cudaMalloc(&devPointInfo, sizeof(PointInfo) * numPnt));
//    gpuErrchk(cudaMemcpy(devPointInfo, pointInfo,
//        sizeof(PointInfo) * numPnt, cudaMemcpyHostToDevice));
//    return devPointInfo;
//}
//
//CentInfo* storeCentInfoOnGPU(CentInfo* centInfo,
//    const int numCent)
//{
//    CentInfo* devCentInfo = NULL;
//    gpuErrchk(cudaMalloc(&devCentInfo, sizeof(CentInfo) * numCent));
//    gpuErrchk(cudaMemcpy(devCentInfo, centInfo,
//        sizeof(CentInfo) * numCent, cudaMemcpyHostToDevice));
//    return devCentInfo;
//}
//
//void warmupGPU(const int numGPU)
//{
//    int gpuIter;
//    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//    {
//        cudaSetDevice(gpuIter);
//        cudaDeviceSynchronize();
//    }
//}
//
////double startSimpleOnGPU(PointInfo* pointInfo,
////    CentInfo* centInfo,
////    DTYPE* pointData,
////    DTYPE* centData,
////    const int numPnt,
////    const int numCent,
////    const int numGrp,
////    const int numDim,
////    const int maxIter,
////    const int numGPU,
////    unsigned int* ranIter,
////    unsigned long long int* countPtr)
////{
////
////    // variable initialization
////    int gpuIter;
////    const int numGPUU = 1;
////    int numPnts[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
////        {
////            numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
////        }
////
////        else
////        {
////            numPnts[gpuIter] = numPnt / numGPU;
////        }
////    }
////
////    unsigned int hostConFlagArr[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        hostConFlagArr[gpuIter] = 1;
////    }
////
////    unsigned int* hostConFlagPtrArr[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
////    }
////
////    unsigned long long int hostDistCalc = 0;
////    unsigned long long int* hostDistCalcCount = &hostDistCalc;
////
////    unsigned long long int* hostDistCalcCountArr;
////    hostDistCalcCountArr = (unsigned long long int*)malloc(sizeof(unsigned long long int) * numGPU);
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        hostDistCalcCountArr[gpuIter] = 0;
////    }
////    unsigned long long int* devDistCalcCountArr[numGPUU];
////
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        gpuErrchk(cudaMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
////        gpuErrchk(cudaMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter],
////            sizeof(unsigned long long int), cudaMemcpyHostToDevice));
////    }
////
////    int grpLclSize = sizeof(unsigned int) * numGrp * BLOCKSIZE;
////
////    int index = 1;
////
////    //unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);
////    unsigned int NBLOCKS[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        NBLOCKS[gpuIter] = ceil(numPnts[gpuIter] * 1.0 / BLOCKSIZE * 1.0);
////    }
////
////
////    // group centroids
////    groupCent(centInfo, centData, numCent, numGrp, numDim);
////
////    // create lower bound data on host
////    DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
////    for (int i = 0; i < numPnt * numGrp; i++)
////    {
////        pointLwrs[i] = INFINITY;
////    }
////
////    // store dataset on device
////    PointInfo* devPointInfo[numGPUU];
////    DTYPE* devPointData[numGPUU];
////    DTYPE* devPointLwrs[numGPUU];
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        cudaSetDevice(gpuIter);
////
////        // alloc dataset to GPU
////        gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
////
////        // copy input data to GPU
////        gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
////            pointInfo + (gpuIter * numPnt / numGPU),
////            (numPnts[gpuIter]) * sizeof(PointInfo),
////            cudaMemcpyHostToDevice));
////
////        gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
////
////        gpuErrchk(cudaMemcpy(devPointData[gpuIter],
////            pointData + ((gpuIter * numPnt / numGPU) * numDim),
////            sizeof(DTYPE) * numPnts[gpuIter] * numDim,
////            cudaMemcpyHostToDevice));
////
////        gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
////            numGrp));
////
////        gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
////            pointLwrs + ((gpuIter * numPnt / numGPU) * numGrp),
////            sizeof(DTYPE) * numPnts[gpuIter] * numGrp,
////            cudaMemcpyHostToDevice));
////    }
////
////    // store centroids on device
////    CentInfo* devCentInfo[numGPUU];
////    DTYPE* devCentData[numGPUU];
////    DTYPE* devOldCentData[numGPUU];
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////
////        // alloc dataset and drift array to GPU
////        gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
////
////        // alloc the old position data structure
////        gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));
////
////        // copy input data to GPU
////        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
////            centInfo, sizeof(CentInfo) * numCent,
////            cudaMemcpyHostToDevice));
////
////        gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
////        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
////            centData, sizeof(DTYPE) * numCent * numDim,
////            cudaMemcpyHostToDevice));
////    }
////
////    DTYPE* devMaxDriftArr[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
////    }
////
////    // centroid calculation data
////    DTYPE* devNewCentSum[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
////    }
////
////    DTYPE* devOldCentSum[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
////    }
////
////    unsigned int* devNewCentCount[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
////    }
////
////    unsigned int* devOldCentCount[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
////    }
////
////    unsigned int* devConFlagArr[numGPUU];
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
////        gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
////            hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
////            cudaMemcpyHostToDevice));
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        clearCentCalcData << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devNewCentSum[gpuIter],
////            devOldCentSum[gpuIter],
////            devNewCentCount[gpuIter],
////            devOldCentCount[gpuIter],
////            numCent,
////            numDim);
////
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        clearDriftArr << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        // do single run of naive kmeans for initial centroid assignments
////        initRunKernel << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter],
////            devCentInfo[gpuIter],
////            devPointData[gpuIter],
////            devPointLwrs[gpuIter],
////            devCentData[gpuIter],
////            numPnts[gpuIter],
////            numCent,
////            numGrp,
////            numDim,
////            devDistCalcCountArr[gpuIter]);
////    }
////
////    CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
////    }
////
////    DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
////    }
////
////    CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
////
////    DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
////    for (int i = 0; i < numCent; i++)
////    {
////        for (int j = 0; j < numDim; j++)
////        {
////            newCentData[(i * numDim) + j] = 0;
////        }
////    }
////
////    DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
////
////    DTYPE* newMaxDriftArr;
////    newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
////    for (int i = 0; i < numGrp; i++)
////    {
////        newMaxDriftArr[i] = 0.0;
////    }
////  
////    unsigned int doesNotConverge = 1;
////
////    // loop until convergence
////    while (doesNotConverge && index < maxIter)
////    {
////        doesNotConverge = 0;
////
////        for (int i = 0; i < numCent; i++)
////        {
////            newCentInfo[i].count = 0;
////        }
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            hostConFlagArr[gpuIter] = 0;
////        }
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
////                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
////                cudaMemcpyHostToDevice));
////        }
////
////        // clear maintained data on device
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            clearDriftArr << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
////
////        }
////
////
////        // calculate data necessary to make new centroids
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            calcCentData << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
////                devPointData[gpuIter], devOldCentSum[gpuIter],
////                devNewCentSum[gpuIter], devOldCentCount[gpuIter],
////                devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
////
////        }
////
////        // make new centroids
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            calcNewCentroids << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
////                devCentData[gpuIter], devOldCentData[gpuIter],
////                devOldCentSum[gpuIter], devNewCentSum[gpuIter],
////                devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
////                devNewCentCount[gpuIter], numCent, numDim);
////
////        }
////
////        
////
////        /*
////        if (numGPU == 2)
////        {
////          if (index == 20)
////          {
////            writeData(newCentData, numCent, numDim, "centroidsAt1_2gpu.txt");
////          }
////        }
////
////        if (numGPU == 3)
////        {
////          if (index == 20)
////          {
////            writeData(newCentData, numCent, numDim, "centroidsAt1_3gpu.txt");
////          }
////        }
////        */
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            assignPointsSimple << <NBLOCKS[gpuIter], BLOCKSIZE, grpLclSize >> > (devPointInfo[gpuIter],
////                devCentInfo[gpuIter],
////                devPointData[gpuIter],
////                devPointLwrs[gpuIter],
////                devCentData[gpuIter],
////                devMaxDriftArr[gpuIter],
////                numPnts[gpuIter], numCent,
////                numGrp, numDim,
////                devDistCalcCountArr[gpuIter]);
////
////        }
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            checkConverge << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter],
////                devConFlagArr[gpuIter],
////                numPnts[gpuIter]);
////
////        }
////
////        index++;
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
////            gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
////                devConFlagArr[gpuIter], sizeof(unsigned int),
////                cudaMemcpyDeviceToHost));
////        }
////
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            if (hostConFlagArr[gpuIter])
////            {
////                doesNotConverge = 1;
////            }
////        }
////    }
////
////    *hostDistCalcCount = 0;
////
////    // calculate data necessary to make new centroids
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        calcCentData << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
////            devPointData[gpuIter], devOldCentSum[gpuIter],
////            devNewCentSum[gpuIter], devOldCentCount[gpuIter],
////            devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
////    }
////
////    // make new centroids
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        calcNewCentroids << <NBLOCKS[gpuIter], BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
////            devCentData[gpuIter], devOldCentData[gpuIter],
////            devOldCentSum[gpuIter], devNewCentSum[gpuIter],
////            devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
////            devNewCentCount[gpuIter], numCent, numDim);
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        cudaDeviceSynchronize();
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////
////        // copy finished clusters and points from device to host
////        gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
////            devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
////    }
////
////    // and the final centroid positions
////    gpuErrchk(cudaMemcpy(centData, devCentData[0],
////        sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        gpuErrchk(cudaSetDevice(gpuIter));
////        gpuErrchk(cudaMemcpy(&hostDistCalcCountArr[gpuIter],
////            devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
////            cudaMemcpyDeviceToHost));
////    }
////
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        *hostDistCalcCount += hostDistCalcCountArr[gpuIter];
////    }
////
////    *countPtr = *hostDistCalcCount;
////
////    *ranIter = index;
////
////    // clean up, return
////    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////    {
////        cudaFree(devPointInfo[gpuIter]);
////        cudaFree(devPointData[gpuIter]);
////        cudaFree(devPointLwrs[gpuIter]);
////        cudaFree(devCentInfo[gpuIter]);
////        cudaFree(devCentData[gpuIter]);
////        cudaFree(devMaxDriftArr[gpuIter]);
////        cudaFree(devNewCentSum[gpuIter]);
////        cudaFree(devOldCentSum[gpuIter]);
////        cudaFree(devNewCentCount[gpuIter]);
////        cudaFree(devOldCentCount[gpuIter]);
////        cudaFree(devConFlagArr[gpuIter]);
////    }
////
////    free(allCentInfo);
////    free(allCentData);
////    free(newCentInfo);
////    free(newCentData);
////    free(oldCentData);
////    free(pointLwrs);
////
////    return 0.0;
////}