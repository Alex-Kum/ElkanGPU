//https://github.com/ctaylor389/k_means_yinyang_gpu


//#include "yy_kmean.h"
//#include "gpufunctions.h"
//#include "omp.h"
//#include <vector>
//
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//   if (code != cudaSuccess)
//   {
//       fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//   }
//}
//
//DTYPE calcDiss(DTYPE* vec1,
//   DTYPE* vec2,
//   const int numDim)
//{
//   unsigned int index;
//   DTYPE total = 0.0;
//   DTYPE square;
//
//   for (index = 0; index < numDim; index++)
//   {
//       square = (vec1[index] - vec2[index]);
//       total += square * square;
//   }
//
//   return sqrt(total);
//}
//
//bool verifyYY(DTYPE* pointData, DTYPE* centData, PointInfo* pointInfo, int n, int k, int dim) {
//   for (int i = 0; i < n; i++) {
//       int closest = pointInfo[i].centroidIndex;
//       double closest_dist2 = calcDiss(&pointData[i * dim], &centData[pointInfo[i].centroidIndex * dim], dim);
//       double original_closest_dist2 = closest_dist2;
//       for (int j = 0; j < k; ++j) {
//           if (j == closest) {
//               continue;
//           }
//           double d2 = calcDiss(&pointData[i * dim], &centData[pointInfo[j].centroidIndex * dim], dim);
//
//           if (d2 < closest_dist2) {
//               closest = j;
//               closest_dist2 = d2;
//           }
//       }
//
//       if (closest != pointInfo[i].centroidIndex) {
//           std::cerr << "assignment error:" << std::endl;
//           std::cerr << "point index           = " << i << std::endl;
//           std::cerr << "closest center        = " << closest << std::endl;
//           std::cerr << "closest center dist2  = " << closest_dist2 << std::endl;
//           std::cerr << "assigned center       = " << pointInfo[i].centroidIndex << std::endl;
//           std::cerr << "assigned center dist2 = " << original_closest_dist2 << std::endl;
//           return false;
//       }
//   }
//   return true;
//}
//
//int generateCentWithDataSame(CentInfo* centInfo,
//   DTYPE* centData,
//   DTYPE* copyData,
//   const int numCent,
//   const int numCopy,
//   const int numDim)
//{
//   //srand(90);
//   int* chosen_pts = new int[numCent];
//   for (int i = 0; i < numCent; ++i) {
//       bool acceptable = true;
//       do {
//           acceptable = true;
//           auto ran = rand() % numCopy;
//           //std::cout << "Rand: " << i << " = " << ran << std::endl;
//           chosen_pts[i] = ran;
//           for (int j = 0; j < i; ++j) {
//               if (chosen_pts[i] == chosen_pts[j]) {
//                   acceptable = false;
//                   break;
//               }
//           }
//       } while (!acceptable);
//       //double* cdp = centData + i * numDim;
//       //memcpy(cdp, centData + chosen_pts[i] * numDim, sizeof(double) * numDim);
//       for (int j = 0; j < numDim; j++) {
//           centData[i * numDim + j] =
//               copyData[chosen_pts[i] * numDim + j];
//       }
//       centInfo[i].groupNum = -1;
//       centInfo[i].drift = 0.0;
//       centInfo[i].count = 0;
//   }
//   delete[] chosen_pts;
//   return 0;
//}
//
//
//int generateCentWithData(CentInfo* centInfo,
//   DTYPE* centData,
//   DTYPE* copyData,
//   const int numCent,
//   const int numCopy,
//   const int numDim)
//{
//   //srand(90);
//   int i;
//   int j;
//   int randomMax = numCopy / numCent;
//   for (i = 0; i < numCent; i++)
//   {
//       for (j = 0; j < numDim; j++)
//       {
//           centData[(i * numDim) + j] =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
//               copyData[((i * randomMax) +
//                   (rand() % randomMax)) * numDim + j];
//       }
//       centInfo[i].groupNum = -1;
//       centInfo[i].drift = 0.0;
//       centInfo[i].count = 0;
//   }
//   return 0;
//}
//
//int groupCent(CentInfo* centInfo,
//   DTYPE* centData,
//   const int numCent,
//   const int numGrp,
//   const int numDim)
//{
//   CentInfo* overInfo = (CentInfo*)malloc(sizeof(CentInfo) * numGrp);
//   DTYPE* overData = (DTYPE*)malloc(sizeof(DTYPE) * numGrp * numDim);
//   generateCentWithData(overInfo, overData, centData, numGrp, numCent, numDim);
//
//   unsigned int iterIndex, centIndex, grpIndex, dimIndex, assignment;
//
//  /* const int numDimm = 10;
//   const int numGrpp = 20;*/
//   DTYPE currMin = INFINITY;
//   DTYPE currDistance = INFINITY;
//   //DTYPE origVec[numGrp][numDim];
//   DTYPE* origVec = (DTYPE*)malloc(sizeof(DTYPE) * numGrp * numDim);
//
//   for (iterIndex = 0; iterIndex < 5; iterIndex++)
//   {
//       // assignment
//       for (centIndex = 0; centIndex < numCent; centIndex++)
//       {
//           for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//           {
//               currDistance = calcDisCPU(&centData[centIndex * numDim],
//                   &overData[grpIndex * numDim],
//                   numDim);
//               if (currDistance < currMin)
//               {
//                   centInfo[centIndex].groupNum = grpIndex;
//                   currMin = currDistance;
//               }
//           }
//           currMin = INFINITY;
//       }
//       // update over centroids
//       for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//       {
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               origVec[(grpIndex * numDim) + dimIndex] =
//                   overData[(grpIndex * numDim) + dimIndex];
//               overData[(grpIndex * numDim) + dimIndex] = 0.0;
//           }
//           overInfo[grpIndex].count = 0;
//       }
//
//       // update over centroids to be average of group
//       for (centIndex = 0; centIndex < numCent; centIndex++)
//       {
//           assignment = centInfo[centIndex].groupNum;
//           overInfo[assignment].count += 1;
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               overData[(assignment * numDim) + dimIndex] +=
//                   centData[(centIndex * numDim) + dimIndex];
//           }
//       }
//
//
//       for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//       {
//           if (overInfo[grpIndex].count > 0)
//           {
//               for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//               {
//                   overData[(grpIndex * numDim) + dimIndex] /=
//                       overInfo[grpIndex].count;
//               }
//           }
//           else
//           {
//               for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//               {
//                   overData[(grpIndex * numDim) + dimIndex] =
//                       origVec[(grpIndex * numDim) + dimIndex];
//               }
//           }
//       }
//   }
//   free(overData);
//   free(overInfo);
//   free(origVec);
//   return 0;
//}
//
//double startFullOnGPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim,
//   const int maxIter,
//   const int numGPUU,
//   unsigned int* ranIter)
//{
//   // start timer
//   double startTime, endTime;
//   startTime = omp_get_wtime();
//
//   // variable initialization
//   int gpuIter;
//   const int numGPU = 1;
//   int numPnts[numGPU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//       {
//           numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//       }
//
//       else
//       {
//           numPnts[gpuIter] = numPnt / numGPU;
//       }
//   }
//
//   unsigned int hostConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagArr[gpuIter] = 1;
//   }
//
//   unsigned int* hostConFlagPtrArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//   }
//
//   int grpLclSize = sizeof(unsigned int) * numGrp * BLOCKSIZE;
//
//   int index = 1;
//
//   unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//   // group centroids
//   groupCent(centInfo, centData, numCent, numGrp, numDim);
//
//   // create lower bound data on host
//   DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
//   for (int i = 0; i < numPnt * numGrp; i++)
//   {
//       pointLwrs[i] = INFINITY;
//   }
//
//   // store dataset on device
//   PointInfo* devPointInfo[numGPU];
//   DTYPE* devPointData[numGPU];
//   DTYPE* devPointLwrs[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaSetDevice(gpuIter);
//
//       // alloc dataset to GPU
//       gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//           pointInfo + (gpuIter * numPnt / numGPU),
//           (numPnts[gpuIter]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//       gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//           pointData + ((gpuIter * numPnt / numGPU) * numDim),
//           sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
//           numGrp));
//
//       gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
//           pointLwrs + ((gpuIter * numPnt / numGPU) * numGrp),
//           sizeof(DTYPE) * numPnts[gpuIter] * numGrp,
//           cudaMemcpyHostToDevice));
//   }
//
//   // store centroids on device
//   CentInfo* devCentInfo[numGPU];
//   DTYPE* devCentData[numGPU];
//   DTYPE* devOldCentData[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // alloc dataset and drift array to GPU
//       gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//       // alloc the old position data structure
//       gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//           centInfo, sizeof(CentInfo) * numCent,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//       gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//           centData, sizeof(DTYPE) * numCent * numDim,
//           cudaMemcpyHostToDevice));
//   }
//
//   DTYPE* devMaxDriftArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
//   }
//
//   // centroid calculation data
//   DTYPE* devNewCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   DTYPE* devOldCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   unsigned int* devNewCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devOldCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//       gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//           hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//           cudaMemcpyHostToDevice));
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearCentCalcData << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//           devOldCentSum[gpuIter],
//           devNewCentCount[gpuIter],
//           devOldCentCount[gpuIter],
//           numCent,
//           numDim);
//
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       // do single run of naive kmeans for initial centroid assignments
//       initRunKernel << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//           devCentInfo[gpuIter],
//           devPointData[gpuIter],
//           devPointLwrs[gpuIter],
//           devCentData[gpuIter],
//           numPnts[gpuIter],
//           numCent,
//           numGrp,
//           numDim);
//   }
//
//   CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//   }
//
//   DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   }
//
//   CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//   DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   for (int i = 0; i < numCent; i++)
//   {
//       for (int j = 0; j < numDim; j++)
//       {
//           newCentData[(i * numDim) + j] = 0;
//       }
//   }
//
//   DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//   DTYPE* newMaxDriftArr;
//   newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
//   for (int i = 0; i < numGrp; i++)
//   {
//       newMaxDriftArr[i] = 0.0;
//   }   
//
//   unsigned int doesNotConverge = 1;
//
//   // loop until convergence
//   while (doesNotConverge && index < maxIter)
//   {
//       doesNotConverge = 0;
//
//       for (int i = 0; i < numCent; i++)
//       {
//           newCentInfo[i].count = 0;
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           hostConFlagArr[gpuIter] = 0;
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//               hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyHostToDevice));
//       }
//
//       // clear maintained data on device
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//
//       }
//
//
//       // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devPointData[gpuIter], devOldCentSum[gpuIter],
//               devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//
//       }
//
//       // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devCentData[gpuIter], devOldCentData[gpuIter],
//               devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//               devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter], numCent, numDim);
//
//       }        
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           assignPointsFull << <NBLOCKS, BLOCKSIZE, grpLclSize >> > (devPointInfo[gpuIter],
//               devCentInfo[gpuIter],
//               devPointData[gpuIter],
//               devPointLwrs[gpuIter],
//               devCentData[gpuIter],
//               devMaxDriftArr[gpuIter],
//               numPnts[gpuIter], numCent,
//               numGrp, numDim);
//
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devConFlagArr[gpuIter],
//               numPnts[gpuIter]);
//
//       }
//
//       index++;
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//               devConFlagArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyDeviceToHost));
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           if (hostConFlagArr[gpuIter])
//           {
//               doesNotConverge = 1;
//           }
//       }
//   }
//
//   // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devPointData[gpuIter], devOldCentSum[gpuIter],
//           devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//   }
//
//   // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devCentData[gpuIter], devOldCentData[gpuIter],
//           devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//           devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numCent, numDim);
//   } 
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaDeviceSynchronize();
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // copy finished clusters and points from device to host
//       gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//           devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//   }
//
//   // and the final centroid positions
//   gpuErrchk(cudaMemcpy(centData, devCentData[0],
//       sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//   *ranIter = index;
//
//   // clean up, return
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaFree(devPointInfo[gpuIter]);
//       cudaFree(devPointData[gpuIter]);
//       cudaFree(devPointLwrs[gpuIter]);
//       cudaFree(devCentInfo[gpuIter]);
//       cudaFree(devCentData[gpuIter]);
//       cudaFree(devMaxDriftArr[gpuIter]);
//       cudaFree(devNewCentSum[gpuIter]);
//       cudaFree(devOldCentSum[gpuIter]);
//       cudaFree(devNewCentCount[gpuIter]);
//       cudaFree(devOldCentCount[gpuIter]);
//       cudaFree(devConFlagArr[gpuIter]);
//   }
//
//   free(allCentInfo);
//   free(allCentData);
//   free(newCentInfo);
//   free(newCentData);
//   free(oldCentData);
//   free(pointLwrs);
//
//   endTime = omp_get_wtime();
//   return endTime - startTime;
//}
//
//
//double startSimpleOnGPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim,
//   const int maxIter,
//   const int numGPU,
//   unsigned int* ranIter)
//{
//   //std::cout << "START SIMPLE" << std::endl;
//   // variable initialization
//    unsigned long long int* d_countDistances;
//    cudaMalloc(&d_countDistances, 1 * sizeof(unsigned long long int));
//    cudaMemset(d_countDistances, 0, 1 * sizeof(unsigned long long int));
//
//   int gpuIter;
//   const int numGPUU = 1;
//   int numPnts[numGPUU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//       {
//           numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//       }
//
//       else
//       {
//           numPnts[gpuIter] = numPnt / numGPU;
//       }
//   }
//
//   unsigned int hostConFlagArr[numGPUU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagArr[gpuIter] = 1;
//   }
//
//   unsigned int* hostConFlagPtrArr[numGPUU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//   }
//
//   int grpLclSize = sizeof(unsigned int) * numGrp * BLOCKSIZE;
//
//   int index = 1;
//
//   unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//   // group centroids
//   groupCent(centInfo, centData, numCent, numGrp, numDim);
//
//   // create lower bound data on host
//   DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
//   for (int i = 0; i < numPnt * numGrp; i++)
//   {
//       pointLwrs[i] = INFINITY;
//   }
//
//   // store dataset on device
//   PointInfo* devPointInfo[numGPUU];
//   DTYPE* devPointData[numGPUU];
//   DTYPE* devPointLwrs[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaSetDevice(gpuIter);
//
//       // alloc dataset to GPU
//       gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//           pointInfo + (gpuIter * numPnt / numGPU),
//           (numPnts[gpuIter]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//       gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//           pointData + ((gpuIter * numPnt / numGPU) * numDim),
//           sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
//           numGrp));
//
//       gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
//           pointLwrs + ((gpuIter * numPnt / numGPU) * numGrp),
//           sizeof(DTYPE) * numPnts[gpuIter] * numGrp,
//           cudaMemcpyHostToDevice));
//   }
//
//   // store centroids on device
//   CentInfo* devCentInfo[numGPUU];
//   DTYPE* devCentData[numGPUU];
//   DTYPE* devOldCentData[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // alloc dataset and drift array to GPU
//       gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//       // alloc the old position data structure
//       gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//           centInfo, sizeof(CentInfo) * numCent,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//       gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//           centData, sizeof(DTYPE) * numCent * numDim,
//           cudaMemcpyHostToDevice));
//   }
//
//   DTYPE* devMaxDriftArr[numGPUU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
//   }
//
//   // centroid calculation data
//   DTYPE* devNewCentSum[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   DTYPE* devOldCentSum[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   unsigned int* devNewCentCount[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devOldCentCount[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devConFlagArr[numGPUU];
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//       gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//           hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//           cudaMemcpyHostToDevice));
//   }
//
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearCentCalcData << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//           devOldCentSum[gpuIter],
//           devNewCentCount[gpuIter],
//           devOldCentCount[gpuIter],
//           numCent,
//           numDim);
//
//   }
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//   }
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       // do single run of naive kmeans for initial centroid assignments
//       initRunKernel << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//           devCentInfo[gpuIter],
//           devPointData[gpuIter],
//           devPointLwrs[gpuIter],
//           devCentData[gpuIter],
//           numPnts[gpuIter],
//           numCent,
//           numGrp,
//           numDim);
//   }
//
//   CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//   }
//
//   DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   }
//
//   CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//   DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   for (int i = 0; i < numCent; i++)
//   {
//       for (int j = 0; j < numDim; j++)
//       {
//           newCentData[(i * numDim) + j] = 0;
//       }
//   }
//
//   DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//   DTYPE* newMaxDriftArr;
//   newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
//   for (int i = 0; i < numGrp; i++)
//   {
//       newMaxDriftArr[i] = 0.0;
//   }
//
//
//
//   unsigned int doesNotConverge = 1;
//
//   // loop until convergence
//   while (doesNotConverge && index < maxIter)
//   {
//       //std::cout << "WUFF WUFF ITERATION" << std::endl;
//       doesNotConverge = 0;
//
//       for (int i = 0; i < numCent; i++)
//       {
//           newCentInfo[i].count = 0;
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           hostConFlagArr[gpuIter] = 0;
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//               hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyHostToDevice));
//       }
//
//       // clear maintained data on device
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDriftArr[gpuIter], numGrp);
//
//       }
//
//
//       // calculate data necessary to make new centroids
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devPointData[gpuIter], devOldCentSum[gpuIter],
//               devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//
//       }
//
//       // make new centroids
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devCentData[gpuIter], devOldCentData[gpuIter],
//               devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//               devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter], numCent, numDim);
//
//       }
//
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           assignPointsSimple << <NBLOCKS, BLOCKSIZE, grpLclSize >> > (devPointInfo[gpuIter],
//               devCentInfo[gpuIter],
//               devPointData[gpuIter],
//               devPointLwrs[gpuIter],
//               devCentData[gpuIter],
//               devMaxDriftArr[gpuIter],
//               numPnts[gpuIter], numCent,
//               numGrp, numDim, d_countDistances);
//       }
//
//       /*gpuErrchk(cudaMemcpy(pointInfo + (0 * numPnt / numGPU),
//           devPointInfo[0],            
//           (numPnts[0]) * sizeof(PointInfo),
//           cudaMemcpyDeviceToHost));
//       gpuErrchk(cudaMemcpy(centData,
//           devCentData[0], sizeof(DTYPE)* numCent* numDim,
//           cudaMemcpyDeviceToHost));
//       gpuErrchk(cudaMemcpy(pointInfo + (0 * numPnt / numGPU),
//           devPointInfo[0],            
//           (numPnts[0]) * sizeof(PointInfo),
//           cudaMemcpyDeviceToHost));
//       verifyYY(pointData, centData, pointInfo, numPnt, numCent, numDim);
//       gpuErrchk(cudaMemcpy(devPointInfo[0],
//           pointInfo + (0 * numPnt / numGPU),
//           (numPnts[0]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));
//       gpuErrchk(cudaMemcpy(devCentData[0],
//           centData, sizeof(DTYPE) * numCent * numDim,
//           cudaMemcpyHostToDevice));
//       gpuErrchk(cudaMemcpy(devPointInfo[0],
//           pointInfo + (0 * numPnt / numGPU),
//           (numPnts[0]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));*/
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devConFlagArr[gpuIter],
//               numPnts[gpuIter]);
//
//       }
//
//       index++;
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//               devConFlagArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyDeviceToHost));
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           if (hostConFlagArr[gpuIter])
//           {
//               doesNotConverge = 1;
//           }
//       }
//   }
//
//   // calculate data necessary to make new centroids
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devPointData[gpuIter], devOldCentSum[gpuIter],
//           devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//   }
//
//   // make new centroids
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devCentData[gpuIter], devOldCentData[gpuIter],
//           devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//           devMaxDriftArr[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numCent, numDim);
//   }
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaDeviceSynchronize();
//   }
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // copy finished clusters and points from device to host
//       gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//           devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//   }
//
//   // and the final centroid positions
//   gpuErrchk(cudaMemcpy(centData, devCentData[0],
//       sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//   *ranIter = index;
//   unsigned long long int cDist;
//   cudaMemcpy(&cDist, d_countDistances, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//   std::cout << "distance calculations: " << cDist << std::endl;
//   // clean up, return
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaFree(devPointInfo[gpuIter]);
//       cudaFree(devPointData[gpuIter]);
//       cudaFree(devPointLwrs[gpuIter]);
//       cudaFree(devCentInfo[gpuIter]);
//       cudaFree(devCentData[gpuIter]);
//       cudaFree(devMaxDriftArr[gpuIter]);
//       cudaFree(devNewCentSum[gpuIter]);
//       cudaFree(devOldCentSum[gpuIter]);
//       cudaFree(devNewCentCount[gpuIter]);
//       cudaFree(devOldCentCount[gpuIter]);
//       cudaFree(devConFlagArr[gpuIter]);
//   }
//
//   free(allCentInfo);
//   free(allCentData);
//   free(newCentInfo);
//   free(newCentData);
//   free(oldCentData);
//   free(pointLwrs);
//
//   return 0.0;
//}
//
//DTYPE* storeDataOnGPU(DTYPE* data,
//   const int numVec,
//   const int numFeat)
//{
//   DTYPE* devData = NULL;
//   gpuErrchk(cudaMalloc(&devData, sizeof(DTYPE) * numVec * numFeat));
//   gpuErrchk(cudaMemcpy(devData, data,
//       sizeof(DTYPE) * numVec * numFeat, cudaMemcpyHostToDevice));
//   return devData;
//}
//
//PointInfo* storePointInfoOnGPU(PointInfo* pointInfo,
//   const int numPnt)
//{
//   PointInfo* devPointInfo = NULL;
//   gpuErrchk(cudaMalloc(&devPointInfo, sizeof(PointInfo) * numPnt));
//   gpuErrchk(cudaMemcpy(devPointInfo, pointInfo,
//       sizeof(PointInfo) * numPnt, cudaMemcpyHostToDevice));
//   return devPointInfo;
//}
//
//CentInfo* storeCentInfoOnGPU(CentInfo* centInfo,
//   const int numCent)
//{
//   CentInfo* devCentInfo = NULL;
//   gpuErrchk(cudaMalloc(&devCentInfo, sizeof(CentInfo) * numCent));
//   gpuErrchk(cudaMemcpy(devCentInfo, centInfo,
//       sizeof(CentInfo) * numCent, cudaMemcpyHostToDevice));
//   return devCentInfo;
//}
//
//void warmupGPU(const int numGPU)
//{
//   int gpuIter;
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaSetDevice(gpuIter);
//       cudaDeviceSynchronize();
//   }
//}
//
//DTYPE calcDisCPU(DTYPE* vec1,
//   DTYPE* vec2,
//   const int numDim)
//{
//   unsigned int index;
//   DTYPE total = 0.0;
//   DTYPE square;
//
//   for (index = 0; index < numDim; index++)
//   {
//       square = (vec1[index] - vec2[index]);
//       total += square * square;
//   }
//
//   return sqrt(total);
//
//}
//
//unsigned int checkConverge(PointInfo* pointInfo,
//   const int numPnt)
//{
//   for (int index = 0; index < numPnt; index++)
//   {
//       if (pointInfo[index].centroidIndex != pointInfo[index].oldCentroid)
//           return 0;
//   }
//   return 1;
//}
//
//double startLloydOnCPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numDim,
//   const int numThread,
//   const int maxIter,
//   unsigned int* ranIter)
//{
//
//
//   unsigned int pntIndex, centIndex, dimIndex;
//   unsigned int index = 0;
//   unsigned int conFlag = 0;
//
//   DTYPE* oldVecs = (DTYPE*)malloc(sizeof(DTYPE) * numDim * numCent);
//
//
//
//   DTYPE currMin, currDis;
//
//   // start standard kmeans algorithm for MAXITER iterations
//   while (!conFlag && index < maxIter)
//   {
//       currMin = INFINITY;
//
//#pragma omp parallel \
//   private(pntIndex, centIndex, currDis, currMin) \
//   shared(pointInfo, centInfo, pointData, centData)
//       {
//#pragma omp for nowait schedule(static)
//           for (pntIndex = 0; pntIndex < numPnt; pntIndex++)
//           {
//               pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;
//               for (centIndex = 0; centIndex < numCent; centIndex++)
//               {
//                   currDis = calcDisCPU(&pointData[pntIndex * numDim],
//                       &centData[centIndex * numDim],
//                       numDim);
//                   if (currDis < currMin)
//                   {
//                       pointInfo[pntIndex].centroidIndex = centIndex;
//                       currMin = currDis;
//                   }
//               }
//               currMin = INFINITY;
//           }
//       }
//
//       // clear centroids features
//       for (centIndex = 0; centIndex < numCent; centIndex++)
//       {
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
//               centData[(centIndex * numDim) + dimIndex] = 0.0;
//           }
//           centInfo[centIndex].count = 0;
//       }
//       // sum all assigned point's features
//#pragma omp parallel \
//   private(pntIndex, dimIndex) \
//   shared(pointInfo, centInfo, pointData, centData)
//       {
//#pragma omp for nowait schedule(static)
//           for (pntIndex = 0; pntIndex < numPnt; pntIndex++)
//           {
//#pragma omp atomic
//               centInfo[pointInfo[pntIndex].centroidIndex].count++;
//
//               for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//               {
//#pragma omp atomic
//                   centData[(pointInfo[pntIndex].centroidIndex * numDim) + dimIndex] += pointData[(pntIndex * numDim) + dimIndex];
//               }
//           }
//       }
//       // take the average of each feature to get new centroid features
//       for (centIndex = 0; centIndex < numCent; centIndex++)
//       {
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               if (centInfo[centIndex].count > 0)
//                   centData[(centIndex * numDim) + dimIndex] /= centInfo[centIndex].count;
//               else
//                   centData[(centIndex * numDim) + dimIndex] = oldVecs[(centIndex * numDim) + dimIndex];
//           }
//       }
//       index++;
//       conFlag = checkConverge(pointInfo, numPnt);
//   }
//
//   *ranIter = index;
//
//
//   free(oldVecs);
//
//   return 0.0;
//}
//
//double warumGehtNichts() {
//    std::cout << "WARUM GEHT NICHTS" << std::endl;
//    int* devTest;
//    int* test;
//    std::cout << "a" << std::endl;
//    test = new int[10000];
//    gpuErrchk(cudaMalloc(&devTest, sizeof(int) * 10000));
//    std::cout << "b" << std::endl;
//    for (int i = 0; i < 10000; i++)
//        test[i] = i % 100;
//    std::cout << "c" << std::endl;
//    gpuErrchk(cudaMemcpy(devTest,
//        test,
//        10000 * sizeof(int),
//        cudaMemcpyHostToDevice));
//    std::cout << "d" << std::endl;
//    std::cout << "call test" << std::endl;
//    std::cout << "vorher: " << test[2] << std::endl;
//    setTestL << <1, 256 >> > (devTest);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaMemcpy(test,
//        devTest,
//        10000 * sizeof(int),
//        cudaMemcpyDeviceToHost));
//    cudaDeviceSynchronize();
//    std::cout << "Nachher: " << test[2] << std::endl;
//    for (int i = 0; i < 10000; i++) {
//        // std::cout << test[i] << std::endl;
//    }
//    std::cout << "e" << std::endl;
//    cudaFree(devTest);
//    free(test);
//    return 0.0;
//}
//double startLloydOnGPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numDim,
//   const int maxIter,
//   const int numGPUU,
//   unsigned int* ranIter)
//{
//  /* int* devTest;
//   int* test;
//
//   test = new int[10000];
//   gpuErrchk(cudaMalloc(&devTest, sizeof(int) * 10000));
//   for (int i = 0; i < 10000; i++)
//       test[i] = i % 100;
//
//   gpuErrchk(cudaMemcpy(devTest,
//       test,
//       10000 * sizeof(int),
//       cudaMemcpyHostToDevice));
//
//   std::cout << "call test" << std::endl;
//   setTestL << <1, 256 >> > (devTest);*/
//   
//   //}
//   //cudaDeviceSynchronize();
//   //gpuErrchk(cudaMemcpy(test,
//   //    devTest,
//   //    10000 * sizeof(int),
//   //    cudaMemcpyDeviceToHost));
//   //cudaDeviceSynchronize();
//   //std::cout << test[2]<<  std::endl;
//   //for (int i = 0; i < 10000; i++) {
//   //    // std::cout << test[i] << std::endl;
//   //}
//
//   // variable initialization
//   int gpuIter;
//   const int numGPU = 1;
//
//   int numPnts[numGPU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//       {
//           numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//       }
//
//       else
//       {
//           numPnts[gpuIter] = numPnt / numGPU;
//       }
//   }
//
//   unsigned int hostConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagArr[gpuIter] = 1;
//   }
//
//   unsigned int* hostConFlagPtrArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//   }
//
//   int index = 1;
//
//   unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//   // store dataset on device
//   PointInfo* devPointInfo[numGPU];
//   DTYPE* devPointData[numGPU];
//
//      
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//      // std::cout << "testfweatgwetwetwet" << numPnts[gpuIter] << std::endl;
//       cudaSetDevice(gpuIter);
//
//       // alloc dataset to GPU
//       gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//           pointInfo + (gpuIter * numPnt / numGPU),
//           (numPnts[gpuIter]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//       gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//           pointData + ((gpuIter * numPnt / numGPU) * numDim),
//           sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//           cudaMemcpyHostToDevice));
//   }
//
//   // store centroids on device
//   CentInfo* devCentInfo[numGPU];
//   DTYPE* devCentData[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // alloc dataset and drift array to GPU
//       gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//           centInfo, sizeof(CentInfo) * numCent,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//       gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//           centData, sizeof(DTYPE) * numCent * numDim,
//           cudaMemcpyHostToDevice));
//   }
//
//   // centroid calculation data
//   DTYPE* devNewCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   unsigned int* devNewCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//       gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//           hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//           cudaMemcpyHostToDevice));
//   }
//
////#pragma omp parallel for num_threads(numGPU)
////        for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
////        {
////            gpuErrchk(cudaSetDevice(gpuIter));
//       //std::cout << "call clearcentcalcdata lloyd" << NBLOCKS << std::endl;
//       //setTestL << <1950, 256 >> > (devTest);
//       /* clearCentCalcDataLloyd2 << <1950, 256 >> > (devNewCentSum[gpuIter],
//           devNewCentCount[gpuIter],
//           numCent,
//           numDim, devTest);*/
//        clearCentCalcDataLloyd << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//           devNewCentCount[gpuIter],
//           numCent,
//           numDim);
//
//   //}
//   
//   CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//   }
//
//   DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   }
//
//   CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//   DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   for (int i = 0; i < numCent; i++)
//   {
//       for (int j = 0; j < numDim; j++)
//       {
//           newCentData[(i * numDim) + j] = 0;
//       }
//   }
//
//   DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//   unsigned int doesNotConverge = 1;
//
//   // loop until convergence
//   while (doesNotConverge && index < maxIter)
//   {
//       doesNotConverge = 0;
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           hostConFlagArr[gpuIter] = 0;
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//               hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyHostToDevice));
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           assignPointsLloyd << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devCentInfo[gpuIter],
//               devPointData[gpuIter],
//               devCentData[gpuIter],
//               numPnts[gpuIter],
//               numCent,
//               numDim);
//
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           clearCentCalcDataLloyd << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//               devNewCentCount[gpuIter],
//               numCent, numDim);
//
//       }
//
//       // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcCentDataLloyd << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devPointData[gpuIter],
//               devNewCentSum[gpuIter],
//               devNewCentCount[gpuIter],
//               numPnts[gpuIter], numDim);
//
//       }
//
//       // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcNewCentroidsLloyd << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devCentInfo[gpuIter],
//               devCentData[gpuIter],
//               devNewCentSum[gpuIter],
//               devNewCentCount[gpuIter],
//               numCent, numDim);
//
//       }
//
//       if (numGPU > 1)
//       {
//#pragma omp parallel for num_threads(numGPU)
//           for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//           {
//               gpuErrchk(cudaSetDevice(gpuIter));
//               gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
//                   devCentInfo[gpuIter], sizeof(CentInfo) * numCent,
//                   cudaMemcpyDeviceToHost));
//           }
//
//#pragma omp parallel for num_threads(numGPU)
//           for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//           {
//               gpuErrchk(cudaSetDevice(gpuIter));
//               gpuErrchk(cudaMemcpy(allCentData[gpuIter],
//                   devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim,
//                   cudaMemcpyDeviceToHost));
//           }
//
//           for (int i = 0; i < numCent; i++)
//           {
//               newCentInfo[i].count = 0;
//           }
//
//           calcWeightedMeansLloyd(newCentInfo, allCentInfo, newCentData, oldCentData,
//               allCentData, numCent, numDim, numGPU);
//
//#pragma omp parallel for num_threads(numGPU)
//           for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//           {
//               gpuErrchk(cudaSetDevice(gpuIter));
//
//               // copy input data to GPU
//               gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//                   newCentInfo, sizeof(cent) * numCent,
//                   cudaMemcpyHostToDevice));
//           }
//
//#pragma omp parallel for num_threads(numGPU)
//           for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//           {
//               gpuErrchk(cudaSetDevice(gpuIter));
//
//               // copy input data to GPU
//               gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//                   newCentData, sizeof(DTYPE) * numCent * numDim,
//                   cudaMemcpyHostToDevice));
//           }
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devConFlagArr[gpuIter],
//               numPnts[gpuIter]);
//       }
//
//       index++;
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//               devConFlagArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyDeviceToHost));
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           if (hostConFlagArr[gpuIter])
//           {
//               doesNotConverge = 1;
//           }
//       }
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaDeviceSynchronize();
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // copy finished clusters and points from device to host
//       gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//           devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//   }
//
//   // and the final centroid positions
//   gpuErrchk(cudaMemcpy(centData, devCentData[0],
//       sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//   *ranIter = index;
//
//   // clean up, return
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaFree(devPointInfo[gpuIter]);
//       cudaFree(devPointData[gpuIter]);
//       cudaFree(devCentInfo[gpuIter]);
//       cudaFree(devCentData[gpuIter]);
//       cudaFree(devNewCentSum[gpuIter]);
//       cudaFree(devNewCentCount[gpuIter]);
//       cudaFree(devConFlagArr[gpuIter]);
//   }
//
//   free(allCentInfo);
//   free(allCentData);
//   free(newCentInfo);
//   free(newCentData);
//   free(oldCentData);
//
//   return 0.0;
//}
//
//double startSuperOnGPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numDim,
//   const int maxIter,
//   const int numGPUU,
//   unsigned int* ranIter)
//{
//
//   // variable initialization
//   int gpuIter;
//   const int numGPU = 1;
//   int numPnts[numGPU];
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       if (numPnt % numGPU != 0 && gpuIter == numGPU - 1)
//       {
//           numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
//       }
//
//       else
//       {
//           numPnts[gpuIter] = numPnt / numGPU;
//       }
//   }
//
//   unsigned int hostConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagArr[gpuIter] = 1;
//   }
//
//   unsigned int* hostConFlagPtrArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
//   }
//
//   int index = 1;
//
//   unsigned int NBLOCKS = ceil(numPnt * 1.0 / BLOCKSIZE * 1.0);
//
//   // group centroids
//   for (int j = 0; j < numCent; j++)
//   {
//       centInfo[j].groupNum = 0;
//   }
//
//   // create lower bound data on host
//   DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt);
//   for (int i = 0; i < numPnt; i++)
//   {
//       pointLwrs[i] = INFINITY;
//   }
//
//   // store dataset on device
//   PointInfo* devPointInfo[numGPU];
//   DTYPE* devPointData[numGPU];
//   DTYPE* devPointLwrs[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaSetDevice(gpuIter);
//
//       // alloc dataset to GPU
//       gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo) * (numPnts[gpuIter])));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
//           pointInfo + (gpuIter * numPnt / numGPU),
//           (numPnts[gpuIter]) * sizeof(PointInfo),
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));
//
//       gpuErrchk(cudaMemcpy(devPointData[gpuIter],
//           pointData + ((gpuIter * numPnt / numGPU) * numDim),
//           sizeof(DTYPE) * numPnts[gpuIter] * numDim,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter]));
//
//       gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
//           pointLwrs + ((gpuIter * numPnt / numGPU)),
//           sizeof(DTYPE) * numPnts[gpuIter],
//           cudaMemcpyHostToDevice));
//   }
//
//   // store centroids on device
//   CentInfo* devCentInfo[numGPU];
//   DTYPE* devCentData[numGPU];
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // alloc dataset and drift array to GPU
//       gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo) * numCent));
//
//       // copy input data to GPU
//       gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
//           centInfo, sizeof(CentInfo) * numCent,
//           cudaMemcpyHostToDevice));
//
//       gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE) * numCent * numDim));
//       gpuErrchk(cudaMemcpy(devCentData[gpuIter],
//           centData, sizeof(DTYPE) * numCent * numDim,
//           cudaMemcpyHostToDevice));
//   }
//
//   DTYPE* devMaxDrift[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devMaxDrift[gpuIter], sizeof(DTYPE));
//   }
//
//   // centroid calculation data
//   DTYPE* devNewCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   DTYPE* devOldCentSum[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   DTYPE* devOldCentData[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numCent * numDim);
//   }
//
//   unsigned int* devNewCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devOldCentCount[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
//   }
//
//   unsigned int* devConFlagArr[numGPU];
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
//       gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//           hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//           cudaMemcpyHostToDevice));
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearCentCalcData << <NBLOCKS, BLOCKSIZE >> > (devNewCentSum[gpuIter],
//           devOldCentSum[gpuIter],
//           devNewCentCount[gpuIter],
//           devOldCentCount[gpuIter],
//           numCent,
//           numDim);
//
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDrift[gpuIter], 1);
//   }
//
//   // do single run of naive kmeans for initial centroid assignments
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       initRunKernel << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//           devCentInfo[gpuIter],
//           devPointData[gpuIter],
//           devPointLwrs[gpuIter],
//           devCentData[gpuIter],
//           numPnts[gpuIter],
//           numCent,
//           1,
//           numDim);
//   }
//
//
//   CentInfo** allCentInfo = (CentInfo**)malloc(sizeof(CentInfo*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentInfo[gpuIter] = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//   }
//
//   DTYPE** allCentData = (DTYPE**)malloc(sizeof(DTYPE*) * numGPU);
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       allCentData[gpuIter] = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   }
//
//   CentInfo* newCentInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
//
//   DTYPE* newCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   for (int i = 0; i < numCent; i++)
//   {
//       for (int j = 0; j < numDim; j++)
//       {
//           newCentData[(i * numDim) + j] = 0;
//       }
//   }
//
//   DTYPE* oldCentData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//
//   DTYPE* newMaxDriftArr;
//   newMaxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * 1);
//   for (int i = 0; i < 1; i++)
//   {
//       newMaxDriftArr[i] = 0.0;
//   }   
//
//   unsigned int doesNotConverge = 1;
//
//   // loop until convergence
//   while (doesNotConverge && index < maxIter)
//   {
//       doesNotConverge = 0;
//
//       for (int i = 0; i < numCent; i++)
//       {
//           newCentInfo[i].count = 0;
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           hostConFlagArr[gpuIter] = 0;
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
//               hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyHostToDevice));
//       }
//
//       // clear maintained data on device
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           clearDriftArr << <NBLOCKS, BLOCKSIZE >> > (devMaxDrift[gpuIter], 1);
//
//       }
//
//       // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devPointData[gpuIter], devOldCentSum[gpuIter],
//               devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//
//       }
//
//       // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//               devCentData[gpuIter], devOldCentData[gpuIter],
//               devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//               devMaxDrift[gpuIter],
//               devOldCentCount[gpuIter],
//               devNewCentCount[gpuIter],
//               numCent, numDim);
//
//       }
//
//      
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           assignPointsSuper << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devCentInfo[gpuIter],
//               devPointData[gpuIter],
//               devPointLwrs[gpuIter],
//               devCentData[gpuIter],
//               devMaxDrift[gpuIter],
//               numPnts[gpuIter], numCent,
//               1, numDim);
//
//       }
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           checkConverge << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter],
//               devConFlagArr[gpuIter],
//               numPnts[gpuIter]);
//
//       }
//
//       index++;
//
//#pragma omp parallel for num_threads(numGPU)
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           gpuErrchk(cudaSetDevice(gpuIter));
//           gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
//               devConFlagArr[gpuIter], sizeof(unsigned int),
//               cudaMemcpyDeviceToHost));
//       }
//
//       for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//       {
//           if (hostConFlagArr[gpuIter])
//           {
//               doesNotConverge = 1;
//           }
//       }
//   }
//
//   // calculate data necessary to make new centroids
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcCentData << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devPointData[gpuIter], devOldCentSum[gpuIter],
//           devNewCentSum[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numPnts[gpuIter], numDim);
//   }
//
//   // make new centroids
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//       calcNewCentroids << <NBLOCKS, BLOCKSIZE >> > (devPointInfo[gpuIter], devCentInfo[gpuIter],
//           devCentData[gpuIter], devOldCentData[gpuIter],
//           devOldCentSum[gpuIter], devNewCentSum[gpuIter],
//           devMaxDrift[gpuIter], devOldCentCount[gpuIter],
//           devNewCentCount[gpuIter], numCent, numDim);
//   }   
//
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaSetDevice(gpuIter);
//       cudaDeviceSynchronize();
//   }
//
//#pragma omp parallel for num_threads(numGPU)
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       gpuErrchk(cudaSetDevice(gpuIter));
//
//       // copy finished clusters and points from device to host
//       gpuErrchk(cudaMemcpy(pointInfo + ((gpuIter * numPnt / numGPU)),
//           devPointInfo[gpuIter], sizeof(PointInfo) * numPnts[gpuIter], cudaMemcpyDeviceToHost));
//   }
//
//   // and the final centroid positions
//   gpuErrchk(cudaMemcpy(centData, devCentData[0],
//       sizeof(DTYPE) * numCent * numDim, cudaMemcpyDeviceToHost));
//
//   *ranIter = index;
//
//   // clean up, return
//   for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
//   {
//       cudaFree(devPointInfo[gpuIter]);
//       cudaFree(devPointData[gpuIter]);
//       cudaFree(devPointLwrs[gpuIter]);
//       cudaFree(devCentInfo[gpuIter]);
//       cudaFree(devCentData[gpuIter]);
//       cudaFree(devMaxDrift[gpuIter]);
//       cudaFree(devNewCentSum[gpuIter]);
//       cudaFree(devOldCentSum[gpuIter]);
//       cudaFree(devNewCentCount[gpuIter]);
//       cudaFree(devOldCentCount[gpuIter]);
//       cudaFree(devConFlagArr[gpuIter]);
//   }
//
//   free(allCentInfo);
//   free(allCentData);
//   free(newCentInfo);
//   free(newCentData);
//   free(oldCentData);
//   free(pointLwrs);
//
//   return 0.0;
//}
//
//void initPoints(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* pointLwrs,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim,
//   const int numThread)
//{
//   unsigned int pntIndex, centIndex;
//
//   DTYPE currDistance;
//
//   // start single standard k-means iteration for initial bounds and cluster assignments
//     // assignment
//#pragma omp parallel \
// private(pntIndex, centIndex, currDistance) \
// shared(pointInfo, centInfo, pointData, pointLwrs, centData)
//   {
//#pragma omp for schedule(static)
//       for (pntIndex = 0; pntIndex < numPnt; pntIndex++)
//       {
//           pointInfo[pntIndex].uprBound = INFINITY;
//
//           // for all centroids
//           for (centIndex = 0; centIndex < numCent; centIndex++)
//           {
//               // currDistance is equal to the distance between the current feature
//               // vector being inspected, and the current centroid being compared
//               currDistance = calcDisCPU(&pointData[pntIndex * numDim],
//                   &centData[centIndex * numDim],
//                   numDim);
//
//               // if the the currDistance is less than the current minimum distance
//               if (currDistance < pointInfo[pntIndex].uprBound)
//               {
//                   if (pointInfo[pntIndex].uprBound != INFINITY)
//                       pointLwrs[(pntIndex * numGrp) +
//                       centInfo[pointInfo[pntIndex].centroidIndex].groupNum] =
//                       pointInfo[pntIndex].uprBound;
//                   // update assignment and upper bound
//                   pointInfo[pntIndex].centroidIndex = centIndex;
//                   pointInfo[pntIndex].uprBound = currDistance;
//               }
//               else if (currDistance < pointLwrs[(pntIndex * numGrp) + centInfo[centIndex].groupNum])
//               {
//                   pointLwrs[(pntIndex * numGrp) + centInfo[centIndex].groupNum] = currDistance;
//               }
//           }
//       }
//   }
//}
//
//void updateCentroids(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   DTYPE* maxDriftArr,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim,
//   const int numThread)
//{
//   unsigned int pntIndex, centIndex, grpIndex, dimIndex;
//
//   DTYPE compDrift;
//
//   // holds the number of points assigned to each centroid formerly and currently
//   unsigned int* oldCounts = (unsigned int*)malloc(sizeof(unsigned int) * numCent);
//   unsigned int* newCounts = (unsigned int*)malloc(sizeof(unsigned int) * numCent);
//
//   // holds the new vector calculated
//   DTYPE* oldVecs = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   DTYPE oldCentFeat;
//
//
//   omp_set_num_threads(numThread);
//
//   omp_lock_t driftLock;
//   omp_init_lock(&driftLock);
//
//   // allocate data for new and old vector sums
//   DTYPE* oldSums = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   DTYPE* newSums = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);
//   DTYPE oldSumFeat;
//   DTYPE newSumFeat;
//
//   for (centIndex = 0; centIndex < numCent; centIndex++)
//   {
//       for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//       {
//           oldSums[(centIndex * numDim) + dimIndex] = 0.0;
//           newSums[(centIndex * numDim) + dimIndex] = 0.0;
//       }
//       oldCounts[centIndex] = 0;
//       newCounts[centIndex] = 0;
//   }
//   for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//   {
//       maxDriftArr[grpIndex] = 0.0;
//   }
//
//   for (pntIndex = 0; pntIndex < numPnt; pntIndex++)
//   {
//       // add one to the old count and new count for each centroid
//       if (pointInfo[pntIndex].oldCentroid >= 0)
//           oldCounts[pointInfo[pntIndex].oldCentroid]++;
//
//       newCounts[pointInfo[pntIndex].centroidIndex]++;
//
//       // if the old centroid does not match the new centroid,
//       // add the points vector to each 
//       if (pointInfo[pntIndex].oldCentroid != pointInfo[pntIndex].centroidIndex)
//       {
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               if (pointInfo[pntIndex].oldCentroid >= 0)
//               {
//                   oldSums[(pointInfo[pntIndex].oldCentroid * numDim) + dimIndex] +=
//                       pointData[(pntIndex * numDim) + dimIndex];
//               }
//               newSums[(pointInfo[pntIndex].centroidIndex * numDim) + dimIndex] +=
//                   pointData[(pntIndex * numDim) + dimIndex];
//           }
//       }
//   }
//
//
//
//#pragma omp parallel \
// private(centIndex, dimIndex, oldCentFeat, oldSumFeat, newSumFeat, compDrift) \
// shared(driftLock, centInfo, centData, maxDriftArr, oldVecs)
//   {
//       // create new centroid points
//#pragma omp for schedule(static)
//       for (centIndex = 0; centIndex < numCent; centIndex++)
//       {
//           for (dimIndex = 0; dimIndex < numDim; dimIndex++)
//           {
//               if (newCounts[centIndex] > 0)
//               {
//                   oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
//                   oldCentFeat = oldVecs[(centIndex * numDim) + dimIndex];
//                   oldSumFeat = oldSums[(centIndex * numDim) + dimIndex];
//                   newSumFeat = newSums[(centIndex * numDim) + dimIndex];
//
//                   centData[(centIndex * numDim) + dimIndex] =
//                       (oldCentFeat * oldCounts[centIndex] - oldSumFeat + newSumFeat)
//                       / newCounts[centIndex];
//                   //printf("(%f * %d - %f + %f) / %d\n", oldCentFeat,oldCounts[centIndex],oldSumFeat,newSumFeat,newCounts[centIndex]);
//
//               }
//               else
//               {
//                   // if the centroid has no current members, no change occurs to its position
//                   oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
//               }
//           }
//           compDrift = calcDisCPU(&oldVecs[centIndex * numDim],
//               &centData[centIndex * numDim], numDim);
//           omp_set_lock(&driftLock);
//           // printf("%d\n",centInfo[centIndex].groupNum);
//           if (compDrift > maxDriftArr[centInfo[centIndex].groupNum])
//           {
//               maxDriftArr[centInfo[centIndex].groupNum] = compDrift;
//           }
//           omp_unset_lock(&driftLock);
//           centInfo[centIndex].drift = compDrift;
//       }
//   }
//   omp_destroy_lock(&driftLock);
//
//   free(oldCounts);
//   free(newCounts);
//   free(oldVecs);
//   free(oldSums);
//   free(newSums);
//
//}
//
//void pointCalcsSimpleCPU(PointInfo* pointInfoPtr,
//   CentInfo* centInfo,
//   DTYPE* pointDataPtr,
//   DTYPE* pointLwrPtr,
//   DTYPE* centData,
//   DTYPE* maxDriftArr,
//   unsigned int* groupArr,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim)
//{
//   // index variables
//   unsigned int centIndex, grpIndex;
//
//   DTYPE compDistance;
//
//   for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//   {
//       // if the group is not blocked by group filter
//       if (groupArr[grpIndex])
//       {
//           // reset the lwrBoundArr to be only new lwrBounds
//           pointLwrPtr[grpIndex] = INFINITY;
//       }
//   }
//
//   for (centIndex = 0; centIndex < numCent; centIndex++)
//   {
//       // if the centroid's group is marked in groupArr
//       if (groupArr[centInfo[centIndex].groupNum])
//       {
//           // if it was the originally assigned cluster, no need to calc dist
//           if (centIndex == pointInfoPtr->oldCentroid)
//               continue;
//
//           // compute distance between point and centroid
//           compDistance = calcDisCPU(pointDataPtr, &centData[centIndex * numDim], numDim);
//
//           if (compDistance < pointInfoPtr->uprBound)
//           {
//               pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] = pointInfoPtr->uprBound;
//               pointInfoPtr->centroidIndex = centIndex;
//               pointInfoPtr->uprBound = compDistance;
//           }
//           else if (compDistance < pointLwrPtr[centInfo[centIndex].groupNum])
//           {
//               pointLwrPtr[centInfo[centIndex].groupNum] = compDistance;
//           }
//       }
//   }
//}
//
//double startSimpleOnCPU(PointInfo* pointInfo,
//   CentInfo* centInfo,
//   DTYPE* pointData,
//   DTYPE* centData,
//   const int numPnt,
//   const int numCent,
//   const int numGrp,
//   const int numDim,
//   const int numThread,
//   const int maxIter,
//   unsigned int* ranIter)
//{
//   // index variables
//   unsigned int pntIndex, grpIndex;
//   unsigned int index = 1;
//   unsigned int conFlag = 0;
//
//   // array to contain the maximum drift of each group of centroids
//   // note: shared amongst all points
//   DTYPE* maxDriftArr = (DTYPE*)malloc(sizeof(DTYPE) * numGrp);
//
//   // array of all the points lower bounds
//   DTYPE* pointLwrs = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numGrp);
//
//   // initiatilize to INFINITY
//   for (grpIndex = 0; grpIndex < numPnt * numGrp; grpIndex++)
//   {
//       pointLwrs[grpIndex] = INFINITY;
//   }
//
//   // array to contain integer flags which mark which groups need to be checked
//   // for a potential new centroid
//   // note: unique to each point
//   unsigned int* groupLclArr = (unsigned int*)malloc(sizeof(unsigned int) * numPnt * numGrp);
//
//   omp_set_num_threads(numThread);
//
//   // the minimum of all the lower bounds for a single point
//   DTYPE tmpGlobLwr = INFINITY;
//
//   // cluster the centroids into NGROUPCPU groups
//   groupCent(centInfo, centData, numCent, numGrp, numDim);
//
//   // run one iteration of standard kmeans for initial centroid assignments
//   initPoints(pointInfo, centInfo, pointData, pointLwrs,
//       centData, numPnt, numCent, numGrp, numDim, numThread);
//   // master loop
//   while (!conFlag && index < maxIter)
//   {
//       // clear drift array each new iteration
//       for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//       {
//           maxDriftArr[grpIndex] = 0.0;
//       }
//       // update centers via optimised update method
//       updateCentroids(pointInfo, centInfo, pointData, centData,
//           maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);
//       // filtering done in parallel
//#pragma omp parallel \
//   private(pntIndex, grpIndex, tmpGlobLwr) \
//   shared(pointInfo, centInfo, pointData, centData, maxDriftArr, groupLclArr)
//       {
//#pragma omp for schedule(static)
//           for (pntIndex = 0; pntIndex < numPnt; pntIndex++)
//           {
//               // reset old centroid before possibly finding a new one
//               pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;
//
//               tmpGlobLwr = INFINITY;
//
//               // update upper bound
//                   // ub = ub + centroid's drift
//               pointInfo[pntIndex].uprBound +=
//                   centInfo[pointInfo[pntIndex].centroidIndex].drift;
//
//               // update group lower bounds
//                   // lb = lb - maxGroupDrift
//               for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//               {
//                   pointLwrs[(pntIndex * numGrp) + grpIndex] -= maxDriftArr[grpIndex];
//
//                   if (pointLwrs[(pntIndex * numGrp) + grpIndex] < tmpGlobLwr)
//                   {
//                       // minimum lower bound
//                       tmpGlobLwr = pointLwrs[(pntIndex * numGrp) + grpIndex];
//                   }
//               }
//
//               // global filtering
//               // if global lowerbound >= upper bound
//               if (tmpGlobLwr < pointInfo[pntIndex].uprBound)
//               {
//                   // tighten upperbound ub = d(x, b(x))
//                   pointInfo[pntIndex].uprBound =
//                       calcDisCPU(&pointData[pntIndex * numDim],
//                           &centData[pointInfo[pntIndex].centroidIndex * numDim],
//                           numDim);
//                   // check condition again
//                   if (tmpGlobLwr < pointInfo[pntIndex].uprBound)
//                   {
//                       // group filtering
//                       for (grpIndex = 0; grpIndex < numGrp; grpIndex++)
//                       {
//                           // mark groups that need to be checked
//                           if (pointLwrs[(pntIndex * numGrp) + grpIndex] < pointInfo[pntIndex].uprBound)
//                               groupLclArr[(pntIndex * numGrp) + grpIndex] = 1;
//                           else
//                               groupLclArr[(pntIndex * numGrp) + grpIndex] = 0;
//                       }
//
//                       // pass group array and point to go execute distance calculations
//                       pointCalcsSimpleCPU(&pointInfo[pntIndex], centInfo,
//                           &pointData[pntIndex * numDim], &pointLwrs[pntIndex * numGrp],
//                           centData, maxDriftArr, &groupLclArr[pntIndex * numGrp],
//                           numPnt, numCent, numGrp, numDim);
//                   }
//               }
//           }
//       }
//       index++;
//       conFlag = checkConverge(pointInfo, numPnt);
//   }
//   updateCentroids(pointInfo, centInfo, pointData, centData,
//       maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);
//
//   *ranIter = index;
//
//   return 0.0;
//}
