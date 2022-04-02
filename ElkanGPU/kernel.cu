
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dataset.h"
//#include "gpufunctions.h"
#include <cublas_v2.h>
#include <random>
#include <fstream>
#include <string>

#include "general_functions.h"
#include "kmeans.h"
#include "elkan_kmean.h"

//#include "yy_kmean.h"
//#include "FB1_elkan_kmeans.h"
//#include "MO_elkan_kmeans.h"

#include <fstream>
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    printf("add: %i\n", a[i]);
    c[i] = a[i] + b[i];
}

__global__ void multiplyKernel(float* a, float* b, float* c) {
    int i = threadIdx.x;
    //printf("ahhh");
    if (i < 5) {
        c[i] = a[i] * b[i];
    }
}

__global__ void setTestt(int* test, unsigned short* arr1, unsigned short* arr2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000) {
        arr2[i] = arr1[i];
        test[i] = 5;
    }
}

Dataset* load_dataset(std::string const& filename) {
    std::ifstream input(filename.c_str());

    int n, d;
    input >> n >> d;

    Dataset* x = new Dataset(n, d);

    double* dataTMP = new double[n * d];


    //double* copyP1 = x->data;
    for (int i = 0; i < n * d; ++i) {
        input >> x->data[i];
        //copyP1++;
    }
    return x;
}

//int importPoints(
//    PointInfo* pointInfo,
//    DTYPE* pointData,
//    const int numPnt,
//    const int numDim)
//{
//    std::string number;
//    fstream file;
//    file.open("file.txt", ios::in);
//    int i = 0;
//    while (getline(file, number, ','))
//    {
//        pointData[i] = atof(number.c_str());
//        i++;
//        if (i == 4992000)
//            break;
//    }
//
//    for (int j = 0; j < numPnt; j++) {
//        pointInfo[j].centroidIndex = -1;
//        pointInfo[j].oldCentroid = -1;
//        pointInfo[j].uprBound = INFINITY;
//    }
//    return 0;
//}

Dataset* load_randDataset(int n, int d) {
    bool createNew = false;
    std::string number;
    Dataset* x = nullptr;
    fstream file;

    if (!createNew) {
        file.open("file.txt", ios::in);
        x = new Dataset(n, d);
        int i = 0;
        while (getline(file, number, ','))
        {
            x->data[i] = atof(number.c_str());
            i++;
        }
        std::cout << "GELESEN: " << i << std::endl;
    }
    else {
        file.open("file.txt", ios::out | ios::trunc);
        file << (double)rand() / (double)RAND_MAX;

        for (int i = 1; i < n * d; ++i) {
            file << ",";
            file << (double)rand() / (double)RAND_MAX;
            // x->data[i] = dis(gen);
            //x->data[i] = (double)rand() / (double)RAND_MAX;
        }
    }
    file.close();
    return x;
}

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.1, 100.0);

int main(){
    cudaSetDevice(0);   
    //int num = 5;
    //float* data1 = new float[num];
    //float* d_data1;
    //cudaMalloc(&d_data1, num * sizeof(float));
    //float* data2 = new float[num];
    //float* d_data2;
    //cudaMalloc(&d_data2, num * sizeof(float));
    //float* data3 = new float[num];
    //float* d_data3;
    //cudaMalloc(&d_data3, num * sizeof(float));

    //for (int i = 0; i < 5; i++)
    //    data1[i] = i;

    //for (int i = 0; i < 5; i++)
    //    data2[i] = 5 - i;
    //data2[2] = 2;
    //data2[3] = 3;

    //cudaMemcpy(d_data1, data1, num * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_data2, data2, num * sizeof(float), cudaMemcpyHostToDevice);
    //multiplyKernel << <1, 5 >> > (d_data1, d_data2, d_data3);
    //cudaDeviceSynchronize();
    //cudaMemcpy(data3, d_data3, num * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //for (int i = 0; i < 5; i++) {
    //    std::cout << "Nicht-Managed:" << data1[i] << " * " << data2[i] << " = " << data3[i] << std::endl;
    //    //std::cout << arr1[i] << " dot " << arr2[i] << std::endl;
    //}
    //cudaFree(d_data1);
    //cudaFree(d_data2);
    //cudaFree(d_data3);
    //delete data1;
    //delete data2;
    //delete data3;
    //warumGehtNichts();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int k = 10;
    ElkanKmeans* alg = new ElkanKmeans();
    //FB1_ElkanKmeans* alg = new FB1_ElkanKmeans();
    //MO_ElkanKmeans* alg = new MO_ElkanKmeans();
    std::cout << "Alg: " << alg->getName() << std::endl;
    //Dataset* x = load_dataset("C:\\Users\\Admin\\Desktop\\MASTER\\skin_nonskin.txt");
    Dataset* x = load_randDataset(499200, 10);
    if (x == nullptr) {
        cout << "Dataset generated" << endl;
        return 0;
    }
    cout << "Dataset loaded" << endl;
    Dataset* initialCenters = init_centers(*x, k);
    unsigned short* assignment = new unsigned short[x->n];
    unsigned short* d_assignment;

    assign(*x, *initialCenters, assignment);
    alg->initialize(x, k, assignment, 1);

    auto start = std::chrono::system_clock::now();
    std::cout << "alg run start" << std::endl;
    alg->run(5000);
    std::cout << "alg run end" << std::endl;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Sekunden: " << elapsed_seconds.count() << "\n";
   
    cudaDeviceSynchronize();

    delete[] assignment;
    delete alg;
    delete x;
    //cudaDeviceReset();

     //!____________________________________________________________________________________________________________________________________________________
   // const int numPnt = 499200;
   // const int numCent = 10;
   // const int numGrp = 5;
   // const int numDim = 10;
   // const  int numThread = 1;
   // const int maxIter = 5000;
   // const  int numGPU = 1;
   // double runtime;
   // int writeCentFlag = 0;
   // int writeAssignFlag = 0;
   // int writeTimeFlag = 0;
   // char* writeCentPath;
   // char* writeAssignPath;
   // char* writeTimePath;
   // int countFlag = 0;
   // unsigned long long int calcCount = 0;

   // std::cout << "bevor pointInfo" << std::endl;
   // //import and create dataset
   // PointInfo* pointInfo = (PointInfo*)malloc(sizeof(PointInfo) * numPnt);
   // //PointInfo* pointInfo = new PointInfo[numPnt];
   // DTYPE* pointData = (DTYPE*)malloc(sizeof(DTYPE) * numPnt * numDim);


   // if (importPoints(pointInfo, pointData, numPnt, numDim))
   // {
   //     // signal erroneous exit
   //     printf("\nERROR: could not import the dataset, please check file location. Exiting program.\n");
   //     free(pointInfo);
   //     free(pointData);
   //     return 1;
   // }

   // std::cout << "bevor centInfo" << std::endl;
   // CentInfo* centInfo = (CentInfo*)malloc(sizeof(CentInfo) * numCent);
   // DTYPE* centData = (DTYPE*)malloc(sizeof(DTYPE) * numCent * numDim);

   // // generate centroid data using dataset points
   // if (generateCentWithData(centInfo, centData, pointData, numCent, numPnt, numDim))
   // {
   //     // signal erroneous exit
   //     printf("\nERROR: Could not generate centroids. Exiting program.\n");
   //     free(pointInfo);
   //     free(pointData);
   //     free(centInfo);
   //     free(centData);
   //     return 1;
   // }
   // unsigned int ranIter;
   // std::cout << "alg run start" << std::endl;
   // std::cout << "numPnt: " << numPnt << std::endl;
   // std::cout << "numCent: " << numCent << std::endl;
   // std::cout << "numGrp: " << numGrp << std::endl;
   // std::cout << "numDim: " << numDim << std::endl;
   // std::cout << "maxIter: " << maxIter << std::endl;
   // std::cout << "numGPU: " << numGPU << std::endl;
   // std::cout << "pInfo: " << pointInfo[40000].centroidIndex << std::endl;
   // std::cout << "cInfo: " << centInfo[4].count << std::endl;
   // std::cout << "pData: " << pointData[400000] << std::endl;
   // std::cout << "cData: " << centData[40] << std::endl;
   // auto start = std::chrono::system_clock::now();
   // warmupGPU(numGPU);
   // //runtime = warumGehtNichts();
   ///* runtime =
   //     startLloydOnGPU(pointInfo, centInfo, pointData, centData,
   //         numPnt, numCent, numDim, maxIter, numGPU, &ranIter);*/
   ///* runtime =
   //     startLloydOnCPU(pointInfo, centInfo, pointData, centData,
   //         numPnt, numCent, numDim, 1, maxIter, &ranIter);*/
   // runtime =
   //     startSimpleOnGPU(pointInfo, centInfo, pointData, centData,
   //         numPnt, numCent, numGrp, numDim, maxIter, numGPU,
   //         &ranIter);
   // auto end = std::chrono::system_clock::now();
   // std::cout << "alg run end" << std::endl;
   // std::chrono::duration<double> elapsed_seconds = end - start;
   // std::cout << "Sekunden: " << elapsed_seconds.count() << "\n";
   // std::cout << "ITERATIONS: " << ranIter << std::endl;

   // free(pointData);
   // free(centData);
   // free(pointInfo);
   // free(centInfo);
   
     // cudaDeviceReset must be called before exiting in order for profiling and
     // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto res = cudaDeviceReset();
    if (res != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
