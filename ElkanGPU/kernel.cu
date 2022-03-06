
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
//#include "FB1_elkan_kmeans.h"
//#include "MO_elkan_kmeans.h"

#include <fstream>
#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    printf("add: %i\n", a[i]);
    c[i] = a[i] + b[i];
}

__global__ void multiplyKernel(const float* a, const float* b, float* c) {
    int i = threadIdx.x;
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

int main()
{           
    cudaSetDevice(0);

    /*const int streamSize = 200;
    const int nStreams = 5;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&stream[i]);

    const int n = 1000;
    int* arr1 = new int[n];
    int* d_arr1;
    auto f = cudaMalloc(&d_arr1, n * sizeof(int));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr1)" << std::endl;
    }

    int* arr2 = new int[n];
    int* d_arr2;
    f = cudaMalloc(&d_arr2, n * sizeof(int));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr2)" << std::endl;
    }

    int* arr3 = new int[n];
    int* d_arr3;
    f = cudaMalloc(&d_arr3, n * sizeof(int));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr2)" << std::endl;
    }

    for (int i = 0; i < n; i++) {
        arr1[i] = i;
        arr2[i] = i;
    }

    int blockSize = 1 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cudaHostRegister(arr1, n * sizeof(int), cudaHostRegisterDefault);
    cudaHostRegister(arr2, n * sizeof(int), cudaHostRegisterDefault);
    cudaHostRegister(arr3, n * sizeof(int), cudaHostRegisterDefault);

    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_arr1[offset], &arr1[offset], n * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_arr2[offset], &arr2[offset], n  * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_arr3[offset], &arr3[offset], n * sizeof(double) / nStreams, cudaMemcpyHostToDevice, stream[i]);
    }
    cudaMemcpy(d_arr1, arr1, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr3, arr3, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaHostUnregister(arr1);
    cudaHostUnregister(arr2);
    cudaHostUnregister(arr3);
    addKernel << <1, n >> > (d_arr3, d_arr1, d_arr2);
    cudaDeviceSynchronize();

    cudaMemcpy(arr1, d_arr1, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr2, d_arr2, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr3, d_arr3, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        cout << arr1[i] << " + " << arr2[i] << " = " << arr3[i] << endl;
    }*/

   /* const int n = 1000;
    unsigned short* arr1 = new unsigned short[n];
    unsigned short* d_arr1;
    auto f = cudaMalloc(&d_arr1, n * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr1)" << std::endl;
    }

    unsigned short* arr2 = new unsigned short[n];
    unsigned short* d_arr2;
    f = cudaMalloc(&d_arr2, n * sizeof(unsigned short));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr2)" << std::endl;
    }

    int* test = new int[n];
    int* d_test;
    f = cudaMalloc(&d_test, n * sizeof(int));
    if (f != cudaSuccess) {
        std::cout << "cudaMalloc failed (arr2)" << std::endl;
    }

    for (int i = 0; i < n; i++) {
        arr1[i] = 0;
        arr2[i] = 0;
        test[i] = i % 100;
    }

    int blockSize = 2 * 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMemcpy(d_test, test, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr1, arr1, n * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, n * sizeof(unsigned short), cudaMemcpyHostToDevice);

    setTestt << <numBlocks, blockSize >> > (d_test, d_arr1, d_arr2);
    cudaDeviceSynchronize();
    cudaMemcpy(test, d_test, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr1, d_arr1, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr2, d_arr2, n * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << "i: " << i << " -> " << test[i] << std::endl;
    }*/

    
   cublasHandle_t cublas_handle;
    cublasStatus_t stat;
    stat = cublasCreate(&cublas_handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //cout << "cudaenginecount: " << prop.asyncEngineCount << endl;
    int k = 10;
    ElkanKmeans* alg = new ElkanKmeans();
    //FB1_ElkanKmeans* alg = new FB1_ElkanKmeans();
    //MO_ElkanKmeans* alg = new MO_ElkanKmeans();
    std::cout << "Alg: " << alg->getName() << std::endl;
    //Dataset* x = load_dataset("C:\\Users\\Admin\\Desktop\\MASTER\\skin_nonskin.txt");
    Dataset* x = load_randDataset(499200,10);
    if (x == nullptr) {
        cout << "Dataset generated" << endl;
        return 0;
    }
    cout << "Dataset loaded" << endl;
    //auto alg = make_unique<ElkanKmeans>(ElkanKmeans());
    Dataset* initialCenters = init_centers(*x, k);
    unsigned short* assignment = new unsigned short[x->n];
    unsigned short* d_assignment;

    //std::cout << "d_assignment malloc n: " << x->n << std::endl;
   
    assign(*x, *initialCenters, assignment);
    alg->initialize(x, k, assignment, 1);

    auto start = std::chrono::system_clock::now();
    std::cout << "alg run start" << std::endl;
    alg->run(5000);
    std::cout << "alg run end" << std::endl;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Sekunden: " << elapsed_seconds.count() << "\n";
    std::cout << std::numeric_limits<double>::max() << std::endl;
    //auto x = Dataset(5, 4);
    //x.fill(3.0);
    //x.print();
    cudaDeviceSynchronize();
    //cudaFree(d_assignment);
    delete assignment;
    delete alg;
    delete x;
    //cudaDeviceReset();

    /*int num = 5;
    float* data1 = new float[num];
    float* d_data1;
    cudaMalloc(&d_data1, num * sizeof(float));
    float* data2 = new float[num];
    float* d_data2;
    cudaMalloc(&d_data2, num * sizeof(float));
    float* data3 = new float[num];
    float* d_data3;
    cudaMalloc(&d_data3, num * sizeof(float));

    for (int i = 0; i < 5; i++)
        data1[i] = i;

    for (int i = 0; i < 5; i++)
        data2[i] = 5 - i;
    data2[2] = 2;
    data2[3] = 3;

    cudaMemcpy(d_data1, data1, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, data2, num * sizeof(float), cudaMemcpyHostToDevice);
   // cudaMemcpy(d_data3, data3, num * sizeof(float), cudaMemcpyHostToDevice);

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    const int s = 5;
    float* arr1;
    float* arr2;
    float* arr3;
    
    cudaMallocManaged(&arr1, 5 * sizeof(float));
    cudaMallocManaged(&arr2, 5 * sizeof(float));
    cudaMallocManaged(&arr3, 5 * sizeof(float));
    double* data;
    double* res;
    cudaMallocManaged(&data, 10 * sizeof(double));
    cudaMallocManaged(&res, 1 * sizeof(double));

    for (int i = 0; i < 5; i++)
        arr1[i] = i;

    for (int i = 0; i < 5; i++)
        arr2[i] = 5-i;
    arr2[2] = 2;
    arr2[3] = 3;

    for (int i = 0; i < 5; i++) {
        data[i] = arr1[i];
    }
    for (int i = 0; i < 5; i++) {
        data[i+5] = arr2[i];
    }
    for (int i = 0; i < 10; i++)
        cout << "i: " << i << " " << data[i] << endl;

    multiplyKernel<<<1,5>>>(arr1, arr2, arr3);
    //mult <<<1, s>>> (arr1, arr2, arr3);
    //dist2<<<1, 10>>> (data, 0, 1, 5, 0, res);
    //cublasSdot(cublas_handle, 5, arr1, 1, arr2, 1, &arr3[0]);
    cudaDeviceSynchronize();
    multiplyKernel << <1, 5 >> > (d_data1, d_data2, d_data3);
    cudaDeviceSynchronize();
    //cudaMemcpy(data1, d_data1, num * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(data2, d_data2, num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data3, d_data3, num * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //std::cout << "ERGEBNIS: " << *res << std::endl;

    
    for (int i = 0; i < 5; i++) {
        std::cout << "Managed:" << arr1[i] << " * " << arr2[i] << " = " << arr3[i] << std::endl;
        //std::cout << arr1[i] << " dot " << arr2[i] << std::endl;
    }

    for (int i = 0; i < 5; i++) {
        std::cout << "Nicht-Managed:" << data1[i] << " * " << data2[i] << " = " << data3[i] << std::endl;
        //std::cout << arr1[i] << " dot " << arr2[i] << std::endl;
    }
    //std::cout << "ERGEBNIS: " << *res << std::endl;
   // std::cout << "= " << arr3[0] << std::endl;
    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(arr3);
    //cublasDestroy(cublas_handle);
    std::cout << "------------" << std::endl;

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
     */
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto res = cudaDeviceReset();
    if (res != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
