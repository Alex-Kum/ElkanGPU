
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

__global__ void addKernel(int* c, const int* a, const int* b)
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

   
     // cudaDeviceReset must be called before exiting in order for profiling and
     // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto res = cudaDeviceReset();
    if (res != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
