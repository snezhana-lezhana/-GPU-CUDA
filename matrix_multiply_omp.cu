#include <iostream>
#include <random>
#include <omp.h> // OpenMP header
#include <cuda_runtime.h>
#include <cstdlib>


__global__ void matrixMultiplyGPU(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[col + k * N] * B[k * N + row]; // Column-major order
        }
        C[col + row * N] = sum; // Column-major order
    }
}

int main(int argc, char** argv) {

int N = 1000;
    
    if(argc == 2)
    {
       N = atoi(argv[1]);
    }

    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;

    h_A = new double[N * N];
    h_B = new double[N * N];
    h_C = new double[N * N];
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));


    // Генерация случайных данных на CPU с OpenMP
    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (double)rand() / RAND_MAX;
        h_B[i] = (double)rand() / RAND_MAX;
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMultiplyGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Обработка результатов на CPU с OpenMP (при необходимости)
    // #pragma omp parallel for ...


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Matrix multiplication complete!" << std::endl;
    return 0;
}
