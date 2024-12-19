#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>

// Ядро для перемножения матриц
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
   int N = 1000;
    
    if(argc == 2)
    {
       N = atoi(argv[1]);
    }

    // Выделение памяти на CPU и GPU
    float* h_A, * h_B, * h_C;
    float* d_A, * d_B, * d_C;
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Генерация случайных данных на CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    // Создание CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Асинхронное копирование данных на GPU (stream1)
    cudaMemcpyAsync(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice, stream1);

    // Вычисления на GPU (stream2)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMultiply << <gridDim, blockDim, 0, stream2 >> > (d_A, d_B, d_C, N);

    // Синхронизация stream2 (дождаться окончания вычислений)
    cudaStreamSynchronize(stream2);

    // Асинхронное копирование данных с GPU (stream1)
    cudaMemcpyAsync(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Синхронизация stream1 (дождаться окончания копирования)
    cudaStreamSynchronize(stream1);

    // Освобождение памяти и streams
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Перемножение матриц завершено!" << std::endl;
    return 0;
}
