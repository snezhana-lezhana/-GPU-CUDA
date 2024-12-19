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

    // Выделение Unified Memory
    float* A, * B, * C;
    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&B, N * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));


    // Генерация случайных данных (доступно как CPU, так и GPU)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Вычисление на GPU
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMultiply << <gridDim, blockDim >> > (A, B, C, N);

    // Синхронизация (необязательно, но рекомендуется для корректности)
    cudaDeviceSynchronize();


    // Проверка на ошибки CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }


    // Результат доступен как на CPU, так и на GPU
    // ... (обработка результата C) ...


    // Освобождение памяти
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    std::cout << "Перемножение матриц завершено!" << std::endl;
    return 0;
}


