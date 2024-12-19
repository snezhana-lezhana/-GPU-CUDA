#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cstdlib>

// Размеры матриц
#define THREADS_PER_BLOCK 512

// Ядро для перемножения матриц
__global__ void matrixMultiply(float* A, float* B, float* C, int N, int M) {
    // Индексы потока
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ матрицы
    if (row < N && col < M) {
        // Вычисление элемента матрицы C
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

int main(int argc, char** argv) {

int N = 1000;
    
    if(argc == 2)
    {
       N = atoi(argv[1]);
    }

    // Генерация случайных матриц на CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * M];

    for (int i = 0; i < N * N; i++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Выделение памяти на GPU
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * M);

    // Передача матриц на GPU
    cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // Вычисление на GPU
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrixMultiply << <gridDim, blockDim >> > (d_A, d_B, d_C, N, M);

    // Синхронизация CPU и GPU
    cudaDeviceSynchronize();

    // Получение результата с GPU
    cudaMemcpy(C, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

    // Освобождение памяти на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Вывод результата
    // (В реальности можно добавить проверку правильности вычислений)
    std::cout << "Перемножение матриц завершено!" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

