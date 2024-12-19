#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>

#define BLOCK_SIZE 16 // Размер блока (количество потоков в блоке)


__global__ void matrixMultiplyShared(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        // Загрузка данных в Shared Memory
        tileA[threadIdx.y][threadIdx.x] = (row < N && k * BLOCK_SIZE + threadIdx.x < N) ? A[row * N + k * BLOCK_SIZE + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (k * BLOCK_SIZE + threadIdx.y < N && col < N) ? B[(k * BLOCK_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads(); // Синхронизация потоков в блоке

        // Вычисление частичной суммы
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Сохранение результата в глобальную память
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


int main(int argc, char** argv) {
    int N = 1000;
    
    if(argc == 2)
    {
       N = atoi(argv[1]);
    }
    // Проверка на корректный размер матрицы
    if (N % BLOCK_SIZE != 0) {
        std::cerr << "Error: Matrix size must be a multiple of BLOCK_SIZE." << std::endl;
        return 1;
    }

    float* h_A, * h_B, * h_C;
    float* d_A, * d_B, * d_C;

    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));


    // Генерация случайных данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplyShared << <gridDim, blockDim >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Перемножение матриц завершено!" << std::endl;
    return 0;
}