#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>


int main(int argc, char** argv) {
int N = 1000;
    
    if(argc == 2)
    {
       N = atoi(argv[1]);
    }
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Выделение pinned памяти на CPU
    float* h_matrix;
    cudaMallocHost((void**)&h_matrix, N * N * sizeof(float));

    // Генерация случайных данных на CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        h_matrix[i] = dis(gen);
    }

    // Выделение памяти на GPU
    float* d_matrix;
    cudaMalloc((void**)&d_matrix, N * N * sizeof(float));

    // Передача данных с CPU на GPU
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);


    // Проверка на ошибки CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // ... (здесь можно выполнить вычисления на GPU с использованием d_matrix) ...

    // Освобождение памяти на GPU
    cudaFree(d_matrix);

    // Освобождение pinned памяти на CPU
    cudaFreeHost(h_matrix);

    std::cout << "Передача матрицы завершена успешно!" << std::endl;
    return 0;
}



