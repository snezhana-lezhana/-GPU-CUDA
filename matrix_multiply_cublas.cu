#include <cublas_v2.h>
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
    // Выделение памяти на CPU
    double* h_A, * h_B, * h_C;
    h_A = new double[N * N];
    h_B = new double[N * N];
    h_C = new double[N * N];

    // Генерация случайных данных на CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    cudaEventRecord(start, 0);
    // Выделение памяти на GPU
    double* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);


    // Инициализация cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Параметры для cuBLAS dgemm
    double alpha = 1.0;
    double beta = 0.0;
    int lda = N; // Leading dimension of A
    int ldb = N; // Leading dimension of B
    int ldc = N; // Leading dimension of C


    // Выполнение перемножения матриц с использованием cuBLAS dgemm
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);


    // Копирование результата с GPU на CPU
    cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);


    // Очистка ресурсов cuBLAS
    cublasDestroy(handle);

    cudaEventRecord(stop, 0);
    // Освобождение памяти на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Освобождение памяти на CPU
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Перемножение матриц завершено!" << std::endl;
    return 0;
}
