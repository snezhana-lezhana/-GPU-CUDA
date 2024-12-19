NVCC := nvcc
CUDA_FLAGS := -O3 -arch=sm_90a

all: matrix_multiply_basic \
    matrix_multiply_pinned \
    matrix_multiply_unified \
    matrix_multiply_streams \
    matrix_multiply_shared \
    matrix_multiply_cublas \
    matrix_multiply_omp

matrix_multiply_basic: matrix_multiply_basic.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_basic matrix_multiply_basic.cu

matrix_multiply_pinned: matrix_multiply_pinned.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_pinned matrix_multiply_pinned.cu

matrix_multiply_unified: matrix_multiply_unified.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_unified matrix_multiply_unified.cu

matrix_multiply_streams: matrix_multiply_streams.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_streams matrix_multiply_streams.cu

matrix_multiply_shared: matrix_multiply_shared.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_shared matrix_multiply_shared.cu

matrix_multiply_cublas: matrix_multiply_cublas.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_cublas matrix_multiply_cublas.cu -lcublas

matrix_multiply_omp: matrix_multiply_omp.cu
  $(NVCC) $(CUDA_FLAGS) -o matrix_multiply_omp matrix_multiply_omp.cu -fopenmp

clean:
  rm -f *.o *~ $(SOURCES:.cu=.out) matrix_multiply_*