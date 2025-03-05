#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matMulKernel(const double *mat1, const double *mat2, double *temp, size_t row, size_t mid, size_t col) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < row && c < col) {
        double sum = 0.0;
        for (size_t k = 0; k < mid; k++) {
            sum += mat1[r * mid + k] * mat2[k * col + c];
        }
        temp[r * col + c] = sum;
    }
}

extern "C" void launchMatrixMultiply(const double* mat1, const double* mat2, double* result, 
                                    size_t row, size_t mid, size_t col) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((col + blockDim.x - 1) / blockDim.x, 
                 (row + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matMulKernel<<<gridDim, blockDim>>>(mat1, mat2, result, row, mid, col);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}