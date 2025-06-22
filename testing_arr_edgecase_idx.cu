#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>


#define N 1000000
#define BLOCK_SIZE 256
#define ITERATIONS 1000

// Method 1: Conditional in main kernel
__global__ void kernel_with_conditional(float* arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) arr[0] = 0.0f;
    if(idx < N) {
        arr[idx] = idx + 1;
    }
}

// Method 2: Main kernel only
__global__ void kernel_main(float* arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        arr[idx] = idx + 1;
    }
}

// Method 3: Separate single-thread kernel
__global__ void kernel_init_single() {
    // Empty - just for launch overhead test
}

__global__ void emptyKernel() { }
void runEmptyKernel() { emptyKernel<<<1, 32>>>(); KERNEL_CHECK_ERROR(); }

double benchmark_conditional() {
    float* d_arr;
    cudaMalloc(&d_arr, N * sizeof(float));

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < ITERATIONS; i++) {
        kernel_with_conditional<<<grid, block>>>(d_arr);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_arr);
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double benchmark_separate_kernel() {
    float* d_arr;
    cudaMalloc(&d_arr, N * sizeof(float));

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < ITERATIONS; i++) {
        kernel_init_single<<<1, 1>>>();
        kernel_main<<<grid, block>>>(d_arr);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_arr);
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double benchmark_memcpy() {
    float* d_arr;
    float zero = 0.0f;
    cudaMalloc(&d_arr, N * sizeof(float));

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < ITERATIONS; i++) {
        cudaMemcpy(d_arr, &zero, sizeof(float), cudaMemcpyHostToDevice);
        kernel_main<<<grid, block>>>(d_arr);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_arr);
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    printf("CUDA Array Initialization Benchmark\n");
    printf("Array size: %d, Iterations: %d\n\n", N, ITERATIONS);

    runEmptyKernel();

    double time1 = benchmark_conditional();
    double time2 = benchmark_separate_kernel();
    double time3 = benchmark_memcpy();

    printf("Method 1 (Conditional):     %.3f ms\n", time1);
    printf("Method 2 (Separate kernel): %.3f ms\n", time2);
    printf("Method 3 (Memcpy):          %.3f ms\n", time3);

    printf("\nSpeedup vs Conditional:\n");
    printf("Separate kernel: %.2fx %s\n", time2/time1, time2 > time1 ? "slower" : "faster");
    printf("Memcpy:          %.2fx %s\n", time3/time1, time3 > time1 ? "slower" : "faster");

    return 0;
}
