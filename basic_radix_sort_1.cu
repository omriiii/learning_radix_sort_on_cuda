/*
*very* simple (and rigid) radix sort on cuda!

this is my implementation of radix sort from scratch as a learning exercise, to help others learn as well.
thus, to keep the code simple and readable but unoptimized, these concessions are made:
    1. this implementation uses "generic-kernels" that use grid-stride GLOBAL memory access patterns
    2. it does NOT use shared memory
    3. the prefix-sum function is not chunked, so array sizes MUST be in powers of 2

- each "radix-sort pass" sorts over 1 bit at a time
  you can see a nice graph of this implementation here: https://www.youtube.com/watch?v=0kLxAK9ANIc&t=1381s
  (if that link is a 404 for whatever reason i've attached a png of it in the repo too)

- the "prefix-sum" algo is taken from cuda gems 3's chapter 39
  you can see find it here: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

*/


#include <stdio.h>
#include <time.h>


//#define ARR_LEN 3200000
#define ARR_LEN 32
#define THREADS_PER_BLOCK 256
#define FAKE_DATA_ENTROPY 6
#define FAKE_DATA_MAX_VAL (1<<FAKE_DATA_ENTROPY)

__device__ void prefixSum(int* arr,
                          int arr_len,
                          int tid)
{
    int offset = 1;
    int temp_tid;

    // up-sweep
    for (int d = arr_len >> 1; d > 0;)
    {
        __syncthreads();

        temp_tid = tid;
        while (temp_tid < d)
        {
            int ai = offset * (2 * temp_tid + 1) - 1;
            int bi = offset * (2 * temp_tid + 2) - 1;

            arr[bi] += arr[ai];

            temp_tid += blockDim.x * gridDim.x;
        }
        offset *= 2;
        d = d >> 1;
    }

   // clear the last element
    if (tid == 0)
    {
        arr[arr_len - 1] = 0;
    }

    // down-sweep
    for (int d = 1; d < arr_len; d *= 2)
    {
        offset = offset >> 1;
        __syncthreads();

        temp_tid = tid;
        while (temp_tid < d)
        {
            int ai = offset * (2 * temp_tid + 1) - 1;
            int bi = offset * (2 * temp_tid + 2) - 1;

            float t = arr[ai];
            arr[ai] = arr[bi];
            arr[bi] += t;

            temp_tid += blockDim.x * gridDim.x;
        }
    }

    __syncthreads();

}

__global__ void radixSortKernel(int* d_data,
                                int* d_flipped_bit_digit,
                                int* d_prefix_sum,
                                int* d_arr_temp,
                                int arr_len,
                                int bit_to_digit_over)
{
    /*
    // debug code you can move around if you need to do an early return and view some value!
    temp_tid = tid;
    while (temp_tid < arr_len)
    {
        // copy the value over the original data since our code copies it to host and prints it after anyway
        d_data[temp_tid] = d_prefix_sum[temp_tid];
        temp_tid += blockDim.x * gridDim.x;
    }
    return;
    //*/


    int digit_offset = 0;
    int temp_tid;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while(bit_to_digit_over > 0)
    {

        // bool flag (e)
        temp_tid = tid;
        while (temp_tid < arr_len)
        {
            d_flipped_bit_digit[temp_tid] = ((~d_data[temp_tid]) & (1<<digit_offset))>>digit_offset;
            d_prefix_sum[temp_tid] = d_flipped_bit_digit[temp_tid];
            temp_tid += blockDim.x * gridDim.x;
        }

        __syncthreads();


        prefixSum(d_prefix_sum, arr_len, tid);

    /*
    // debug code you can move around if you need to do an early return and view some value!
    temp_tid = tid;
    while (temp_tid < arr_len)
    {
        // copy the value over the original data since our code copies it to host and prints it after anyway
        d_data[temp_tid] = ((~d_data[temp_tid]) & (1<<digit_offset))>>digit_offset;
        temp_tid += blockDim.x * gridDim.x;
    }
    return;
    */

        int total_falses = d_prefix_sum[ARR_LEN-1] + d_flipped_bit_digit[ARR_LEN-1];
        temp_tid = tid;

        while (temp_tid < arr_len)
        {
            // t = i - f + total_falses
            d_flipped_bit_digit[temp_tid] = temp_tid - d_prefix_sum[temp_tid] + total_falses;

            // FINAL_IDX = b ? t : f
            d_prefix_sum[temp_tid] = (d_data[temp_tid] & (1<<digit_offset)) ? d_flipped_bit_digit[temp_tid] : d_prefix_sum[temp_tid];

            temp_tid += blockDim.x * gridDim.x;
        }

        __syncthreads();

        temp_tid = tid;
        while (temp_tid < arr_len)
        {
            d_arr_temp[d_prefix_sum[temp_tid]] = d_data[temp_tid];
            temp_tid += blockDim.x * gridDim.x;
        }

        __syncthreads();

        temp_tid = tid;
        while (temp_tid < arr_len)
        {
            d_data[temp_tid] = d_arr_temp[temp_tid];
            temp_tid += blockDim.x * gridDim.x;
        }

        bit_to_digit_over--;
        digit_offset++;

    }

}

void radixSort(int h_arr[],
               int arr_len)
{
    int blocksPerGrid = min((arr_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1024);
    printf("blocksPerGrid: %d\n", blocksPerGrid);

    // *not optimal
    int* d_arr;
    int* d_flipped_bit_digit;
    int* d_prefix_sum;
    int* d_arr_temp;

    cudaMalloc(&d_arr,        ARR_LEN * sizeof(int));
    cudaMalloc(&d_flipped_bit_digit,      ARR_LEN * sizeof(int));
    cudaMalloc(&d_prefix_sum, ARR_LEN * sizeof(int));
    cudaMalloc(&d_arr_temp, ARR_LEN * sizeof(int));

    cudaMemcpy(d_arr, h_arr, ARR_LEN * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start = clock();
    radixSortKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_arr,
                                                          d_flipped_bit_digit,
                                                          d_prefix_sum,
                                                          d_arr_temp,
                                                          arr_len,
                                                          FAKE_DATA_ENTROPY);

    clock_t end = clock();
    printf("took %f ms\n", 1000.0*(double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(h_arr, d_arr, ARR_LEN * sizeof(int), cudaMemcpyDeviceToHost);

}



void checkArrIsSorted(int* arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            printf("Array is NOT sorted! Error at index %d\n", i);
            return;
        }
    }
    printf("Array is sorted correctly!\n");
}

void printArr(int* arr, int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}


int main()
{
    // Make sure ARR_LEN is a power of 2
    int arr_len_temp = ARR_LEN;
    while(arr_len_temp)
    {
        if ((arr_len_temp & 1) && (arr_len_temp >> 1))
        {
            printf("ARR_LEN must be a power of 2!\n");
            return 1;
        }
        arr_len_temp = arr_len_temp >> 1;
    }

    static int h_arr[ARR_LEN];

    //srand(time(NULL));
    for (int i = 0; i < ARR_LEN; i++)
    {
        h_arr[i] = rand()%FAKE_DATA_MAX_VAL;
    }

    if (ARR_LEN < 256) { printArr(h_arr, ARR_LEN); }

    radixSort(h_arr, ARR_LEN);

    if (ARR_LEN < 256) { printArr(h_arr, ARR_LEN); }
    else { checkArrIsSorted(h_arr, ARR_LEN); }

    return 0;
}
