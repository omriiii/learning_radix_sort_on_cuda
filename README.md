# Learning Radix Sort On Cuda

This is a public repo documenting my learning.

`single-bit-radix-sort-pass-algorithm.png`'s source is from [AstroGPU - CUDA Data Parallel Algorithms - Mark Harris](https://www.youtube.com/watch?v=0kLxAK9ANIc&t=1381s) 

`testing_arr_edgecase_idx.cu` is me comparing what's fastest way of inserting "one-off" values in array-wide operations (eg. arr[i] = i+1, expect arr[0] = 0 -> do you have an if statment in the kernel? do you fire antoher kernel? run memcpy?) (spoilers: it's the one-off if statemnt)
