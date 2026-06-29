#pragma once

#include <cuda_runtime.h>

namespace manage_memory {
    template <typename T> struct allocator {
        __host__ __device__ T* allocate(size_t size)
        {
            if (size == 0) { return; }
            #ifndef __CUDA_ARCH__
                // Host
                T *ptr = nullptr;
                cudaMallocManaged(&ptr, size * sizeof(T));
                return ptr;
            #else 
                // Device
                return new T[size]{};    
            #endif
        }  
        
        __host__ __device__ void deallocate(T* ptr) 
        {
            #ifndef __CUDA_ARCH__
                // Host
                cudaFree(ptr);
            #else
                // Device
                delete[] ptr;
            #endif
        }
    };
}