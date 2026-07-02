#pragma once

#include <iostream>
#include <cstddef> // std::size_t
#include <stdexcept>
#include <cuda_runtime.h>

namespace manage_memory {
    template <typename T> struct unified_allocator {
        using value_type = T;
        using pointer = value_type*;
        using size_type = std::size_t;

        unified_allocator() noexcept = default;
        template <typename U> 
        unified_allocator(unified_allocator<U> const& right) noexcept {}

        __host__ __device__ pointer allocate(size_type count, const void* _Hint = 0)
        {
            if (count == 0) { return nullptr; }
            #ifndef __CUDA_ARCH__
                // Host
                pointer ptr = nullptr;
                cudaError_t err = cudaMallocManaged(&ptr, count * sizeof(value_type));
                if (err != cudaSuccess) {
                    throw std::runtime_error {cudaGetErrorString(err)};
                }
                return ptr;
            #else 
                // Device
                return new value_type[count]{};    
            #endif
        }  
        
        __host__ __device__ void deallocate(pointer ptr, size_type count = 0) 
        {
            if (ptr) {
                #ifndef __CUDA_ARCH__
                    // Host
                    cudaError_t err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        std::cerr << cudaGetErrorString(err);
                    }
                #else
                    // Device
                    delete[] ptr;
                #endif
            }
        }
    };

    template <typename T, typename U>
    __host__ __device__ inline bool operator==(unified_allocator<T> const &left, unified_allocator<U> const &right) {
        return true;
    }
}