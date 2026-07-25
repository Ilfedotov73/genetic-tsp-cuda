#include <cuda_runtime.h>
#include <stdexcept>
#include <cstddef> // std::size_t

#include "test_cumem_kernel.hpp"
#include "../../include/memory/cumem.hpp"

using namespace cu_gtsp;

template <typename T>
__global__ void testUnifiedAllocatorDevice(T* dev_seq, std::size_t count, T increment)
{
    int tid = blockIdx.x;
    if (tid < count) {
        dev_seq[tid] += increment;
    }
}
template <typename T, std::size_t N> 
void cumem_test_device::test_unified_allocator_allocate_seq(T (&host_seq)[N])
{
    cu_alloc::unified_allocator<T> alloc_;
    
    T *dev_seq = nullptr;
    dev_seq = alloc_.allocate(N);

    for (std::size_t i = 0; i < N; ++i) {
        dev_seq[i] = host_seq[i];
    }

    //testUnifiedAllocatorDevice<<<N, 1>>>(dev_seq, increment);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    for (std::size_t i = 0; i < N; ++i) {
        host_seq[i] = dev_seq[i];
    }
    alloc_.deallocate(dev_seq);   
}