#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../include/math/point.hpp"
#include "test_point_kernel.hpp"

__global__ void testPointDefaultConstructorDevice(float *dev_x, float *dev_y)
{
    math::point_2_8_t p;
    *dev_x = p.x();
    *dev_y = p.y();
}
void device_test::test_point_default_constructor(float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cudaMalloc(&dev_x, sizeof(float));
    cudaMalloc(&dev_y, sizeof(float));
    
    testPointDefaultConstructorDevice<<<1,1>>>(out_x, out_y);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }
    
    cudaMemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointParamConstructorDevice(float x, float y, float *dev_x, float *dev_y)
{
    math::point_2_8_t p(x, y);
    *dev_x = p.x();
    *dev_y = p.y();
}
void device_test::test_point_param_constructor(float x, float y, float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cudaMalloc(&dev_x, sizeof(float));
    cudaMalloc(&dev_y, sizeof(float));

    testPointParamConstructorDevice<<<1,1>>>(x, y, dev_x, dev_y);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cudaMemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_x);
    cudaFree(dev_y);
}

/*
__global__ void testPointCopyConstructorDevice(const math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    math::point_2_8_t p(*dev_src);
    *dev_x = p.x();
    *dev_y = p.y();
}
void device_test::test_point_copy_constructor(const math::point_2_8_t *src, float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cudaMalloc(&dev_x, sizeof(float));
    cudaMalloc(&dev_y, sizeof(float));

    math::point_2_8_t *dev_src;
    cudaMalloc(&dev_src, sizeof(math::point_2_8_t));
    cudaMemcpy(dev_src, src, sizeof(math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointCopyConstructorDevice<<<1,1>>>(dev_src, dev_x, dev_y);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cudaMemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}
*/