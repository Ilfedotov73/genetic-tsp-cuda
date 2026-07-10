#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../include/math/point.hpp"
#include "../../../include/manage_memroy/memory.hpp"
#include "test_point_kernel.hpp"

// nvcc -c test/device/math/test_point_kernel.cu test_point_kernel.o -O3

__global__ void testPointDefaultConstructorDevice(float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_default_constructor(float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc<float>(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc<float>(&dev_y, sizeof(float));
        
    testPointDefaultConstructorDevice<<<1,1>>>(dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }
    
    cugtsp_memory::cumemcpy<float>(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy<float>(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointParamConstructorDevice(float src_x, float src_y, float *dev_x, float *dev_y)
{   
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(src_x, src_y);
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_param_constructor(float src_x, float src_y, float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc<float>(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc<float>(&dev_y, sizeof(float));

    testPointParamConstructorDevice<<<1,1>>>(src_x, src_y, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }
    
    cugtsp_memory::cumemcpy<float>(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy<float>(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointCopyConstructorDevice(const cugtsp_math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(*dev_src);
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_copy_constructor(float src_x, float src_y, float *out_x, float *out_y)
{    
    cugtsp_math::point_2_8_t src(src_x, src_y);
    
    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc<float>(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc<float>(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cugtsp_memory::cumalloc<cugtsp_math::point_2_8_t>(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointCopyConstructorDevice<<<1,1>>>(dev_src, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy<float>(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy<float>(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointGetDevice(float src_x, float src_y, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(src_x, src_y);
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_get(float src_x, float src_y, float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc<float>(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc<float>(&dev_y, sizeof(float));

    testPointGetDevice<<<1,1>>>(src_x, src_y, dev_x, dev_y);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy<float>(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy<float>(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointUnaryMinusDevice(float src_x, float src_y, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(src_x, src_y);
        *dev_x = -p.x();
        *dev_y = -p.y();
    }
}
void device_test::test_point_unary_minus(float src_x, float src_y, float *out_x, float *out_y)
{
    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    testPointUnaryMinusDevice<<<1,1>>>(src_x, src_y, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void testPointAssignmentDevice(const cugtsp_math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_assignment(float src_x, float src_y, float *out_x, float *out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);

    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cugtsp_memory::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointAssignmentDevice<<<1, 1>>>(dev_src, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointMinusPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                          float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a - *dev_src_b;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_minus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                         float *out_x, float *out_y)
{   
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay);
    cugtsp_math::point_2_8_t src_b(src_bx, src_by);

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cugtsp_memory::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cugtsp_memory::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    testPointMinusPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointPlusPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                         float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a + *dev_src_b;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_plus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                           float *out_x, float *out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay);
    cugtsp_math::point_2_8_t src_b(src_bx, src_by);

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cugtsp_memory::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cugtsp_memory::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    testPointPlusPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointMultPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                         float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a * *dev_src_b;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_mult_point(float src_ax, float src_ay, float src_bx, float src_by, 
                           float *out_x, float *out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay);
    cugtsp_math::point_2_8_t src_b(src_bx, src_by);

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cugtsp_memory::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cugtsp_memory::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    testPointMultPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointDivPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                         float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a / *dev_src_b;
        *dev_x = p.x();
        *dev_y = p.y();
    }
}
void device_test::test_point_div_point(float src_ax, float src_ay, float src_bx, float src_by, 
                          float *out_x, float *out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay);
    cugtsp_math::point_2_8_t src_b(src_bx, src_by);

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cugtsp_memory::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cugtsp_memory::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cugtsp_memory::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    float *dev_x, *dev_y;
    cugtsp_memory::cumalloc(&dev_x, sizeof(float));
    cugtsp_memory::cumalloc(&dev_y, sizeof(float));

    testPointDivPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cugtsp_memory::cumemcpy(out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cugtsp_memory::cumemcpy(out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

