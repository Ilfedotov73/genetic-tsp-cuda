/* nvcc -c test/math/test_point_kernel.cu test_point_kernel.o -O3 */

#include <cuda_runtime.h>
#include <stdexcept>

#include "../../include/math/point.hpp"
#include "../../include/memory/cumem.hpp"
#include "test_point_kernel.hpp"

using namespace cu_gtsp;

__global__ void testPointDefaultConstructorDevice(float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p;
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_default_constructor(float &out_x, float &out_y)
{
    float *dev_x, *dev_y;
    cu_mem::cumalloc<float>(&dev_x, sizeof(float));
    cu_mem::cumalloc<float>(&dev_y, sizeof(float));
        
    testPointDefaultConstructorDevice<<<1,1>>>(dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }
    
    cu_mem::cumemcpy<float>(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy<float>(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointParamConstructorDevice(float src_x, float src_y, float *dev_x, float *dev_y)
{   
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(src_x, src_y);
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_param_constructor(float src_x, float src_y, float &out_x, float &out_y)
{
    float *dev_x, *dev_y;
    cu_mem::cumalloc<float>(&dev_x, sizeof(float));
    cu_mem::cumalloc<float>(&dev_y, sizeof(float));

    testPointParamConstructorDevice<<<1,1>>>(src_x, src_y, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }
    
    cu_mem::cumemcpy<float>(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy<float>(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointCopyConstructorDevice(const cugtsp_math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p(*dev_src);
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_copy_constructor(float src_x, float src_y, float &out_x, float &out_y)
{    
    cugtsp_math::point_2_8_t src(src_x, src_y);
    
    float *dev_x, *dev_y;
    cu_mem::cumalloc<float>(&dev_x, sizeof(float));
    cu_mem::cumalloc<float>(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc<cugtsp_math::point_2_8_t>(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointCopyConstructorDevice<<<1,1>>>(dev_src, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy<float>(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy<float>(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointGetDevice(const cugtsp_math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_x = dev_src->x();
        *dev_y = dev_src->y();
    }
}
void point_test_device::test_point_get(float src_x, float src_y, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);

    float *dev_x, *dev_y;
    cu_mem::cumalloc<float>(&dev_x, sizeof(float));
    cu_mem::cumalloc<float>(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointGetDevice<<<1,1>>>(dev_src, dev_x, dev_y);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy<float>(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy<float>(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointUnaryMinusDevice(cugtsp_math::point_2_8_t *dev_src, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_x = -(*dev_src).x_;
        *dev_y = -(*dev_src).y_;
    }
}
void point_test_device::test_point_unary_minus(float src_x, float src_y, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointUnaryMinusDevice<<<1,1>>>(dev_src, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void testPointAssignmentDevice(const cugtsp_math::point_2_8_t *dev_src, cugtsp_math::point_2_8_t *p, 
                                          float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *p = *dev_src;
        *dev_x = p->x_;
        *dev_y = p->y_;
    }
}
void point_test_device::test_point_assignment(float src_x, float src_y, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src, *p;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&p, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointAssignmentDevice<<<1, 1>>>(dev_src, p, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointMinusEqualDevice(cugtsp_math::point_2_8_t *dev_src, const cugtsp_math::point_2_8_t *dev_p, 
                                          float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_src -= *dev_p; 
        *dev_x = dev_src->x_;
        *dev_y = dev_src->y_;
    }
}
void point_test_device::test_point_minus_equal(float src_ax, float src_ay, float src_bx, float src_by,
                                               float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_ax, src_ay), p(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src, *dev_p;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_p, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_p, &p, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointMinusEqualDevice<<<1, 1>>>(dev_src, dev_p, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_p);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointPlusEqualDevice(cugtsp_math::point_2_8_t *dev_src, const cugtsp_math::point_2_8_t *dev_p, 
                                          float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_src += *dev_p; 
        *dev_x = dev_src->x_;
        *dev_y = dev_src->y_;
    }
}
void point_test_device::test_point_plus_equal(float src_ax, float src_ay, float src_bx, float src_by, 
                                              float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_ax, src_ay), p(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src, *dev_p;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_p, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_p, &p, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointPlusEqualDevice<<<1, 1>>>(dev_src, dev_p, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_p);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointScalarMultEqualDevice(cugtsp_math::point_2_8_t *dev_src, float t, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_src *= t;
        *dev_x = dev_src->x_;
        *dev_y = dev_src->y_;
    }
}
void point_test_device::test_point_scalar_mult(float src_x, float src_y, float t, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);
    
    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointScalarMultEqualDevice<<<1, 1>>>(dev_src, t, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointScalarDivEqualDevice(cugtsp_math::point_2_8_t *dev_src, float t, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        *dev_src /= t;
        *dev_x = dev_src->x_;
        *dev_y = dev_src->y_;
    }
}
void point_test_device::test_point_scalar_div(float src_x, float src_y, float t, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src(src_x, src_y);
    
    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointScalarDivEqualDevice<<<1, 1>>>(dev_src, t, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointMinusPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                          float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a - *dev_src_b;
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_minus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                               float &out_x, float &out_y)
{   
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay), src_b(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));
    
    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cu_mem::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointMinusPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

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
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_plus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                              float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay), src_b(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cu_mem::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointPlusPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

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
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_mult_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                              float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay), src_b(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cu_mem::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointMultPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointMultScalarDevice(const cugtsp_math::point_2_8_t *dev_p, float t, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_p * t;
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_mult_scalar(float src_x, float src_y, float t, float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src_p(src_x, src_y);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_p;
    cu_mem::cumalloc(&dev_p, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_p, &src_p, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointMultScalarDevice<<<1, 1>>>(dev_p, t, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_p);
}

__global__ void testPointDivPointDevice(const cugtsp_math::point_2_8_t *dev_src_a, const cugtsp_math::point_2_8_t *dev_src_b,
                                         float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_src_a / *dev_src_b;
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
}
void point_test_device::test_point_div_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                             float &out_x, float &out_y)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay), src_b(src_bx, src_by);

    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t *dev_src_a, *dev_src_b;
    cu_mem::cumalloc(&dev_src_a, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_src_b, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_src_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointDivPointDevice<<<1, 1>>>(dev_src_a, dev_src_b, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src_a);
    cudaFree(dev_src_b);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__global__ void testPointDivScalarDevice(const cugtsp_math::point_2_8_t *dev_p, float t, float *dev_x, float *dev_y)
{
    if (threadIdx.x == 0) {
        cugtsp_math::point_2_8_t p = *dev_p / t;
        *dev_x = p.x_;
        *dev_y = p.y_;
    }
} 
void point_test_device::test_point_div_scalar(float src_x, float src_y, float t, float &out_x, float &out_y)
{
    float *dev_x, *dev_y;
    cu_mem::cumalloc(&dev_x, sizeof(float));
    cu_mem::cumalloc(&dev_y, sizeof(float));

    cugtsp_math::point_2_8_t src(src_x, src_y);
    cugtsp_math::point_2_8_t *dev_src;
    cu_mem::cumalloc(&dev_src, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_src, &src, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointDivScalarDevice<<<1, 1>>>(dev_src, t, dev_x, dev_y);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);
    cu_mem::cumemcpy(&out_y, dev_y, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_src);
}

__global__ void testPointValidDevice(const cugtsp_math::point_2_8_t *dev_p, bool *dev_valid)
{
    if (threadIdx.x == 0) {
        *dev_valid = dev_p->valid();
    }
}

void point_test_device::test_point_valid(float src_x, float src_y, bool &out_valid)
{
    cugtsp_math::point_2_8_t src_p(src_x, src_y);

    bool *dev_valid;
    cu_mem::cumalloc(&dev_valid, sizeof(bool));

    cugtsp_math::point_2_8_t *dev_p;
    cu_mem::cumalloc(&dev_p, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_p, &src_p, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);

    testPointValidDevice<<<1, 1>>>(dev_p, dev_valid);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_valid, dev_valid, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(dev_p);
    cudaFree(dev_valid);
}

__global__ void testPointEqualityDevice(const cugtsp_math::point_2_8_t *dev_pa, const cugtsp_math::point_2_8_t *dev_pb, 
                                     bool *dev_equal)
{
    if (threadIdx.x == 0) {
        *dev_pa == *dev_pb ?  *dev_equal = true : *dev_equal = false;
    }
}
void point_test_device::test_point_equality(float src_ax, float src_ay, float src_bx, float src_by, bool &out_equal)
{
    cugtsp_math::point_2_8_t src_pa(src_ax, src_ay), src_pb(src_bx, src_by);

    bool *dev_equal;
    cu_mem::cumalloc(&dev_equal, sizeof(bool));

    cugtsp_math::point_2_8_t *dev_pa, *dev_pb;
    cu_mem::cumalloc(&dev_pa, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_pb, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_pa, &src_pa, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_pb, &src_pb, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    
    testPointEqualityDevice<<<1, 1>>>(dev_pa, dev_pb, dev_equal);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_equal, dev_equal, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(dev_pa);
    cudaFree(dev_pb);
    cudaFree(dev_equal);
}

__global__ void testPointDistance(const cugtsp_math::point_2_8_t *dev_a, const cugtsp_math::point_2_8_t *dev_b,
                                  float *dev_dist)
{
    if (threadIdx.x == 0) {
        *dev_dist = cugtsp_math::distance(*dev_a, *dev_b);
    }
}
void point_test_device::test_point_distaince(float src_ax, float src_ay, float src_bx, float src_by,
                                             float &out_dist)
{
    cugtsp_math::point_2_8_t src_a(src_ax, src_ay), src_b(src_bx, src_by);

    float *dev_dist;
    cu_mem::cumalloc(&dev_dist, sizeof(float));
    
    cugtsp_math::point_2_8_t *dev_a, *dev_b;
    cu_mem::cumalloc(&dev_a, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumalloc(&dev_b, sizeof(cugtsp_math::point_2_8_t));
    cu_mem::cumemcpy(dev_a, &src_a, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    cu_mem::cumemcpy(dev_b, &src_b, sizeof(cugtsp_math::point_2_8_t), cudaMemcpyHostToDevice);
    
    testPointDistance<<<1, 1>>>(dev_a, dev_b, dev_dist);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::runtime_error{cudaGetErrorString(err)};
    }

    cu_mem::cumemcpy(&out_dist, dev_dist, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_dist);
}