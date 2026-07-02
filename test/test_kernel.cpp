#include <gtest/gtest.h>
#include "kernel.hpp"

/*
  * nvcc -c kernel.cu -o kernel.o -O3
  * g++ -c test_kernel.cpp -o test_kernel.o -O3
  * g++ kernel.o test_kernel.o -o run_tests -L/usr/local/cuda-13.3/lib64 -lcudart -lgtest -lgtest_main
*/

// Test Case for verifying GPU Vector Addition
TEST(CudaVectorAddTest, HandlesPositiveElements) {
    const int N = 5;
    float h_a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_b[N] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float h_c[N] = {0.0f};

    // Run the wrapped CUDA function
    runVectorAdd(h_a, h_b, h_c, N);

    // Verify expectations
    EXPECT_FLOAT_EQ(h_c[0], 11.0f);
    EXPECT_FLOAT_EQ(h_c[1], 22.0f);
    EXPECT_FLOAT_EQ(h_c[2], 33.0f);
    EXPECT_FLOAT_EQ(h_c[3], 44.0f);
    EXPECT_FLOAT_EQ(h_c[4], 55.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}