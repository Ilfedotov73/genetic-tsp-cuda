#include <cmath>
#include <gtest/gtest.h>

#include "math/test_point_kernel.hpp"

/* 
 * nvcc -c kernel.cu -o kernel.o -O3
 * g++ -c test_kernel.cpp -o test_kernel.o -O3
 * g++ kernel.o test_kernel.o -o run_tests -L/usr/local/cuda-13.3/lib64 -lcudart -lgtest -lgtest_main
 */

TEST(Point_2_8_Test, TestPointDefaultConstructorDevice)
{
    float x = -1, y = -1;
    device_test::test_point_default_constructor(&x, &y);
    EXPECT_TRUE(std::isnan(x));
    EXPECT_TRUE(std::isnan(y));
}

TEST(Point_2_8_Test, TestPointParamConstructor)
{
    float x = 12, y = 13;
    float out_x, out_y;
    device_test::test_point_param_constructor(x, y, &out_x, &out_y);   
    EXPECT_FLOAT_EQ(out_x, x);
    EXPECT_FLOAT_EQ(out_y, y); 
}

/*
TEST(Point_2_8_Test, TestPointCopyConstructor)
{
    math::point_2_8_t src(12, 13);
    float out_x, out_y;
    device_test::test_point_copy_constructor(&src, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, src.x());
    EXPECT_FLOAT_EQ(out_y, src.y());
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}