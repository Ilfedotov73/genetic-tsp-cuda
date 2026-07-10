#include <cmath>
#include <gtest/gtest.h>

#include "math/test_point_kernel.hpp"

/* 
 * g++ -c test/device/test_main_kernel.cpp -o test_main_kernel.o -O3
 * g++ test_main_kernel.o test_point_kernel.o -o run_tests -lcudart -L/usr/local/cuda-13.3/lib64 -lgtest -lgtest_main 
 */

TEST(Point_2_8_Test, TestPointDefaultConstructorDevice)
{
    float x = -1.0f, y = -1.0f;
    device_test::test_point_default_constructor(&x, &y);
    EXPECT_TRUE(std::isnan(x));
    EXPECT_TRUE(std::isnan(y));
}

TEST(Point_2_8_Test, TestPointParamConstructor)
{
    float x = 12.0f, y = 13.0f;
    float out_x, out_y;
    device_test::test_point_param_constructor(x, y, &out_x, &out_y);   
    EXPECT_FLOAT_EQ(out_x, x);
    EXPECT_FLOAT_EQ(out_y, y); 
}


TEST(Point_2_8_Test, TestPointCopyConstructor)
{  
    float x = 12.0f, y = 13.0f;
    float out_x, out_y;
    device_test::test_point_copy_constructor(x, y, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, x);
    EXPECT_FLOAT_EQ(out_y, y);
}

TEST(Point_2_8_Test, TestPointGet)
{
    float x = 12.0f, y = 13.0f;
    float out_x, out_y;
    device_test::test_point_get(x, y, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, x);
    EXPECT_FLOAT_EQ(out_y, y); 
}

TEST(Point_2_8_Test, TestPointUnaryMinus)
{
    float x = 12.0f, y = 13.0f;
    float out_x, out_y;
    device_test::test_point_unary_minus(x, y, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, -x);
    EXPECT_FLOAT_EQ(out_y, -y);
}

TEST(Point_2_8_Test, TestPointAssignment)
{
    float x = 12.0f, y = 13.0f;
    float out_x, out_y;
    device_test::test_point_assignment(x, y, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, x);
    EXPECT_FLOAT_EQ(out_y, y);
}

TEST(Point_2_8_Test, TestPointMinusPoint)
{
    float ax = 12.0f, ay = 13.0f;
    float bx = 12.1f, by = 13.1f;
    float out_x, out_y;
    device_test::test_point_minus_point(ax, ay, bx, by, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, ax - bx);
    EXPECT_FLOAT_EQ(out_y, ay - by);
}

TEST(Point_2_8_Test, TestPointPlusPoint)
{
    float ax = 12.0f, ay = 13.0f;
    float bx = 12.1f, by = 13.1f;
    float out_x, out_y;
    device_test::test_point_plus_point(ax, ay, bx, by, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, ax + bx);
    EXPECT_FLOAT_EQ(out_y, ay + by);
}

TEST(Point_2_8_Test, TestPointMultPoint)
{
    float ax = 12.0f, ay = 13.0f;
    float bx = 12.1f, by = 13.1f;
    float out_x, out_y;
    device_test::test_point_mult_point(ax, ay, bx, by, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, ax * bx);
    EXPECT_FLOAT_EQ(out_y, ay * by);
}

TEST(Point_2_8_Test, TestPointDivPoint)
{
    float ax = 12.0f, ay = 13.0f;
    float bx = 12.1f, by = 13.1f;
    float out_x, out_y;
    device_test::test_point_div_point(ax, ay, bx, by, &out_x, &out_y);
    EXPECT_FLOAT_EQ(out_x, ax / bx);
    EXPECT_FLOAT_EQ(out_y, ay / by);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}