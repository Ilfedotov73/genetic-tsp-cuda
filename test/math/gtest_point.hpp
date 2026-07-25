#pragma once

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

#include "test_point_kernel.hpp"

namespace point_test_device {
    TEST(Point_2_8_Test, TestPointDefaultConstructorDevice)
    {
        float x = -1.0f, y = -1.0f;
        test_point_default_constructor(x, y);
        EXPECT_TRUE(std::isnan(x));
        EXPECT_TRUE(std::isnan(y));
    }
    
    TEST(Point_2_8_Test, TestPointParamConstructor)
    {
        float x = 12.0f, y = 13.0f;
        float out_x, out_y;
        test_point_param_constructor(x, y, out_x, out_y);   
        EXPECT_FLOAT_EQ(out_x, x);
        EXPECT_FLOAT_EQ(out_y, y); 
    }
    
    TEST(Point_2_8_Test, TestPointCopyConstructor)
    {  
        float x = 12.0f, y = 13.0f;
        float out_x, out_y;
        test_point_copy_constructor(x, y, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x);
        EXPECT_FLOAT_EQ(out_y, y);
    }
    
    TEST(Point_2_8_Test, TestPointGet)
    {
        float x = 12.0f, y = 13.0f;
        float out_x, out_y;
        test_point_get(x, y, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x);
        EXPECT_FLOAT_EQ(out_y, y); 
    }
    
    TEST(Point_2_8_Test, TestPointUnaryMinus)
    {
        float x = 12.0f, y = 13.0f;
        float out_x, out_y;
        test_point_unary_minus(x, y, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, -x);
        EXPECT_FLOAT_EQ(out_y, -y);
    }
    
    TEST(Point_2_8_Test, TestPointAssignment)
    {
        float x = 12.0f, y = 13.0f;
        float out_x, out_y;
        test_point_assignment(x, y, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x);
        EXPECT_FLOAT_EQ(out_y, y);
    }
    
    TEST(Point_2_8_Test, TestPointMinusEqual)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.0f;
        float out_x, out_y;
        test_point_minus_equal(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax -= bx);
        EXPECT_FLOAT_EQ(out_y, ay -= by);
    }

    TEST(Point_2_8_Test, TestPointPlusEqual)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.1f;
        float out_x, out_y;
        test_point_plus_equal(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax + bx);
        EXPECT_FLOAT_EQ(out_y, ay + by);
    }

    TEST(Point_2_8_Test, TestPointScalarMutl)
    {
        float x = 12.0f, y = 13.0f, t = 5.0f;
        float out_x, out_y;
        test_point_scalar_mult(x, y, t, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x *= t);
        EXPECT_FLOAT_EQ(out_y, y *= t);
    }
    
    TEST(Point_2_8_Test, TestPointScalarDiv)
    {
        float x = 12.0f, y = 13.0f, t = 5.0f;
        float out_x, out_y;
        test_point_scalar_div(x, y, t, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x /= t);
        EXPECT_FLOAT_EQ(out_y, y /= t);
    }

    TEST(Point_2_8_Test, TestPointMinusPoint)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.1f;
        float out_x, out_y;
        test_point_minus_point(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax - bx);
        EXPECT_FLOAT_EQ(out_y, ay - by);
    }
    
    TEST(Point_2_8_Test, TestPointPlusPoint)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.1f;
        float out_x, out_y;
        test_point_plus_point(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax + bx);
        EXPECT_FLOAT_EQ(out_y, ay + by);
    }
    
    TEST(Point_2_8_Test, TestPointMultPoint)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.1f;
        float out_x, out_y;
        point_test_device::test_point_mult_point(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax * bx);
        EXPECT_FLOAT_EQ(out_y, ay * by);
    }
    
    TEST(Point_2_8_Test, TestPointDivPoint)
    {
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.1f, by = 13.1f;
        float out_x, out_y;
        point_test_device::test_point_div_point(ax, ay, bx, by, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, ax / bx);
        EXPECT_FLOAT_EQ(out_y, ay / by);
    }

    TEST(Point_2_8_Test, TestPointFalseValid)
    {
        float x = 13.0f, y = NAN;
        bool valid;
        point_test_device::test_point_valid(x, y, valid);
        EXPECT_FALSE(valid);
    }

    TEST(Point_2_8_Test, TestPointTrueValid)
    {
        float x = 12.0f, y = 13.0f;
        bool valid;
        point_test_device::test_point_valid(x, y, valid);
        EXPECT_TRUE(valid);   
    }

    TEST(Point_2_8_Test, TestPointMultScalar)
    {
        float x = 12.0f, y = 13.0f , t = 5.0f;
        float out_x, out_y;
        test_point_mult_scalar(x, y, t, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x * t);
        EXPECT_FLOAT_EQ(out_y, y * t);
    }

    TEST(Point_2_8_Test, TestPointDivScalar)
    {
        float x = 12.0f, y = 13.0f , t = 5.0f;
        float out_x, out_y;
        test_point_div_scalar(x, y, t, out_x, out_y);
        EXPECT_FLOAT_EQ(out_x, x / t);
        EXPECT_FLOAT_EQ(out_y, y / t);
    }

    TEST(Point_2_8_Test, TestPointFalseEquality)
    {
        bool equality;
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.0f, by = 13.1f;
        test_point_equality(ax, ay, bx, by, equality);
        EXPECT_FALSE(equality);
    }

    TEST(Point_2_8_Test, TestPointTrueEquality)
    {
        bool equality;
        float ax = 12.0f, ay = 13.0f;
        float bx = 12.0f, by = 13.0f;
        test_point_equality(ax, ay, bx, by, equality);
        EXPECT_TRUE(equality);
    }

    // A - (0,0) B - (0,2) -> dist = 2
    TEST(Point_2_8_Test, TestPointDistance)
    {
        float dist;
        float ax = 0.0f, ay = 0.0f;
        float bx = 0.0f, by = 2.0f; 
        test_point_distaince(ax, ay, bx, by, dist);
        EXPECT_FLOAT_EQ(dist, 2.0f); 
    }

    //-----------------TEST_INPUT------------------------------------
    int start_point_test(int argc, char **argv) {
        std::cerr << "\033[32m[Start point tests]\033[0m" << '\n';
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}
