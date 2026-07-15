#pragma once

namespace point_test_device {
    void test_point_default_constructor(float *out_x, float *out_y);
    void test_point_param_constructor(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_copy_constructor(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_get(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_unary_minus(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_assignment(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_minus_equal(float src_ax, float src_ay, float src_bx, float src_by,
                                float *out_x, float *out_y);
    void test_point_plus_equal(float src_ax, float src_ay, float src_bx, float src_by,
                                float *out_x, float *out_y);
    void test_point_scalar_mult(float src_x, float src_y, float t, float *out_x, float *out_y);
    void test_point_scalar_div(float src_x, float src_y, float t, float *out_x, float *out_y);
    void test_point_minus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                float *out_x, float *out_y);
    void test_point_plus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                               float *out_x, float *out_y);
    void test_point_mult_point(float src_ax, float src_ay, float src_bx, float src_by, 
                               float *out_x, float *out_y);
    void test_point_mult_scalar(float src_x, float src_y, float t, float *out_x, float *out_y);
    void test_point_div_point(float src_ax, float src_ay, float src_bx, float src_by, 
                              float *out_x, float *out_y);
    void test_point_div_scalar(float src_x, float src_y, float t, float *out_x, float *out_y);
    void test_point_valid(float src_x, float src_y, bool *out_valid);
    void test_point_equality(float src_ax, float src_ay, float src_bx, float src_by, bool *out_equal);
    void test_point_distaince(float src_ax, float src_ay, float src_bx, float src_by,
                              float *out_dist);
}