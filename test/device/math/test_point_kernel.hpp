#pragma once

namespace device_test {
    void test_point_default_constructor(float *out_x, float *out_y);
    void test_point_param_constructor(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_copy_constructor(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_get(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_unary_minus(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_assignment(float src_x, float src_y, float *out_x, float *out_y);
    void test_point_minus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                                float *out_x, float *out_y);
    void test_point_plus_point(float src_ax, float src_ay, float src_bx, float src_by, 
                               float *out_x, float *out_y);
    void test_point_mult_point(float src_ax, float src_ay, float src_bx, float src_by, 
                               float *out_x, float *out_y);
    void test_point_div_point(float src_ax, float src_ay, float src_bx, float src_by, 
                              float *out_x, float *out_y);
    // void test_point_valid(const cugtsp_math::point_2_8_t *src, bool *valid);
}