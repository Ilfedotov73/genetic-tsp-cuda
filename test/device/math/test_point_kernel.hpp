#pragma once

namespace device_test {
    void test_point_default_constructor(float *out_x, float *out_y);
    void test_point_param_constructor(float x, float y, float *out_x, float *out_y);

    /*
    void test_point_copy_constructor(const math::point_2_8_t *src, float *out_x, float *out_y);
    void test_point_get(const math::point_2_8_t *src, float *out_x, float *out_y);
    void test_point_unary_minus(const math::point_2_8_t *src, float *out_x, float *out_y);
    void test_point_assignment(const math::point_2_8_t *src, float *out_x, float *out_y);
    void test_point_minus_equal(const math::point_2_8_t *src_a, const math::point_2_8_t *src_b, 
                                    float *out_x, float *out_y);
    void test_point_plus_equal(const math::point_2_8_t *src_a, const math::point_2_8_t *src_b,
                                    float *out_x, float *out_y);
    void test_point_mult_equal(const math::point_2_8_t *src_a, const math::point_2_8_t *src_b,
                                    float *out_x, float *out_y);
    void test_point_div_equal(const math::point_2_8_t *src_a, const math::point_2_8_t *src_b,
                                    float *out_x, float *out_y);
    void test_point_valid(const math::point_2_8_t *src, bool *valid);

    */
}