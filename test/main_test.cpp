/* 
 * g++ -c test/main_test.cpp -o main_test.o -O3
 * g++ main_test.o test_point_kernel.o -o run_tests -lcudart -L/usr/local/cuda-13.3/lib64 -lgtest -lgtest_main 
 */

#include "math/gtest_point.hpp"

int main(int argc, char **argv) {
    point_test_device::start_point_test(argc, argv);
}