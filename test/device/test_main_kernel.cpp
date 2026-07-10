/* 
 * g++ -c test/device/test_main_kernel.cpp -o test_main_kernel.o -O3
 * g++ test_main_kernel.o test_point_kernel.o -o run_tests -lcudart -L/usr/local/cuda-13.3/lib64 -lgtest -lgtest_main 
 */

#include "math/test_point.hpp"

int main(int argc, char **argv) {
    start_point_test(argc, argv);
}