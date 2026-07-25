#pragma once

#include <vector>

#include "test_cumem_kernel.hpp"
#include "gtest/gtest.h"

namespace cumem_test_device {
    TEST(Cumem, TestUnifiedAllocatorAllocate)
    {
        std::vector<float> host_seq = { 1.0f, 1.0f, 1.0f, 1.0f};
        test_unified_allocator_allocate_seq<float>(host_seq, 1.0f);
    }

    //-----------------TEST_INPUT------------------------------------
    int start_cumem_test(int argc, char **argv) {
        std::cerr << "\033[32m[Start cumem tests]\033[0m" << '\n';
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }  
}