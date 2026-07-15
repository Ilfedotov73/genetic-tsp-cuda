#pragma once

#include "gtest/gtest.h"

namespace cumem_test_device {
    //-----------------TEST_INPUT------------------------------------
    int start_cumem_test(int argc, char **argv) {
        std::cerr << "\033[32m[Start point tests]\033[0m" << '\n';
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }  
}