#pragma once

#include <vector>
#include <cstddef> // std::size_t

namespace cumem_test_device {
    template <typename T, std::size_t N> 
    void cumem_test_device::test_unified_allocator_allocate_seq(T (&host_seq)[N], T increment);
}