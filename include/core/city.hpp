#pragma once

#include <iostream>

#include "include/math/point.hpp"

namespace core {
    class alignas(16) city_2_12_t {
        int id_;
        cugtsp_math::point_2_8_t p_;
    public:
        __host__ __device__ city_2_12_t() noexcept : id_(-1), p_() {} // p_{NAN;NAN}
        __host__ __device__ city_2_12_t(int id, const cugtsp_math::point_2_8_t &p) : id_(id), p_(p) {}
        __host__ __device__ city_2_12_t(const city_2_12_t &city) : id_(city.id_), p_(city.p_) {}

        __host__ __device__ const int &get_id() const {
            return id_;
        }
        __host__ __device__ const cugtsp_math::point_2_8_t &get_pos() const {
            return p_;
        }

        __host__ __device__ city_2_12_t &operator=(const city_2_12_t &c) 
        {
            id_ = c.id_;
            p_ = c.p_;
            return *this;
        }

        __host__ void print() const
        {
            std::cout << "City id: " << id_ << '\n' 
                      << "City pos: " << p_;
        }
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const city_2_12_t &city) 
    {
        return out << "City id: " << city.get_id() << '\n' 
                   << "City pos: " << city.get_pos();
    }

    __host__ __device__ inline bool operator==(const city_2_12_t &lcity, const city_2_12_t &rcity) {
        return ((lcity.get_pos() == rcity.get_pos()) && (lcity.get_id() == rcity.get_id()));
    }
}