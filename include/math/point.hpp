#pragma once

#include <iostream>

namespace cugtsp_math {
    struct alignas(8) point_2_8_t {
        float x_, y_;

        __host__ __device__ point_2_8_t() noexcept : x_(NAN), y_(NAN) {}        
        __host__ __device__ point_2_8_t(float x, float y) noexcept : x_(x), y_(y) {}
        __host__ __device__ point_2_8_t(const point_2_8_t &p) : x_(p.x_), y_(p.y_) {}
        
        __host__ __device__ const float &x() const { 
            return x_; 
        }
        __host__ __device__ const float &y() const { 
            return y_; 
        }
        
        __host__ __device__ point_2_8_t operator-() const { 
            return point_2_8_t(-x_, -y_); 
        }

        __host__ __device__ point_2_8_t &operator=(const point_2_8_t &v)
        {
            x_ = v.x_;
            y_ = v.y_;
            return *this;
        }
        __host__ __device__ point_2_8_t &operator-=(const point_2_8_t &v)
        {
            x_ -= v.x_;
            y_ -= v.y_;
            return *this;
        }
        __host__ __device__ point_2_8_t &operator+=(const point_2_8_t &v) 
        {
            x_ += v.x_;
            y_ += v.y_;
            return *this;
        }
        __host__ __device__ point_2_8_t &operator*=(float t)
        {
            x_ *= t;
            y_ *= t;
            return *this;
        }
        __host__ __device__ point_2_8_t &operator/=(float t) { 
            return *this *= 1.0f / t; 
        }

        __host__ __device__ bool valid() const {
            return !(x_ != x_ || y_ != y_);
        }

        __host__ void print() const {
            std::cout << '(' << x_ << ';' << y_ << ')';
        }
    };

    __host__ __device__ inline point_2_8_t operator-(const point_2_8_t &u, const point_2_8_t &v)
    {
        return point_2_8_t(
            u.x_ - v.x_,
            u.y_ - v.y_
        );
    }
    __host__ __device__ inline point_2_8_t operator+(const point_2_8_t &u, const point_2_8_t &v)
    {
        return point_2_8_t(
            u.x_ + v.x_,
            u.y_ + v.y_
        );
    }
    __host__ __device__ inline point_2_8_t operator*(const point_2_8_t &u, const point_2_8_t &v)
    {
        return point_2_8_t(
            u.x_ * v.x_,
            u.y_ * v.y_
        );
    }
    __host__ __device__ inline point_2_8_t operator*(float t, const point_2_8_t &u)
    {
        return point_2_8_t(
            u.x_ * t, 
            u.y_ * t
        );
    }
    __host__ __device__ inline point_2_8_t operator*(const point_2_8_t &u, float t) {
        return t * u;
    }
    __host__ __device__ inline point_2_8_t operator/(const point_2_8_t &u, const point_2_8_t &v)
    {
        return point_2_8_t(
            u.x_ / v.x_,
            u.y_ / v.y_
        );
    }
    __host__ __device__ inline point_2_8_t operator/(const point_2_8_t &u, float t) {
        return (1.0f / t) * u;
    }
    __host__ __device__ inline bool operator==(const point_2_8_t &u, const point_2_8_t &v) {
        return ((u.x_ == v.x_) && (u.y_ == v.y_));
    }

    __host__ __device__ inline float distance(const point_2_8_t &u, const point_2_8_t &v)
    {
        float dx = u.x_ - v.x_;
        float dy = u.y_ - v.y_;
        return(sqrtf(dx * dx + dy * dy));
    }

    __host__ inline std::ostream &operator<<(std::ostream &out, const point_2_8_t &p) {
        return out << '(' << p.x_ << ';' << p.y_ << ')' << std::endl;
    }
}