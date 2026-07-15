#pragma once

#include <iostream>

#include "city.hpp"
#include "include/memory/cumem.hpp"

using namespace cu_gtsp;

namespace core {
    class tour_2_20_t {
        size_t tour_size_;
        city_2_12_t *citieslist_;

        cu_alloc::unified_allocator<city_2_12_t> alloc_;
        
        float distance_ = -1;
        float fitness_ = -1;

        __host__ __device__ void resize(size_t new_size)
        {
            city_2_12_t *tmp = alloc_.allocate(new_size);
            size_t old_size = tour_size_;

            if (new_size <= old_size) {
                for (size_t i = 0; i < new_size; ++i) {
                    tmp[i] = citieslist_[i];
                }
            }
            else { // new_size > old_size.
                for (size_t i = 0; i < old_size; ++i) {
                    tmp[i] = citieslist_[i];
                }
            }

            alloc_.deallocate(citieslist_);
            citieslist_ = tmp;
            tour_size_ = new_size;
        }
    public:
        __host__ __device__ tour_2_20_t() noexcept : tour_size_(-1), citieslist_(nullptr) {}
        __host__ __device__ tour_2_20_t(size_t tour_size) : tour_size_(tour_size), citieslist_(nullptr) {
            citieslist_ = alloc_.allocate(tour_size_);
        }     
        __host__ __device__ tour_2_20_t(const tour_2_20_t &tour) = delete;
        __host__ __device__ ~tour_2_20_t() {
            alloc_.deallocate(citieslist_);
        }

        __host__ __device__ size_t get_size() const {
            return tour_size_;
        }
        __host__ __device__ float get_distance() const {
            return distance_;
        }
        __host__ __device__ float get_fitness() const {
            return fitness_;
        }
        __host__ __device__ const city_2_12_t *const get_data() const {
            return citieslist_;
        }

        __host__ __device__ tour_2_20_t &operator=(const tour_2_20_t &t)
        {
            if (tour_size_ != t.tour_size_) {
                /* Устанавливает новое значение *this.tour_size_ как t.tour_size_. */
                resize(t.tour_size_); 
            }
            for (size_t i = 0; i < tour_size_; ++i) {
                citieslist_[i] = t.citieslist_[i];
            }
            distance_ = t.distance_;
            fitness_ = t.fitness_;
            return *this;
        }

        __host__ __device__ void set_distance()
        {
            if (citieslist_ == nullptr) { 
                return; 
            }

            distance_ = 0;
            for (size_t i = 1; i < tour_size_; ++i) {
                distance_ += cugtsp_math::distance(citieslist_[i - 1].get_pos(), citieslist_[i].get_pos());
            }
        }

        __host__ __device__ void set_fitness() 
        {
            if (citieslist_ == nullptr) {
                return;
            }

            set_distance();
            if (distance_ != 0) {
                fitness_ = 1/distance_;
            }
            else {
                fitness_ = 0;
            }
        }

        __host__ __device__ void print() const {
            std::cout << "Tour size: " << tour_size_ << '\n'
                      << "Tour distance: " << distance_ << '\n'
                      << "Tour fitness value: " << fitness_ << '\n'
                      << "Cities list:\n";
            for (size_t i = 0; i < tour_size_; ++i) {
                std::cout << citieslist_[i] << '\n';
            }
        }
    };

    __host__ __device__ inline std::ostream &operator<<(std::ostream &out, const tour_2_20_t &tour)
    {
        size_t s = tour.get_size();
        out << "Tour size: " << s << '\n'
            << "Tour distance: " << tour.get_distance() << '\n'
            << "Tour fitness value: " << tour.get_fitness() << '\n' 
            << "Cities list:\n";
        for (size_t i = 0; i < s; ++i) {
            out << tour.get_data()[i] << '\n';
        }
        return out;
    }

    __host__ __device__ inline bool operator==(const tour_2_20_t &ltour, const tour_2_20_t &rtour)
    {
        if (ltour.get_size() != rtour.get_size()) {
            return false;
        }
        if (ltour.get_distance() != rtour.get_distance()) {
            return false;
        }
        if (ltour.get_fitness() != rtour.get_fitness()) {
            return false;
        }

        size_t s = ltour.get_size();
        const city_2_12_t *lcity_ptr = ltour.get_data();
        const city_2_12_t *rcity_ptr = rtour.get_data();
        for (size_t i = 0; i < s; ++i) {
            if (lcity_ptr[i] != rcity_ptr[i]) {
                return false;
            }
        }
        return true;
    } 

    __host__ __device__ inline bool operator<(const tour_2_20_t &ltour, const tour_2_20_t &rtour) {
        return(ltour.get_fitness() > rtour.get_fitness());
    }
}