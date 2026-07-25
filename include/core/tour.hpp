#pragma once

#include <iostream>
#include <cfloat>
#include <cstddef> // std::size_t

#include "city.hpp"
#include "include/memory/cumem.hpp"

using namespace cu_gtsp;

namespace core {
    class tour_2_20_t {
        size_t tour_size_;
        city_2_12_t *citieslist_; // POD

        cu_alloc::unified_allocator<city_2_12_t> alloc_;
        
        float distance_ = -1;
        float fitness_ = -1;

        __host__ __device__ void deallocate_citieslist() 
        {
            if (citieslist_ != nullptr) {
                alloc_.deallocate(citieslist_);
                citieslist_ = nullptr;
            }
        }

        __host__ __device__ void invalidate_cache()
        {
            distance_ = -1.0f;
            fitness_ = -1.0f;
        }

        __host__ __device__ void set_distance()
        {
            if (citieslist_ == nullptr) { 
                return; 
            }

            distance_ = 0;
            for (std::size_t i = 1; i < tour_size_; ++i) {
                distance_ += cugtsp_math::distance(citieslist_[i - 1].get_pos(), citieslist_[i].get_pos());
            }
            distance_ += cugtsp_math::distance(citieslist_[tour_size_ - 1].get_pos(), citieslist_[0].get_pos());
        }

        __host__ __device__ bool resize(std::size_t new_size)
        {
            if (new_size == 0) {
                deallocate_citieslist();
                tour_size_ = 0;
                return false;
            }

            city_2_12_t *buffer = alloc_.allocate(new_size);
            if (buffer) {
                /**
                 * Если this->citieslist_ == nullptr и new_size != 0, то функция resize() приведет
                 * this->citieslist_ в валидное состояние, заполнив его дефолтными значениями.
                 */
                std::size_t elem_to_cpy = (citieslist_ != nullptr) ? (
                    (new_size < tour_size_) ? new_size : tour_size_
                ) : 0;
            
                for (std::size_t i = 0; i < elem_to_cpy; ++i) {
                    buffer[i] = citieslist_[i];
                }
                for (std::size_t i = elem_to_cpy; i < new_size; ++i) {
                    buffer[i] = city_2_12_t();
                }
            
                deallocate_citieslist();
                citieslist_ = buffer;
                tour_size_ = new_size;
            }
            else {
                return false;
            }
            return true;
        }
    public:
        __host__ __device__ tour_2_20_t() noexcept : tour_size_(0), citieslist_(nullptr) {} 
        __host__ __device__ tour_2_20_t(std::size_t tour_size) : tour_size_(tour_size), citieslist_(nullptr)  
        {
            if (tour_size_ > 0) {
                citieslist_ = alloc_.allocate(tour_size_);
                for (std::size_t i = 0; i < tour_size_; ++i) {
                    citieslist_[i] = city_2_12_t();
                }
            }
        }    
        __host__ __device__ tour_2_20_t(const tour_2_20_t &other) : tour_size_(other.tour_size_), citieslist_(nullptr),
                                                                    distance_(other.distance_), fitness_(other.fitness_)
        {
            if (tour_size_ > 0) {
                citieslist_ = alloc_.allocate(tour_size_);
                if (citieslist_) {
                    for (std::size_t i = 0; i < tour_size_; ++i) {
                        citieslist_[i] = other.citieslist_[i];
                    }
                }
            }
        }
        __host__ __device__ ~tour_2_20_t() {
            deallocate_citieslist();
        }
       
        __host__ __device__ void push_back(const city_2_12_t &val)
        {
            std::size_t old_size = tour_size_;
            bool is_ok = resize(tour_size_ + 1);
            if (!is_ok) {
                return;
            }

            if (citieslist_ && tour_size_ == old_size + 1) {
                citieslist_[old_size] = val;
            }
            invalidate_cache();
        }

        __host__ __device__ const size_t &get_size() const noexcept {
            return tour_size_;
        }
        __host__ __device__ const float &get_distance() const noexcept {
            return distance_;
        }
        __host__ __device__ const float &get_fitness() const noexcept {
            return fitness_;
        }
        __host__ __device__ const city_2_12_t *const get_const_data() const noexcept {
            return citieslist_;
        }

        __host__ __device__ tour_2_20_t &operator=(const tour_2_20_t &other)
        {
            if (this == &other) { 
                return *this;
            }
            if (tour_size_ != other.tour_size_) {
                /* Устанавливает новое значение this->tour_size_ как other.tour_size_. */
                bool is_ok = resize(other.tour_size_); 
                if (!is_ok) {
                    return *this;
                }
            }
            if (citieslist_) {
                for (std::size_t i = 0; i < tour_size_; ++i) {
                    citieslist_[i] = other.citieslist_[i];
                }
            }
            distance_ = other.distance_;
            fitness_ = other.fitness_;
            return *this;
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
                fitness_ = FLT_MAX;
            }
        }

        __host__ void print() const {
            std::cout << "Tour size: " << tour_size_ << '\n'
                      << "Tour distance: " << distance_ << '\n'
                      << "Tour fitness value: " << fitness_ << '\n'
                      << "Cities list:\n";
            for (std::size_t i = 0; i < tour_size_; ++i) {
                std::cout << citieslist_[i] << '\n';
            }
        }
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const tour_2_20_t &tour)
    {

        std:: size_t s = tour.get_size();
        out << "Tour size: " << s << '\n'
            << "Tour distance: " << tour.get_distance() << '\n'
            << "Tour fitness value: " << tour.get_fitness() << '\n' 
            << "Cities list:\n";
        for (std::size_t i = 0; i < s; ++i) {
            out << tour.get_const_data()[i] << '\n';
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

        std::size_t s = ltour.get_size();
        const city_2_12_t *lcity_ptr = ltour.get_const_data();
        const city_2_12_t *rcity_ptr = rtour.get_const_data();
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
} // namespace core