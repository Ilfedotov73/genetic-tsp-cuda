#pragma once

#include <iostream>
#include <cfloat>
#include <cstddef> // std::size_t

#include "tour.hpp"
#include "include/memory/cumem.hpp"

using namespace cu_gtsp;

namespace core {
    class alignas(32) population_2_28_t {
        size_t pop_size_;
        tour_2_20_t *tourlist_;

        cu_alloc::unified_allocator<tour_2_20_t> alloc_;
    
        __host__ __device__ void deallocate_tourlist()
        {
            if (tourlist_ != nullptr) {
                alloc_.deallocate(tourlist_);
                tourlist_ = nullptr;
            }
        }

        __host__ __device__ bool resize(std::size_t new_size) 
        {
            if (new_size == 0) {
                deallocate_tourlist();
                pop_size_ = 0;
                return false;
            }

            tour_2_20_t *buffer = alloc_.allocate(new_size);
            if (buffer) {
                /**
                 * Если this->tourlist_ == nullptr и new_size != 0, то функия resize() приведет
                 * this->tourlist_ в валидное состояние, заполнив его дефолтными значениями.
                 */
                std::size_t elem_to_cpy = (tourlist_ != nullptr) ? (
                    (new_size < pop_size_) ? new_size : pop_size_
                ) : 0;
    
                for (std::size_t i = 0; i < elem_to_cpy; ++i) {
                    new (&buffer[i]) tour_2_20_t(tourlist_[i]);
                }
                for (std::size_t i = elem_to_cpy; i < new_size; ++i) {
                    new (&buffer[i]) tour_2_20_t();
                }

                deallocate_tourlist();
                tourlist_ = buffer;
                pop_size_ = new_size;
            }
            else {
                return false;
            }
            return true;
        }
    public:
        __host__ __device__ population_2_28_t() noexcept : pop_size_(0), tourlist_(nullptr) {} 
        __host__ __device__ population_2_28_t(std::size_t pop_size) : pop_size_(pop_size), tourlist_(nullptr)  
        {
            if (pop_size_ > 0) {
                tourlist_ = alloc_.allocate(pop_size_);
                for (std::size_t i = 0; i < pop_size_; ++i) {
                    new (&tourlist_[i]) tour_2_20_t();
                }
            }
        }
        __host__ __device__ population_2_28_t(std::size_t pop_size, std::size_t tour_size) :
                                              pop_size_(pop_size), tourlist_(nullptr)  
        {
            if (pop_size_ > 0) {
                tourlist_ = alloc_.allocate(pop_size_);
                for (std::size_t i = 0; i < pop_size_; ++i) {
                    new (&tourlist_[i]) tour_2_20_t(tour_size);
                }
            }
        }
        __host__ __device__ population_2_28_t(const population_2_28_t &other) = delete;
        __host__ __device__ ~population_2_28_t() {
            deallocate_tourlist();
        }

        __host__ __device__ void push_back(const tour_2_20_t &val)
        {
            std::size_t old_size = pop_size_;
            bool is_ok = resize(pop_size_ + 1);
            if (!is_ok) {
                return;
            }

            if (tourlist_ && pop_size_ == old_size + 1) {
                tourlist_[old_size] = val;
            }
        }

        __host__ __device__ const size_t &get_size() const noexcept {
            return pop_size_;
        }
        __host__ __device__ const tour_2_20_t *const get_data() const noexcept {
            return tourlist_;
        } 

        __host__ __device__ population_2_28_t &operator=(const population_2_28_t &other)
        {
            if (this == &other) {
                return *this;
            }
            if (pop_size_ != other.pop_size_) {
                /* Устанавливает новое значение this.pop_size_ как p.pop_size_. */
                bool is_ok = resize(other.pop_size_);
                if (!is_ok) {
                    return *this;
                }
            }
            if (tourlist_) {
                for (size_t i = 0; i < pop_size_; ++i) {
                    tourlist_[i] = other.tourlist_[i];
                }
            }
            return *this;
        }

        __host__ void print() const {
            std::cout << "Population:\n" 
                      << " * Population size: " << pop_size_ << '\n'
                      << " * Populaton composition:\n";
            for (size_t i = 0; i < pop_size_; ++i) {
                std::cout << " * * "<< tourlist_[i] << '\n';
            }     
        }
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const population_2_28_t &pop)
    {
        size_t size = pop.get_size();
        out << "Population:\n" 
            << " * Population size: " << pop.get_size() << '\n'
            << " * Populaton composition:\n";
        for (size_t i = 0; i < size; ++i) {
            out << " * * "<< pop.get_data()[i] << '\n';
        }   
        return out;      
    }

    __host__ __device__ inline bool operator==(const population_2_28_t &lpop, const population_2_28_t &rpop) 
    {
        if (lpop.get_size() != rpop.get_size()) {
            return false;
        }

        size_t s = lpop.get_size();
        const tour_2_20_t *ltour_ptr = lpop.get_data();
        const tour_2_20_t *rtour_ptr = rpop.get_data();
        for (size_t i = 0; i < s ;++i) {
            if (ltour_ptr[i] != rtour_ptr[i]) {
                return false;
            }
        }
        return true;
    }
}