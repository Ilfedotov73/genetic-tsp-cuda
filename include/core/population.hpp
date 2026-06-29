#pragma once

#include <iostream>

#include "tour.hpp"
#include "include/manage_memroy/memory.hpp"

namespace core {
    class alignas(32) population_2_28_t {
        size_t pop_size_;
        tour_2_20_t *tourlist_;

        manage_memory::allocator<tour_2_20_t> alloc_;
    
        __host__ __device__ void resize(size_t new_size) 
        {
            tour_2_20_t *tmp = alloc_.allocate(new_size);
            size_t old_size = pop_size_;

            if (new_size <= old_size) {
                for (size_t i = 0; i < new_size; ++i) {
                    tmp[i] = tourlist_[i];
                }
            }
            else { // new_size > old_size.
                for (size_t i = 0; i < old_size; ++i) {
                    tmp[i] = tourlist_[i];
                }
            }

            alloc_.deallocate(tourlist_);
            tourlist_ = tmp;
            pop_size_ = new_size;
        }
    public:
        __host__ __device__ population_2_28_t() : pop_size_(-1), tourlist_(nullptr) {}
        __host__ __device__ population_2_28_t(size_t pop_size) : pop_size_(pop_size), tourlist_(nullptr) {
            tourlist_ = alloc_.allocate(pop_size_);
        }
        __host__ __device__ population_2_28_t(const population_2_28_t &pop) = delete;
        __host__ __device__ ~population_2_28_t() {
            alloc_.deallocate(tourlist_);
        }

        __host__ __device__ population_2_28_t &operator=(const population_2_28_t &p)
        {
            if (pop_size_ != p.pop_size_) {
                /* Устанавливает новое значение *this.pop_size_ как p.pop_size_. */
                resize(p.pop_size_);
            }
            for (size_t i = 0; i < pop_size_; ++i) {
                tourlist_[i] = p.tourlist_[i];
            }
            return *this;
        }

        __host__ __device__ size_t get_size() const {
            return pop_size_;
        }
        __host__ __device__ const tour_2_20_t *const get_const_pop_ptr() const {
            return tourlist_;
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

    __host__ __device__ inline std::ostream &operator<<(std::ostream &out, const population_2_28_t &pop)
    {
        size_t size = pop.get_size();
        out << "Population:\n" 
            << " * Population size: " << pop.get_size() << '\n'
            << " * Populaton composition:\n";
        for (size_t i = 0; i < size; ++i) {
            out << " * * "<< pop.get_const_pop_ptr()[i] << '\n';
        }   
        return out;      
    }

    __host__ __device__ inline bool operator==(const population_2_28_t &lpop, const population_2_28_t &rpop) 
    {
        if (lpop.get_size() != rpop.get_size()) {
            return false;
        }

        size_t s = lpop.get_size();
        const tour_2_20_t *ltour_ptr = lpop.get_const_pop_ptr();
        const tour_2_20_t *rtour_ptr = rpop.get_const_pop_ptr();
        for (size_t i = 0; i < s ;++i) {
            if (ltour_ptr[i] != rtour_ptr[i]) {
                return false;
            }
        }
        return true;
    }
}