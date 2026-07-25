#pragma once

#include <iostream>
#include <cstddef> // std::size_t

#include "tour.hpp"
#include "include/memory/cumem.hpp"

using namespace cu_gtsp;

namespace core {
    class alignas(32) population_2_28_t {
        size_t pop_size_;
        tour_2_20_t *tourlist_;

        cu_alloc::unified_allocator<tour_2_20_t> alloc_;

        tour_2_20_t *last_value = nullptr;
    
        __host__ __device__ void resize(std::size_t new_size) 
        {
            tour_2_20_t *buffer = alloc_.allocate(new_size);
            if (!buffer) {
                return;
            }

            size_t old_size = pop_size_;
            std::size_t elem_to_cpy = (new_size < old_size) ? new_size : old_size;

            for (std::size_t i = 0; i < elem_to_cpy; ++i) {
                new (&buffer[i]) tour_2_20_t(tourlist_[i]);
            }
            for (std::size_t i = elem_to_cpy; i < new_size; ++i) {
                new (&buffer[i]) tour_2_20_t();
            }

            /* Освабождение старой памяти. */
            if (tourlist_) { 
                for (std::size_t i = 0; i < old_size; ++i) {
                    tourlist_[i].~tour_2_20_t();
                } 
                alloc_.deallocate(tourlist_);
            }

            tourlist_ = buffer;
            pop_size_ = new_size;
        }
    public:
        __host__ __device__ population_2_28_t() noexcept : pop_size_(0), tourlist_(nullptr) {} 
        __host__ __device__ population_2_28_t(std::size_t pop_size) : pop_size_(pop_size), tourlist_(nullptr)  
        {
            tour_2_20_t *buffer = alloc_.allocate(pop_size_);
            if (buffer) {
                for (std::size_t i = 0; i < pop_size_; ++i) {
                    new (&buffer[i]) tour_2_20_t();
                }
                tourlist_ = buffer;
            }
        }
        __host__ __device__ population_2_28_t(std::size_t pop_size, std::size_t tour_size) :
                                              pop_size_(pop_size), tourlist_(nullptr)  
        {
            tour_2_20_t *buffer = alloc_.allocate(pop_size_);
            if (buffer) {
                for (std::size_t i = 0; i < pop_size_; ++i) {
                    new (&buffer[i]) tour_2_20_t(tour_size);
                }
                tourlist_ = buffer;
            }   
        }
        __host__ __device__ population_2_28_t(const population_2_28_t &pop) = delete;
        __host__ __device__ ~population_2_28_t() {
            alloc_.deallocate(tourlist_);
        }

        __host__ __device__ void push_back(const tour_2_20_t &val)
        {
            // TODO
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

        __host__ __device__ const size_t &get_size() const {
            return pop_size_;
        }
        __host__ __device__ const tour_2_20_t *const get_data() const {
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