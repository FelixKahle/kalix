// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef KALIX_BASE_VECTOR_H_
#define KALIX_BASE_VECTOR_H_

#include <cstdint>
#include <vector>
#include <iostream>
#include <utility>
#include "kalix/base/config.h"
#include "kalix/base/constants.h"

namespace kalix
{
    /// @brief A hyper-sparse vector implementation for high-performance linear algebra.
    ///
    /// This class maintains both a dense array of values and a list of indices for non-zero entries,
    /// allowing for O(1) random access and O(nnz) iteration. It is optimized for operations
    /// where the vector may be extremely sparse (hyper-sparse), common in linear programming (LP)
    /// and simplex algorithms.
    ///
    /// @tparam Real The floating-point type (e.g., double).
    template <typename Real>
    class Vector
    {
    public:
        /// @brief Pointer to the next vector in a linked list (if used in a pool or factorization).
        Vector* next_link;

        /// @brief Array of indices corresponding to non-zero values in @ref dense_values.
        std::vector<int64_t> non_zero_indices;

        /// @brief Dense array containing the values of the vector.
        ///
        /// Only entries at positions specified by @ref non_zero_indices are guaranteed to be
        /// valid/non-zero during sparse operations.
        std::vector<Real> dense_values;

        /// @brief Packed storage for indices, used during specific linear algebra routines (e.g., PFI).
        std::vector<int64_t> packed_indices;

        /// @brief Packed storage for values, used in conjunction with @ref packed_indices.
        std::vector<Real> packed_values;

        /// @brief Character workspace array for temporary flags or markers.
        std::vector<char> char_workspace;

        /// @brief Integer workspace array for temporary indexing or mapping.
        std::vector<int64_t> integer_workspace;

        /// @brief The total dimension of the vector space.
        int64_t dimension{};

        /// @brief The number of non-zero elements currently tracked.
        int64_t non_zero_count{};

        /// @brief The number of elements currently stored in the packed arrays.
        int64_t packed_element_count{};

        /// @brief A synthetic timestamp or tolerance marker used for structural equality checks.
        double synthetic_clock_tick{};

        /// @brief Flag indicating if the packed arrays need to be updated.
        bool should_update_packed_storage{};

        /// @brief Default constructor.
        Vector() = default;

        /// @brief Move constructor.
        /// Transfers ownership of internal storage from @p other to this vector.
        KALIX_FORCE_INLINE Vector(Vector&& other) noexcept
        {
            *this = std::move(other);
        }

        /// @brief Move assignment operator.
        /// @param other The vector to move from.
        /// @return Reference to this vector.
        KALIX_FORCE_INLINE Vector& operator=(Vector&& other) noexcept
        {
            if (this != &other)
            {
                // Move standard vectors (cheap pointer swap)
                non_zero_indices = std::move(other.non_zero_indices);
                dense_values = std::move(other.dense_values);
                packed_indices = std::move(other.packed_indices);
                packed_values = std::move(other.packed_values);
                char_workspace = std::move(other.char_workspace);
                integer_workspace = std::move(other.integer_workspace);

                // Copy scalars
                dimension = other.dimension;
                non_zero_count = other.non_zero_count;
                packed_element_count = other.packed_element_count;
                synthetic_clock_tick = other.synthetic_clock_tick;
                should_update_packed_storage = other.should_update_packed_storage;
                next_link = other.next_link;

                // Reset other to safe "empty" state
                other.dimension = 0;
                other.non_zero_count = 0;
                other.next_link = nullptr;
            }
            return *this;
        }

        /// @brief Copy constructor.
        KALIX_FORCE_INLINE Vector(const Vector&) = default;

        /// @brief Copy assignment operator.
        KALIX_FORCE_INLINE Vector& operator=(const Vector&) = default;

        /// @brief Returns an iterator to the beginning of the dense array.
        KALIX_FORCE_INLINE auto begin()
        {
            return dense_values.begin();
        }

        /// @brief Returns an iterator to the end of the dense array.
        KALIX_FORCE_INLINE auto end()
        {
            return dense_values.end();
        }

        /// @brief Returns a const iterator to the beginning of the dense array.
        KALIX_FORCE_INLINE auto begin() const
        {
            return dense_values.begin();
        }

        /// @brief Returns a const iterator to the end of the dense array.
        KALIX_FORCE_INLINE auto end() const
        {
            return dense_values.end();
        }

        /// @brief Provides read-write access to the element at the given index.
        /// @param i The index to access.
        KALIX_FORCE_INLINE Real& operator[](const size_t i)
        {
            return dense_values[i];
        }

        /// @brief Provides read-only access to the element at the given index.
        /// @param i The index to access.
        KALIX_FORCE_INLINE const Real& operator[](const size_t i) const
        {
            return dense_values[i];
        }

        /// @brief Checks if the vector dimension is zero.
        [[nodiscard]] KALIX_FORCE_INLINE bool empty() const
        {
            return dimension == 0;
        }

        /// @brief Returns the capacity of the underlying dense storage.
        [[nodiscard]] KALIX_FORCE_INLINE size_t capacity() const
        {
            return dense_values.capacity();
        }

        /// @brief Allocates memory and initializes the vector structure.
        /// @param new_dimension The dimension of the vector space.
        KALIX_FORCE_INLINE void setup(const int64_t new_dimension)
        {
            dimension = new_dimension;
            non_zero_count = 0;
            non_zero_indices.resize(new_dimension);
            dense_values.assign(new_dimension, Real{0});
            char_workspace.assign(new_dimension + 6400, 0); // Allocation includes workspace padding
            integer_workspace.assign(new_dimension * 4, 0);

            packed_element_count = 0;
            packed_indices.resize(new_dimension);
            packed_values.resize(new_dimension);

            should_update_packed_storage = false;
            synthetic_clock_tick = 0;
            next_link = nullptr;
        }

        /// @brief Resets the vector to zero.
        ///
        /// Uses a heuristic to determine the most efficient clearing method. If the vector
        /// is sparse (< 30% filled), it iterates over indices to zero them. Otherwise,
        /// it performs a full memset/assign on the dense array.
        KALIX_FORCE_INLINE void clear()
        {
            if (non_zero_count < 0 || non_zero_count > dimension * 0.3)
            {
                dense_values.assign(dimension, Real{0});
            }
            else
            {
                for (int64_t i = 0; i < non_zero_count; i++)
                {
                    dense_values[non_zero_indices[i]] = 0;
                }
            }

            clear_scalars();
        }

        /// @brief Resets scalar members and flags without clearing the data arrays.
        KALIX_FORCE_INLINE void clear_scalars()
        {
            should_update_packed_storage = false;
            non_zero_count = 0;
            synthetic_clock_tick = 0;
            next_link = 0;
        }

        /// @brief Filters out values smaller than @ref kTiny and repacks indices.
        ///
        /// If @ref non_zero_count is negative, it scans the entire dense array to rebuild
        /// the index list, treating values < kTiny as zero.
        KALIX_FORCE_INLINE void prune_small_values()
        {
            if (non_zero_count < 0)
            {
                for (auto& val : dense_values)
                {
                    if (std::abs(val) < kTiny)
                    {
                        val = 0;
                    }
                }
            }
            else
            {
                int64_t current_count = 0;
                for (int64_t i = 0; i < non_zero_count; i++)
                {
                    const int64_t index = non_zero_indices[i];
                    if (const Real& value = dense_values[index]; std::abs(value) >= kTiny)
                    {
                        non_zero_indices[current_count++] = index;
                    }
                    else
                    {
                        dense_values[index] = Real{0};
                    }
                }
                non_zero_count = current_count;
            }
        }

        /// @brief Packs the current non-zero values into contiguous memory.
        ///
        /// Populates @ref packed_indices and @ref packed_values based on the current sparse structure.
        /// Only performs work if @ref should_update_packed_storage is true.
        KALIX_FORCE_INLINE void create_packed_storage()
        {
            if (!should_update_packed_storage)
            {
                return;
            }

            should_update_packed_storage = false;
            packed_element_count = 0;

            for (int64_t i = 0; i < non_zero_count; i++)
            {
                const int64_t index = non_zero_indices[i];
                packed_indices[packed_element_count] = index;
                packed_values[packed_element_count] = dense_values[index];
                packed_element_count++;
            }
        }

        /// @brief Rebuilds the sparse index list from the dense array.
        ///
        /// Typically used when the sparse structure has been invalidated or if the
        /// vector was populated via direct dense access.
        KALIX_FORCE_INLINE void rebuild_indices_from_dense()
        {
            if (non_zero_count >= 0 && non_zero_count <= dimension * 0.1)
            {
                return;
            }

            non_zero_count = 0;
            for (int64_t i = 0; i < dimension; i++)
            {
                if (static_cast<double>(dense_values[i]))
                {
                    non_zero_indices[non_zero_count++] = i;
                }
            }
        }

        /// @brief Deep copies data from another vector, potentially casting types.
        /// @tparam FromReal The numeric type of the source vector.
        /// @param source Pointer to the source vector.
        template <typename FromReal>
        KALIX_FORCE_INLINE void copy_from(const Vector<FromReal>* source)
        {
            clear();

            synthetic_clock_tick = source->synthetic_clock_tick;
            const int64_t source_count = non_zero_count = source->non_zero_count;
            const int64_t* source_indices = &source->non_zero_indices[0];
            const FromReal* source_values = &source->dense_values[0];

            for (int64_t i = 0; i < source_count; i++)
            {
                const int64_t index = source_indices[i];
                const FromReal value = source_values[index];
                non_zero_indices[i] = index;
                dense_values[index] = Real(value);
            }
        }

        /// @brief Computes the squared Euclidean norm (L2-norm squared) of the vector.
        /// @return The sum of squares of the vector elements.
        KALIX_FORCE_INLINE Real squared_euclidean_norm() const
        {
            const int64_t count_local = non_zero_count;
            const int64_t* indices_local = &non_zero_indices[0];
            const Real* values_local = &dense_values[0];

            Real result = Real{0};
            for (int64_t i = 0; i < count_local; i++)
            {
                Real value = values_local[indices_local[i]];
                result += value * value;
            }
            return result;
        }

        /// @brief Performs the sparse AXPY operation: y = y + alpha * x.
        ///
        /// This method adds a scaled version of the source vector to this vector.
        /// It efficiently handles sparsity by iterating only over the non-zeros of the source.
        ///
        /// @tparam RealScalar Type of the scalar alpha.
        /// @tparam RealVector Type of the source vector elements.
        /// @param multiplier The scalar alpha multiplier.
        /// @param vector_to_add The vector x to add.
        template <typename RealScalar, typename RealVector>
        KALIX_FORCE_INLINE void saxpy(const RealScalar multiplier, const Vector<RealVector>* vector_to_add)
        {
            int64_t current_count = non_zero_count;
            int64_t* current_indices = &non_zero_indices[0];
            Real* current_values = &dense_values[0];

            const int64_t add_count = vector_to_add->non_zero_count;
            const int64_t* add_indices = &vector_to_add->non_zero_indices[0];
            const RealVector* add_values = &vector_to_add->dense_values[0];

            for (int64_t k = 0; k < add_count; k++)
            {
                const int64_t row_index = add_indices[k];
                const Real original_value = current_values[row_index];
                const Real new_value = Real(original_value + multiplier * add_values[row_index]);

                // If previous value was zero, we have a new non-zero entry
                if (original_value == Real{0})
                {
                    current_indices[current_count++] = row_index;
                }

                // Tiny values are flushed to kTiny (symbolic zero)
                current_values[row_index] = (std::abs(new_value) < kTiny) ? kZero : new_value;
            }
            non_zero_count = current_count;
        }

        /// @brief Checks structural equality with another vector.
        /// @param other The vector to compare against.
        /// @return True if dimension, count, indices, values, and synthetic properties match.
        KALIX_FORCE_INLINE bool operator==(const Vector<Real>& other) const
        {
            if (dimension != other.dimension)
            {
                return false;
            }
            if (non_zero_count != other.non_zero_count)
            {
                return false;
            }
            if (non_zero_indices != other.non_zero_indices)
            {
                return false;
            }
            if (dense_values != other.dense_values)
            {
                return false;
            }
            if (synthetic_clock_tick != other.synthetic_clock_tick)
            {
                return false;
            }
            return true;
        }

        /// @brief Checks structural inequality with another vector.
        KALIX_FORCE_INLINE bool operator!=(const Vector<Real>& other) const
        {
            return !(*this == other);
        }

        /// @brief In-place addition of another vector.
        /// Calls @ref saxpy with alpha = 1.0.
        KALIX_FORCE_INLINE Vector& operator+=(const Vector<Real>& other)
        {
            this->saxpy(Real{1}, &other);
            return *this;
        }

        /// @brief In-place subtraction of another vector.
        /// Calls @ref saxpy with alpha = -1.0.
        KALIX_FORCE_INLINE Vector& operator-=(const Vector<Real>& other)
        {
            this->saxpy(Real{-1}, &other);
            return *this;
        }

        /// @brief Stream output operator for debugging.
        /// Prints the vector dimension, count, and non-zero entries.
        friend KALIX_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const Vector<Real>& v)
        {
            os << "Vector(dim=" << v.dimension << ", nnz=" << v.non_zero_count << ") {\n";
            os << "  Non-zeros: [";
            for (int64_t i = 0; i < v.non_zero_count; ++i)
            {
                int64_t idx = v.non_zero_indices[i];
                os << "(" << idx << ": " << v.dense_values[idx] << ")";
                if (i < v.non_zero_count - 1) os << ", ";
            }
            os << "]\n}";
            return os;
        }
    };
}

#endif // KALIX_BASE_VECTOR_H_
