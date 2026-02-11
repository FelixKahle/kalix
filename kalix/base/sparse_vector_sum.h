// Copyright (c) 2026 Felix Kahle.
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

#ifndef KALIX_BASE_SPARSE_VECTOR_SUM_H_
#define KALIX_BASE_SPARSE_VECTOR_SUM_H_

#include <algorithm>
#include <vector>
#include <concepts>
#include <limits>
#include "absl/log/check.h"
#include "kalix/base/compensated_double.h"

namespace kalix
{
    /// @brief Manages high-precision accumulation of a sparse vector.
    ///
    /// This class provides a "scattered" accumulation structure. It maintains a dense
    /// vector of values for constant-time updates and a separate vector of indices
    /// to track non-zero entries. This is particularly efficient for sparse linear
    /// algebra operations where many additions are performed on a subset of vector entries.
    ///
    /// The use of @ref CompensatedDouble ensures that precision is maintained even
    /// when summing many values of varying magnitudes.
    class SparseVectorSum
    {
    public:
        /// @brief Dense storage for the vector components.
        std::vector<CompensatedDouble> values;

        /// @brief List of indices containing non-zero (or sentinel-zero) values.
        std::vector<int64_t> non_zero_indices;

        /// @brief Default constructor.
        KALIX_FORCE_INLINE SparseVectorSum() = default;

        /// @brief Constructs a sparse vector with a specific dimension.
        /// @param dimension The number of elements in the vector.
        explicit KALIX_FORCE_INLINE SparseVectorSum(const int64_t dimension)
        {
            set_dimension(dimension);
        }

        /// @brief Resizes the underlying dense storage.
        /// @param dimension The new dimension of the vector.
        KALIX_FORCE_INLINE void set_dimension(const int64_t dimension)
        {
            values.resize(dimension);
            non_zero_indices.reserve(dimension);
        }

        /// @brief Adds a double value to a specific index in the vector.
        ///
        /// If the index was previously zero, it is added to the non-zero index list.
        /// If the result of the addition is exactly zero, the value is replaced by
        /// @c std::numeric_limits<double>::min() to preserve its presence in the
        /// sparse structure (sentinel logic).
        ///
        /// @param index The vector index to modify.
        /// @param value The value to add.
        KALIX_FORCE_INLINE void add(const int64_t index, const double value)
        {
            DCHECK_GE(index, 0);
            DCHECK_LT(index, static_cast<int64_t>(values.size()));

            if (values[index] != 0.0)
            {
                values[index] += value;
            }
            else
            {
                values[index] = CompensatedDouble(value);
                non_zero_indices.push_back(index);
            }

            // Sentinel logic: Keep the index in non_zero_indices even if the sum is zero
            if (values[index] == 0.0)
            {
                values[index] = CompensatedDouble((std::numeric_limits<double>::min)());
            }
        }

        /// @brief Adds a CompensatedDouble value to a specific index.
        /// @see add(const int64_t, double)
        /// @param index The vector index to modify.
        /// @param value The high-precision value to add.
        void add(const int64_t index, const CompensatedDouble value)
        {
            DCHECK_GE(index, 0);
            DCHECK_LT(index, static_cast<int64_t>(values.size()));

            if (values[index] != 0.0)
            {
                values[index] += value;
            }
            else
            {
                values[index] = value;
                non_zero_indices.push_back(index);
            }

            if (values[index] == 0.0)
            {
                values[index] = CompensatedDouble((std::numeric_limits<double>::min)());
            }
        }

        /// @brief Gets the list of currently active (non-zero) indices.
        /// @return Constant reference to the vector of indices.
        [[nodiscard]] const std::vector<int64_t>& get_non_zeros() const
        {
            return non_zero_indices;
        }

        /// @brief Retrieves the value at a specific index.
        /// @param index The index to query.
        /// @return The double-precision approximation of the value.
        [[nodiscard]] double get_value(const int64_t index) const
        {
            DCHECK_GE(index, 0);
            DCHECK_LT(index, static_cast<int64_t>(values.size()));

            return static_cast<double>(values[index]);
        }

        /// @brief Clears the vector, resetting all values to zero.
        ///
        /// Uses an optimized path: if the vector is very sparse, it only zeroes
        /// active indices. Otherwise, it performs a full dense reset.
        void clear()
        {
            // Performance heuristic for sparse vs dense reset
            // If fewer than 30% of entries are non-zero, zero only those. Otherwise, reset all.
            // This is the same as
            // non_zero_count < 0.3 × total_size
            // but by using integer arithmetic to avoid floating-point division which is slightly
            // less efficient.
            if (10 * non_zero_indices.size() < 3 * values.size())
            {
                for (const int64_t i : non_zero_indices)
                {
                    DCHECK_GE(i, 0);
                    DCHECK_LT(i, static_cast<int64_t>(values.size()));

                    values[i] = CompensatedDouble(0.0);
                }
            }
            else
            {
                values.assign(values.size(), CompensatedDouble(0.0));
            }

            non_zero_indices.clear();
        }

        /// @brief Partitions the non-zero indices based on a predicate.
        ///
        /// Rearranges the @c non_zero_indices such that elements satisfying the
        /// predicate come first.
        ///
        /// @tparam Pred A callable with signature @c bool(int64_t) .
        /// @param pred The predicate to apply to indices.
        /// @return The number of indices that satisfy the predicate.
        template <typename Pred>
            requires std::predicate<Pred, int64_t>
        int64_t partition(Pred&& pred)
        {
            return std::partition(non_zero_indices.begin(), non_zero_indices.end(), pred) - non_zero_indices.begin();
        }

        /// @brief Removes indices from the sparse tracking if they meet a "zero" criteria.
        ///
        /// Iterates through active indices and applies @c isZero. If true, the value
        /// is reset to absolute zero and removed from tracking.
        ///
        /// @tparam IsZero A callable with signature @c bool(int64_t, double) .
        /// @param isZero Predicate to determine if a value should be pruned.
        template <typename IsZero>
            requires std::predicate<IsZero, int64_t, double>
        void cleanup(IsZero&& isZero)
        {
            auto num_nz = static_cast<int64_t>(non_zero_indices.size());

            for (int64_t i = num_nz - 1; i >= 0; --i)
            {
                int64_t pos = non_zero_indices[i];
                DCHECK_GE(pos, 0);
                DCHECK_LT(pos, static_cast<int64_t>(values.size()));

                if (auto val = static_cast<double>(values[pos]); isZero(pos, val))
                {
                    values[pos] = CompensatedDouble(0.0);
                    --num_nz;
                    std::swap(non_zero_indices[num_nz], non_zero_indices[i]);
                }
            }

            non_zero_indices.resize(num_nz);
        }
    };
}

#endif // KALIX_BASE_SPARSE_VECTOR_SUM_H_
