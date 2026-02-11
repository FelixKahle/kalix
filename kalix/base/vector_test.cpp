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

#include <gtest/gtest.h>
#include <vector>
#include <utility>

#include "kalix/base/compensated_double.h"
#include "kalix/base/vector.h"
#include "kalix/base/constants.h"

class VectorTest : public ::testing::Test
{
protected:
    kalix::Vector<double> vec;
    const int64_t kSize = 10;

    void SetUp() override
    {
        vec.setup(kSize);
    }

    void TearDown() override
    {
        vec.clear();
    }
};

TEST_F(VectorTest, Initialization)
{
    EXPECT_EQ(vec.dimension, kSize);
    EXPECT_EQ(vec.non_zero_count, 0);
    EXPECT_EQ(vec.dense_values.size(), static_cast<size_t>(kSize));
    EXPECT_EQ(vec.non_zero_indices.size(), static_cast<size_t>(kSize));
    EXPECT_EQ(vec.should_update_packed_storage, false);
    EXPECT_EQ(vec.synthetic_clock_tick, 0.0);

    for (const auto& val : vec.dense_values)
    {
        EXPECT_DOUBLE_EQ(val, 0.0);
    }
}

TEST_F(VectorTest, ClearSparse)
{
    // Simulate sparse data (count < 30% of size)
    vec.dense_values[1] = 5.0;
    vec.dense_values[3] = 10.0;
    vec.non_zero_indices[0] = 1;
    vec.non_zero_indices[1] = 3;
    vec.non_zero_count = 2;

    vec.clear();

    EXPECT_EQ(vec.non_zero_count, 0);
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 0.0);
    EXPECT_DOUBLE_EQ(vec.dense_values[3], 0.0);
}

TEST_F(VectorTest, ClearDense)
{
    // Simulate dense data (force clear to loop over entire array)
    vec.non_zero_count = 5; // > 30% of 10

    vec.dense_values[0] = 1.0;
    vec.dense_values[9] = 2.0;

    vec.clear();

    EXPECT_EQ(vec.non_zero_count, 0);
    EXPECT_DOUBLE_EQ(vec.dense_values[0], 0.0);
    EXPECT_DOUBLE_EQ(vec.dense_values[9], 0.0);
}

TEST_F(VectorTest, PruneSmallValues)
{
    // Setup values, one of which is tiny
    vec.dense_values[0] = 1.0;
    vec.dense_values[1] = kalix::kTiny * 0.1;
    vec.dense_values[2] = 5.0;

    vec.non_zero_indices[0] = 0;
    vec.non_zero_indices[1] = 1;
    vec.non_zero_indices[2] = 2;
    vec.non_zero_count = 3;

    vec.prune_small_values();

    EXPECT_EQ(vec.non_zero_count, 2);
    // Indices should be packed: 0, 2
    EXPECT_EQ(vec.non_zero_indices[0], 0);
    EXPECT_EQ(vec.non_zero_indices[1], 2);
    // Tiny value must be zeroed
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 0.0);
}

TEST_F(VectorTest, CreatePackedStorage)
{
    vec.dense_values[2] = 10.0;
    vec.dense_values[5] = 20.0;
    vec.non_zero_indices[0] = 2;
    vec.non_zero_indices[1] = 5;
    vec.non_zero_count = 2;

    vec.should_update_packed_storage = true;
    vec.create_packed_storage();

    EXPECT_FALSE(vec.should_update_packed_storage);
    EXPECT_EQ(vec.packed_element_count, 2);

    EXPECT_EQ(vec.packed_indices[0], 2);
    EXPECT_DOUBLE_EQ(vec.packed_values[0], 10.0);

    EXPECT_EQ(vec.packed_indices[1], 5);
    EXPECT_DOUBLE_EQ(vec.packed_values[1], 20.0);
}

TEST_F(VectorTest, RebuildIndicesFromDense)
{
    vec.dense_values[2] = 5.0;
    vec.dense_values[8] = -3.0;
    vec.non_zero_count = -1; // Invalid count state

    vec.rebuild_indices_from_dense();

    EXPECT_EQ(vec.non_zero_count, 2);
    EXPECT_EQ(vec.non_zero_indices[0], 2);
    EXPECT_EQ(vec.non_zero_indices[1], 8);
}

TEST_F(VectorTest, CopyFrom)
{
    kalix::Vector<double> source;
    source.setup(kSize);
    source.dense_values[1] = 42.0;
    source.non_zero_indices[0] = 1;
    source.non_zero_count = 1;
    source.synthetic_clock_tick = 123.456;

    vec.copy_from(&source);

    EXPECT_TRUE(vec == source);
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 42.0);
    EXPECT_EQ(vec.synthetic_clock_tick, 123.456);
}

TEST_F(VectorTest, SquaredEuclideanNorm)
{
    vec.dense_values[1] = 3.0;
    vec.dense_values[2] = 4.0;
    vec.non_zero_indices[0] = 1;
    vec.non_zero_indices[1] = 2;
    vec.non_zero_count = 2;

    // 3^2 + 4^2 = 9 + 16 = 25
    EXPECT_DOUBLE_EQ(vec.squared_euclidean_norm(), 25.0);
}

TEST_F(VectorTest, SaxpyOperation)
{
    // Pivot vector (x)
    kalix::Vector<double> pivot;
    pivot.setup(kSize);
    pivot.dense_values[1] = 2.0;
    pivot.dense_values[3] = 4.0;
    pivot.non_zero_indices[0] = 1;
    pivot.non_zero_indices[1] = 3;
    pivot.non_zero_count = 2;

    // Target vector (y)
    vec.dense_values[1] = 10.0;
    vec.dense_values[2] = 5.0;
    vec.non_zero_indices[0] = 1;
    vec.non_zero_indices[1] = 2;
    vec.non_zero_count = 2;

    // y = y + 0.5 * x
    vec.saxpy(0.5, &pivot);

    // Index 1: 10.0 + 0.5 * 2.0 = 11.0
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 11.0);
    // Index 2: Unchanged
    EXPECT_DOUBLE_EQ(vec.dense_values[2], 5.0);
    // Index 3: 0.0 + 0.5 * 4.0 = 2.0
    EXPECT_DOUBLE_EQ(vec.dense_values[3], 2.0);

    EXPECT_EQ(vec.non_zero_count, 3);
}

TEST_F(VectorTest, EqualityCheck)
{
    kalix::Vector<double> v2;
    v2.setup(kSize);

    EXPECT_TRUE(vec == v2);

    vec.dense_values[0] = 1.0;
    EXPECT_FALSE(vec == v2);

    v2.dense_values[0] = 1.0;
    EXPECT_TRUE(vec == v2);

    vec.synthetic_clock_tick = 1.0;
    EXPECT_FALSE(vec == v2);
}

TEST_F(VectorTest, SubscriptAndAccessors)
{
    // The vector was set up with size 10, so it is NOT empty dimensionally.
    EXPECT_FALSE(vec.empty());
    EXPECT_EQ(vec.dimension, kSize);

    // However, it has no non-zero elements yet
    EXPECT_EQ(vec.non_zero_count, 0);

    // Test Write access via subscript
    vec[0] = 10.5;
    vec[5] = -3.2;

    // Test Read access
    EXPECT_DOUBLE_EQ(vec[0], 10.5);
    EXPECT_DOUBLE_EQ(vec[5], -3.2);

    // Const correctness check (read-only access)
    const auto& const_vec = vec;
    EXPECT_DOUBLE_EQ(const_vec[0], 10.5);
    EXPECT_DOUBLE_EQ(const_vec[5], -3.2);
}

TEST_F(VectorTest, MoveConstructor)
{
    // Setup source
    kalix::Vector<double> source;
    source.setup(kSize);
    source.dense_values[1] = 10.0;
    source.non_zero_indices[0] = 1;
    source.non_zero_count = 1;
    source.synthetic_clock_tick = 55.5;

    // Move
    kalix::Vector<double> dest(std::move(source));

    // Verify Destination
    EXPECT_EQ(dest.dimension, kSize);
    EXPECT_EQ(dest.non_zero_count, 1);
    EXPECT_DOUBLE_EQ(dest.dense_values[1], 10.0);
    EXPECT_EQ(dest.synthetic_clock_tick, 55.5);

    // Verify Source is nuked (dimension=0)
    EXPECT_EQ(source.dimension, 0);
    EXPECT_EQ(source.non_zero_count, 0);
    EXPECT_TRUE(source.dense_values.empty());
    EXPECT_EQ(source.next_link, nullptr);
}

TEST_F(VectorTest, MoveAssignment)
{
    kalix::Vector<double> source;
    source.setup(kSize);
    source.dense_values[2] = 7.0;
    source.non_zero_indices[0] = 2;
    source.non_zero_count = 1;

    // Move into 'vec' (which was setup in SetUp())
    vec = std::move(source);

    EXPECT_DOUBLE_EQ(vec.dense_values[2], 7.0);
    EXPECT_EQ(vec.non_zero_count, 1);

    // Source should be empty
    EXPECT_EQ(source.dimension, 0);
}

TEST_F(VectorTest, SelfMoveAssignment)
{
    // Edge case: v = std::move(v)
    vec.dense_values[0] = 1.0;
    vec.non_zero_count = 1;

    // Suppress self-move warning if compiler flags are aggressive
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
    vec = std::move(vec);
#pragma GCC diagnostic pop

    // Should remain unchanged
    EXPECT_DOUBLE_EQ(vec.dense_values[0], 1.0);
    EXPECT_EQ(vec.dimension, kSize);
}

TEST_F(VectorTest, Iterators)
{
    // 1. Test write via iterator
    for (auto& val : vec)
    {
        val = 1.0;
    }

    // 2. Test read via iterator
    double sum = 0;
    for (const auto& val : vec)
    {
        sum += val;
    }

    EXPECT_DOUBLE_EQ(sum, static_cast<double>(kSize));
    EXPECT_DOUBLE_EQ(vec[0], 1.0);
    EXPECT_DOUBLE_EQ(vec[kSize - 1], 1.0);
}

TEST_F(VectorTest, OperatorPlusEquals)
{
    // vec += other
    kalix::Vector<double> other;
    other.setup(kSize);

    // Setup Other: index 1 = 2.0
    other.dense_values[1] = 2.0;
    other.non_zero_indices[0] = 1;
    other.non_zero_count = 1;

    // Setup Vec: index 1 = 3.0
    vec.dense_values[1] = 3.0;
    vec.non_zero_indices[0] = 1;
    vec.non_zero_count = 1;

    vec += other;

    // Expect 3.0 + 1.0 * 2.0 = 5.0
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 5.0);
    EXPECT_EQ(vec.non_zero_count, 1); // No new fill-in, just update
}

TEST_F(VectorTest, OperatorMinusEquals)
{
    // vec -= other
    kalix::Vector<double> other;
    other.setup(kSize);

    other.dense_values[2] = 5.0;
    other.non_zero_indices[0] = 2;
    other.non_zero_count = 1;

    vec.dense_values[2] = 10.0;
    vec.non_zero_indices[0] = 2;
    vec.non_zero_count = 1;

    vec -= other;

    // Expect 10.0 + (-1.0 * 5.0) = 5.0
    EXPECT_DOUBLE_EQ(vec.dense_values[2], 5.0);
}

TEST_F(VectorTest, OperatorMinusEqualsCancellation)
{
    // Test that -= correctly produces zero (and saxpy logic handles tiny)
    kalix::Vector<double> other;
    other.setup(kSize);

    other.dense_values[5] = 2.0;
    other.non_zero_indices[0] = 5;
    other.non_zero_count = 1;

    vec.dense_values[5] = 2.0;
    vec.non_zero_indices[0] = 5;
    vec.non_zero_count = 1;

    vec -= other; // 2.0 - 2.0 = 0.0

    // Saxpy replaces strict 0.0 (or < tiny) with kHighsZero (which is 0.0 in your test file)
    EXPECT_NEAR(vec.dense_values[5], 0.0, 1e-9);
}

TEST_F(VectorTest, StreamOperator)
{
    // Basic verification that operator<< outputs the correct format
    vec.dense_values[1] = 42.0;
    vec.non_zero_indices[0] = 1;
    vec.non_zero_count = 1;

    std::stringstream ss;
    ss << vec;

    std::string output = ss.str();

    // Check for key components in the output string
    EXPECT_TRUE(output.find("Vector(dim=10, nnz=1)") != std::string::npos);
    EXPECT_TRUE(output.find("(1: 42)") != std::string::npos);
}

TEST_F(VectorTest, CapacityCheck)
{
    // Just verify capacity is accessible and sane
    EXPECT_GE(vec.capacity(), static_cast<size_t>(kSize));
}

TEST_F(VectorTest, ReInitialization)
{
    // dirty the vector first
    vec.dense_values[0] = 1.0;
    vec.non_zero_count = 1;

    // Re-setup with larger size
    vec.setup(20);

    EXPECT_EQ(vec.dimension, 20);
    EXPECT_EQ(vec.non_zero_count, 0);
    EXPECT_EQ(vec.dense_values.size(), 20u);
    // Previous data should be gone/zeroed
    EXPECT_DOUBLE_EQ(vec.dense_values[0], 0.0);
}

TEST_F(VectorTest, CopyAssignment)
{
    kalix::Vector<double> source;
    source.setup(kSize);
    source.dense_values[1] = 99.0;
    source.non_zero_indices[0] = 1;
    source.non_zero_count = 1;

    // Copy assignment (not move)
    vec = source;

    // Target check
    EXPECT_EQ(vec.non_zero_count, 1);
    EXPECT_DOUBLE_EQ(vec.dense_values[1], 99.0);

    // Source integrity check (should remain unchanged)
    EXPECT_EQ(source.non_zero_count, 1);
    EXPECT_DOUBLE_EQ(source.dense_values[1], 99.0);
}

TEST_F(VectorTest, PackEmpty)
{
    // Ensure packing an empty vector doesn't crash and resets flag
    vec.should_update_packed_storage = true;
    vec.create_packed_storage();

    EXPECT_EQ(vec.packed_element_count, 0);
    EXPECT_FALSE(vec.should_update_packed_storage);
}

TEST_F(VectorTest, SaxpyEmptyPivot)
{
    // Adding an empty vector should do nothing
    kalix::Vector<double> pivot;
    pivot.setup(kSize);
    // pivot is empty

    vec.dense_values[0] = 5.0;
    vec.non_zero_indices[0] = 0;
    vec.non_zero_count = 1;

    vec.saxpy(1.0, &pivot);

    EXPECT_DOUBLE_EQ(vec.dense_values[0], 5.0);
    EXPECT_EQ(vec.non_zero_count, 1);
}

TEST_F(VectorTest, RebuildIndicesFullyDense)
{
    // Fill every element
    for (int64_t i = 0; i < kSize; ++i)
    {
        vec.dense_values[i] = static_cast<double>(i + 1);
    }
    vec.non_zero_count = -1; // Invalidate count

    vec.rebuild_indices_from_dense();

    EXPECT_EQ(vec.non_zero_count, kSize);
    for (int64_t i = 0; i < kSize; ++i)
    {
        EXPECT_EQ(vec.non_zero_indices[i], i);
    }
}

class VectorCompensatedTest : public ::testing::Test
{
protected:
    kalix::Vector<kalix::CompensatedDouble> vec;
    const int64_t kSize = 10;

    void SetUp() override
    {
        vec.setup(kSize);
    }

    void TearDown() override
    {
        vec.clear();
    }
};

TEST_F(VectorCompensatedTest, Initialization)
{
    EXPECT_EQ(vec.dimension, kSize);
    EXPECT_EQ(vec.non_zero_count, 0);

    // Verify default value is essentially zero
    for (const auto& val : vec.dense_values)
    {
        EXPECT_DOUBLE_EQ(static_cast<double>(val), 0.0);
    }
}

TEST_F(VectorCompensatedTest, ArithmeticOperations)
{
    // Check if operator+= and basic math works through the Vector template
    kalix::CompensatedDouble val1(10.0);

    vec.dense_values[0] = val1;
    vec.non_zero_indices[0] = 0;
    vec.non_zero_count = 1;

    // Test subscript read/write
    vec[0] += 5.0; // Should use CompensatedDouble::operator+=(double)

    EXPECT_DOUBLE_EQ(static_cast<double>(vec[0]), 15.0);
}

TEST_F(VectorCompensatedTest, SaxpyWithCompensatedDouble)
{
    // Pivot vector (x)
    kalix::Vector<kalix::CompensatedDouble> pivot;
    pivot.setup(kSize);
    pivot.dense_values[1] = kalix::CompensatedDouble(2.0);
    pivot.non_zero_indices[0] = 1;
    pivot.non_zero_count = 1;

    // Target vector (y)
    vec.dense_values[1] = kalix::CompensatedDouble(10.0);
    vec.non_zero_indices[0] = 1;
    vec.non_zero_count = 1;

    // y = y + 0.5 * x
    // 10.0 + 0.5 * 2.0 = 11.0
    vec.saxpy(kalix::CompensatedDouble(0.5), &pivot);

    EXPECT_DOUBLE_EQ(static_cast<double>(vec.dense_values[1]), 11.0);
}

TEST_F(VectorCompensatedTest, PruneSmallValues)
{
    // Test that tiny CompensatedDouble values are correctly pruned.
    // Assuming kalix::kTiny applies to the double representation.

    vec.dense_values[0] = kalix::CompensatedDouble(1.0);
    vec.dense_values[1] = kalix::CompensatedDouble(kalix::kTiny * 0.1);

    vec.non_zero_indices[0] = 0;
    vec.non_zero_indices[1] = 1;
    vec.non_zero_count = 2;

    vec.prune_small_values();

    EXPECT_EQ(vec.non_zero_count, 1);
    EXPECT_EQ(vec.non_zero_indices[0], 0);

    // Check that prune zeroed out the value
    EXPECT_DOUBLE_EQ(static_cast<double>(vec.dense_values[1]), 0.0);
}

TEST_F(VectorCompensatedTest, SquaredEuclideanNorm)
{
    // 3.0^2 + 4.0^2 = 25.0
    vec.dense_values[1] = kalix::CompensatedDouble(3.0);
    vec.dense_values[2] = kalix::CompensatedDouble(4.0);
    vec.non_zero_indices[0] = 1;
    vec.non_zero_indices[1] = 2;
    vec.non_zero_count = 2;

    const kalix::CompensatedDouble norm = vec.squared_euclidean_norm();

    EXPECT_DOUBLE_EQ(static_cast<double>(norm), 25.0);
}

TEST_F(VectorCompensatedTest, CopyFromDoubleVector)
{
    // Test copying FROM a standard double vector TO a CompensatedDouble vector
    kalix::Vector<double> source;
    source.setup(kSize);
    source.dense_values[1] = 42.0;
    source.non_zero_indices[0] = 1;
    source.non_zero_count = 1;

    vec.copy_from(&source);

    EXPECT_EQ(vec.non_zero_count, 1);
    EXPECT_DOUBLE_EQ(static_cast<double>(vec.dense_values[1]), 42.0);
}

TEST_F(VectorCompensatedTest, CopyFromCompensatedVector)
{
    // Test copying FROM a CompensatedDouble vector TO a CompensatedDouble vector
    kalix::Vector<kalix::CompensatedDouble> source;
    source.setup(kSize);
    source.dense_values[1] = kalix::CompensatedDouble(99.0);
    source.non_zero_indices[0] = 1;
    source.non_zero_count = 1;

    vec.copy_from(&source);

    EXPECT_EQ(vec.non_zero_count, 1);
    EXPECT_DOUBLE_EQ(static_cast<double>(vec.dense_values[1]), 99.0);
}
