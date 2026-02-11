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
#include <limits>
#include "kalix/base/sparse_vector_sum.h"

class SparseVectorSumTest : public ::testing::Test
{
protected:
    const int64_t kDimension = 100;
};

TEST_F(SparseVectorSumTest, BasicAdditionAndRetrieval)
{
    kalix::SparseVectorSum svc(kDimension);

    svc.add(10, 5.5);
    svc.add(20, 10.2);

    EXPECT_DOUBLE_EQ(svc.get_value(10), 5.5);
    EXPECT_DOUBLE_EQ(svc.get_value(20), 10.2);
    EXPECT_DOUBLE_EQ(svc.get_value(30), 0.0); // Unset index

    // Check sparsity tracking
    const auto& nzs = svc.get_non_zeros();
    EXPECT_EQ(nzs.size(), 2);
    EXPECT_EQ(nzs[0], 10);
    EXPECT_EQ(nzs[1], 20);
}

TEST_F(SparseVectorSumTest, AccumulatedPrecision)
{
    kalix::SparseVectorSum svc(kDimension);

    // Test that CompensatedDouble logic is working through the sparse vector
    constexpr double large = 1.0;
    constexpr double small = 1e-18;

    svc.add(5, large);
    svc.add(5, small);

    // Standard double would lose 'small'.
    // If we subtract 'large', we should see 'small' again.
    svc.add(5, -large);

    EXPECT_NEAR(svc.get_value(5), small, 1e-25);
}

TEST_F(SparseVectorSumTest, ZeroSentinelLogic)
{
    kalix::SparseVectorSum svc(kDimension);

    // Adding 5.0 and -5.0 results in 0.0
    svc.add(42, 5.0);
    svc.add(42, -5.0);

    // The library logic should replace 0.0 with the smallest possible double
    // to keep the index in the non-zero list.
    EXPECT_EQ(svc.get_value(42), std::numeric_limits<double>::min());
    EXPECT_EQ(svc.get_non_zeros().size(), 1);
}

TEST_F(SparseVectorSumTest, ClearFunctionality)
{
    kalix::SparseVectorSum svc(kDimension);

    svc.add(1, 1.0);
    svc.add(50, 2.0);
    svc.clear();

    EXPECT_EQ(svc.get_non_zeros().size(), 0);
    EXPECT_DOUBLE_EQ(svc.get_value(1), 0.0);
    EXPECT_DOUBLE_EQ(svc.get_value(50), 0.0);
}

TEST_F(SparseVectorSumTest, CleanupZeroValues)
{
    kalix::SparseVectorSum svc(kDimension);

    svc.add(10, 1.0);
    svc.add(20, 2.0);
    svc.add(30, 0.0000000001); // Effectively zero for our predicate

    // Provide a predicate that treats values < 1e-5 as zero
    svc.cleanup([]([[maybe_unused]] int64_t index, const double val)
    {
        return std::abs(val) < 1e-5;
    });

    const auto& nzs = svc.get_non_zeros();
    EXPECT_EQ(nzs.size(), 2);
    // Index 30 should have been removed and its value reset to 0.0
    EXPECT_DOUBLE_EQ(svc.get_value(30), 0.0);
}

TEST_F(SparseVectorSumTest, Partitioning)
{
    kalix::SparseVectorSum svc(kDimension);

    svc.add(10, 1.0);
    svc.add(20, 10.0);
    svc.add(30, 2.0);
    svc.add(40, 15.0);

    // Partition indices where the value is > 5.0
    const int64_t split = svc.partition([&svc](const int64_t idx)
    {
        return svc.get_value(idx) > 5.0;
    });

    // Indices 20 and 40 should be in the first part of the none_zero_indices
    EXPECT_EQ(split, 2);
    const auto& nzs = svc.get_non_zeros();
    for (int i = 0; i < split; ++i)
    {
        EXPECT_GT(svc.get_value(nzs[i]), 5.0);
    }
}

TEST_F(SparseVectorSumTest, IteratorsAndRangeLoop)
{
    kalix::SparseVectorSum svc(kDimension);

    // Initialize some values
    svc.add(0, 1.0);
    svc.add(1, 2.0);

    // Test non-const iteration
    double sum = 0.0;
    for (auto& val : svc)
    {
        sum += static_cast<double>(val);
    }
    // Only index 0 and 1 are set, rest are 0.0. Total sum should be 3.0.
    EXPECT_DOUBLE_EQ(sum, 3.0);

    // Test const iteration
    const auto& const_svc = svc;
    sum = 0.0;
    for (const auto& val : const_svc)
    {
        sum += static_cast<double>(val);
    }
    EXPECT_DOUBLE_EQ(sum, 3.0);

    // Test modification via iterator
    for (auto& val : svc)
    {
        val = kalix::CompensatedDouble(10.0);
    }
    EXPECT_DOUBLE_EQ(svc.get_value(0), 10.0);
    EXPECT_DOUBLE_EQ(svc.get_value(kDimension - 1), 10.0);
}

TEST_F(SparseVectorSumTest, SubscriptOperator)
{
    kalix::SparseVectorSum svc(kDimension);

    // Test Write Access
    svc[10] = kalix::CompensatedDouble(42.0);

    // Note: Writing via subscript does NOT update non_zero_indices automatically
    // in this implementation (it exposes the dense vector).
    // This test ensures the dense storage is updated correctly.
    EXPECT_DOUBLE_EQ(static_cast<double>(svc[10]), 42.0);

    // Test Read Access
    EXPECT_DOUBLE_EQ(static_cast<double>(svc[10]), 42.0);

    // Test Const Access
    const auto& const_svc = svc;
    EXPECT_DOUBLE_EQ(static_cast<double>(const_svc[10]), 42.0);
}

TEST_F(SparseVectorSumTest, AddCompensatedDoubleOverload)
{
    kalix::SparseVectorSum svc(kDimension);

    // Use the public single-argument constructor
    // (Assuming the 2-arg constructor (value, error) is internal/private)
    kalix::CompensatedDouble cd(10.5);

    svc.add(5, cd);

    EXPECT_DOUBLE_EQ(svc.get_value(5), 10.5);

    // Verify internal tracking
    const auto& nzs = svc.get_non_zeros();
    EXPECT_EQ(nzs.size(), 1);
    EXPECT_EQ(nzs[0], 5);
}

TEST_F(SparseVectorSumTest, EmptyAndCapacity)
{
    kalix::SparseVectorSum svc(0);
    EXPECT_TRUE(svc.empty());

    svc.set_dimension(100);
    EXPECT_FALSE(svc.empty());
    EXPECT_GE(svc.capacity(), 100u);
}

TEST_F(SparseVectorSumTest, DenseClearHeuristic)
{
    kalix::SparseVectorSum svc(10); // Small vector

    // Case 1: Sparse Fill (< 30%)
    svc.add(1, 1.0); // 1/10 = 10%
    svc.clear();
    EXPECT_EQ(svc.get_non_zeros().size(), 0);
    EXPECT_DOUBLE_EQ(svc.get_value(1), 0.0);

    // Case 2: Dense Fill (> 30%)
    // Fill 4 entries (40%)
    svc.add(0, 1.0);
    svc.add(2, 1.0);
    svc.add(4, 1.0);
    svc.add(6, 1.0);

    // This should trigger the dense memset/assign path
    svc.clear();

    EXPECT_EQ(svc.get_non_zeros().size(), 0);
    EXPECT_DOUBLE_EQ(svc.get_value(0), 0.0);
    EXPECT_DOUBLE_EQ(svc.get_value(6), 0.0);
}

TEST_F(SparseVectorSumTest, StreamOperator)
{
    kalix::SparseVectorSum svc(kDimension);
    svc.add(1, 10.0);
    svc.add(5, 20.0);

    std::stringstream ss;
    ss << svc;
    std::string output = ss.str();

    // Verify format: SparseVectorSum(dim=..., nnz=...) { ... }
    EXPECT_TRUE(output.find("SparseVectorSum(dim=100, nnz=2)") != std::string::npos);

    // Verify content
    EXPECT_TRUE(output.find("(1: 10)") != std::string::npos);
    EXPECT_TRUE(output.find("(5: 20)") != std::string::npos);
}
