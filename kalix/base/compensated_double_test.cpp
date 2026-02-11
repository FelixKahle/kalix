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
#include "kalix/base/compensated_double.h"

TEST(CompensatedDoubleTest, ConstructionAndCast)
{
    const kalix::CompensatedDouble cd(5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(cd), 5.0);

    constexpr kalix::CompensatedDouble zero{};
    EXPECT_DOUBLE_EQ(static_cast<double>(zero), 0.0);
}

TEST(CompensatedDoubleTest, Addition)
{
    kalix::CompensatedDouble a(10.0);
    const kalix::CompensatedDouble b(20.0);

    // Test Comp + Comp
    const kalix::CompensatedDouble c = a + b;
    EXPECT_DOUBLE_EQ(static_cast<double>(c), 30.0);

    // Test Comp + double
    const kalix::CompensatedDouble d = a + 5.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(d), 15.0);

    // Test double + Comp
    const kalix::CompensatedDouble e = 5.0 + a;
    EXPECT_DOUBLE_EQ(static_cast<double>(e), 15.0);

    // Test +=
    a += 5.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(a), 15.0);
}

TEST(CompensatedDoubleTest, Subtraction)
{
    kalix::CompensatedDouble a(10.0);
    const kalix::CompensatedDouble b(3.0);

    const kalix::CompensatedDouble c = a - b;
    EXPECT_DOUBLE_EQ(static_cast<double>(c), 7.0);

    a -= 2.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(a), 8.0);
}

TEST(CompensatedDoubleTest, Multiplication)
{
    kalix::CompensatedDouble a(2.0);
    const kalix::CompensatedDouble b(3.0);

    const kalix::CompensatedDouble c = a * b;
    EXPECT_DOUBLE_EQ(static_cast<double>(c), 6.0);

    a *= 4.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(a), 8.0);
}

TEST(CompensatedDoubleTest, Division)
{
    kalix::CompensatedDouble a(10.0);
    const kalix::CompensatedDouble b(2.0);

    const kalix::CompensatedDouble c = a / b;
    EXPECT_DOUBLE_EQ(static_cast<double>(c), 5.0);

    a /= 2.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(a), 5.0);
}

// =========================================================================
// PRECISION TESTS
// These tests verify that kalix::CompensatedDouble actually retains more data
// than a standard double.
// =========================================================================

TEST(CompensatedDoubleTest, PrecisionLossRecovery)
{
    // 1.0 + 1e-16 is often lost in standard double precision arithmetic
    // if the compiler or FPU isn't using 80-bit extended precision intermediates.

    constexpr double large = 1.0;

    // Using standard double: (1 + small) - 1 might be 0.0 due to precision loss
    // (Note: behavior depends on specific small value vs epsilon)
    // Let's use a value guaranteed to be problematic for simple accumulation chains
    constexpr double tiny = 1e-19;

    const kalix::CompensatedDouble c_large(large);
    const kalix::CompensatedDouble c_tiny(tiny);

    // Perform operation
    const kalix::CompensatedDouble sum = c_large + c_tiny;

    // If we cast 'sum' to double, we lose the tiny part because double can't hold it.
    // BUT, if we subtract the large part using kalix::CompensatedDouble arithmetic,
    // we should get the tiny part back exactly.
    const kalix::CompensatedDouble recovered = sum - c_large;

    EXPECT_NEAR(static_cast<double>(recovered), tiny, 1e-25);
}

TEST(CompensatedDoubleTest, PrecisionMultiplication)
{
    // (1 + x)(1 - x) = 1 - x^2
    // If x is small (1e-9), x^2 is 1e-18.
    // In standard double, 1 - 1e-18 == 1.0.

    constexpr double x_val = 1e-9;
    const kalix::CompensatedDouble one(1.0);
    const kalix::CompensatedDouble x(x_val);

    const kalix::CompensatedDouble result = (one + x) * (one - x);

    // result should be 1.0 - 1e-18.
    // If we check (1.0 - result), we should see 1e-18.
    const kalix::CompensatedDouble diff = one - result;

    EXPECT_NEAR(static_cast<double>(diff), 1e-18, 1e-24);
}

// =========================================================================
// MATH FUNCTIONS
// =========================================================================

TEST(CompensatedDoubleTest, Sqrt)
{
    const kalix::CompensatedDouble four(4.0);
    const kalix::CompensatedDouble two = sqrt(four);
    EXPECT_DOUBLE_EQ(static_cast<double>(two), 2.0);

    const kalix::CompensatedDouble zero(0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(sqrt(zero)), 0.0);
}

TEST(CompensatedDoubleTest, Abs)
{
    const kalix::CompensatedDouble neg(-5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(abs(neg)), 5.0);

    const kalix::CompensatedDouble pos(5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(abs(pos)), 5.0);
}

TEST(CompensatedDoubleTest, FloorCeilRound)
{
    const kalix::CompensatedDouble val(5.7);

    EXPECT_DOUBLE_EQ(static_cast<double>(floor(val)), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ceil(val)), 6.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(round(val)), 6.0);

    const kalix::CompensatedDouble neg_val(-5.7);
    EXPECT_DOUBLE_EQ(static_cast<double>(floor(neg_val)), -6.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ceil(neg_val)), -5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(round(neg_val)), -6.0);

    // Special case |x| < 1 mentioned in code comments
    const kalix::CompensatedDouble small_pos(0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(floor(small_pos)), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ceil(small_pos)), 1.0);

    const kalix::CompensatedDouble small_neg(-0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(floor(small_neg)), -1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ceil(small_neg)), 0.0);
}

TEST(CompensatedDoubleTest, Ldexp)
{
    const kalix::CompensatedDouble val(2.0);
    // 2.0 * 2^3 = 16.0
    const kalix::CompensatedDouble res = ldexp(val, 3);
    EXPECT_DOUBLE_EQ(static_cast<double>(res), 16.0);
}

TEST(CompensatedDoubleTest, Comparisons)
{
    const kalix::CompensatedDouble a(10.0);
    const kalix::CompensatedDouble b(20.0);
    const kalix::CompensatedDouble a_copy(10.0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a == a_copy);
    EXPECT_TRUE(a != b);

    // Mixed types
    EXPECT_TRUE(a < 20.0);
    EXPECT_TRUE(20.0 > a);
}
