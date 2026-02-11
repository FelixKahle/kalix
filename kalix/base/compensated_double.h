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

#ifndef KALIX_BASE_COMPENSATED_DOUBLE_H_
#define KALIX_BASE_COMPENSATED_DOUBLE_H_

#include <cmath>
#include <iostream>
#include "kalix/base/config.h"

namespace kalix
{
    /// @brief A high-precision floating-point number using compensated arithmetic (Double-Double).
    ///
    /// The CompensatedDouble class represents a real number as the unevaluated sum of two
    /// standard IEEE 754 double-precision values: \f$ x = hi + lo \f$.
    ///
    /// By tracking the rounding error (`lo`) of every arithmetic operation separately from the
    /// approximation (`hi`), this class provides approximately 106 bits of significand precision
    /// (roughly 31 decimal digits), effectively doubling the precision of a standard `double`.
    ///
    /// This implementation uses Error-Free Transformations (EFT) based on the algorithms described
    /// by Siegfried M. Rump in "High precision evaluation of nonlinear functions" (2005).
    ///
    /// @note This is a software implementation. While it is much faster than arbitrary-precision
    /// libraries (like MPFR), it is slower than native hardware `double` arithmetic.
    class CompensatedDouble
    {
        // The following functions are implemented as described in:
        // Rump, Siegfried M. "High precision evaluation of nonlinear functions."
        // Proceedings of. 2005.

        /// @brief Computes the exact sum of two numbers as a non-overlapping expansion (Knuth's TwoSum).
        ///
        /// This function calculates a pair \f$(x, y)\f$ such that:
        /// \f[ a + b = x + y \f]
        /// where \f$x = \text{fl}(a + b)\f$ is the standard floating-point sum rounded to nearest,
        /// and \f$y\f$ represents the exact rounding error.
        ///
        /// Unlike the faster `FastTwoSum` algorithm, this version does not require the inputs
        /// to be sorted by magnitude (i.e., it works correctly even if $|a| < |b|$).
        ///
        /// @param[out] x The high-order component (approximation).
        /// @param[out] y The low-order component (exact error).
        /// @param[in]  a The first summand.
        /// @param[in]  b The second summand.
        ///
        /// @note Cost: 6 floating-point operations.
        static KALIX_FORCE_INLINE void two_sum(double& x, double& y, const double a, const double b)
        {
            x = a + b;
            const double z = x - a;
            y = (a - (x - z)) + (b - z);
        }

        /// @brief Splits a 53-bit double into two non-overlapping 26-bit parts (Veltkamp's Split).
        ///
        /// This transformation splits a floating-point number \f$ a \f$ into two parts \f$ x \f$ and \f$ y \f$
        /// such that \f$ a = x + y \f$ exactly. Both parts fit into at most 26 bits of the significand.
        ///
        /// This is a prerequisite for exact multiplication without overflow in the significand.
        ///
        /// @param[out] x The high part of the split.
        /// @param[out] y The low part of the split.
        /// @param[in]  a The number to split.
        ///
        /// @note Cost: 4 floating-point operations.
        static KALIX_FORCE_INLINE void split(double& x, double& y, const double a)
        {
            constexpr auto factor = static_cast<double>((1 << 27) + 1);
            const double c = factor * a;
            x = c - (c - a);
            y = a - x;
        }

        /// @brief Computes the exact product of two numbers (Dekker's TwoProduct).
        ///
        /// This function calculates a pair \f$(x, y)\f$ such that:
        /// \f[ a \cdot b = x + y \f]
        /// where \f$x = \text{fl}(a \cdot b)\f$ is the standard floating-point product,
        /// and \f$y\f$ represents the exact rounding error.
        ///
        /// @param[out] x The high-order component (approximation).
        /// @param[out] y The low-order component (exact error).
        /// @param[in]  a The first factor.
        /// @param[in]  b The second factor.
        ///
        /// @note Cost: 17 floating-point operations.
        static KALIX_FORCE_INLINE void two_product(double& x, double& y, const double a, const double b)
        {
            x = a * b;
            double a1, a2, b1, b2;
            split(a1, a2, a);
            split(b1, b2, b);
            y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2);
        }

        /// @brief Private constructor for creating a CompensatedDouble from explicit components.
        /// @param hi_ The high-order component (approximation).
        /// @param lo_ The low-order component (error term).
        KALIX_FORCE_INLINE CompensatedDouble(const double hi_, const double lo_)
            : hi(hi_), lo(lo_)
        {
        }

    public:
        /// @brief Default constructor. Initializes to 0.0.
        KALIX_FORCE_INLINE CompensatedDouble() = default;

        /// @brief Constructs a CompensatedDouble from a standard double.
        ///
        /// The low-order component is initialized to 0.0.
        /// @param val The initial value.
        explicit KALIX_FORCE_INLINE CompensatedDouble(const double val)
            : hi(val), lo(0.0)
        {
        }

        /// @brief explicit conversion to standard double precision.
        /// @return The result of \f$ hi + lo \f$ (loss of precision).
        explicit KALIX_FORCE_INLINE operator double() const
        {
            return hi + lo;
        }

        /// @brief Adds a standard double to this number in place.
        /// @param v The value to add.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator+=(const double v)
        {
            double c;
            two_sum(hi, c, v, hi);
            lo += c;
            return *this;
        }

        /// @brief Adds another CompensatedDouble to this number in place.
        /// @param v The value to add.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator+=(const CompensatedDouble& v)
        {
            (*this) += v.hi;
            lo += v.lo;
            return *this;
        }

        /// @brief Subtracts a standard double from this number in place.
        /// @param v The value to subtract.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator-=(const double v)
        {
            (*this) += -v;
            return *this;
        }

        /// @brief Subtracts another CompensatedDouble from this number in place.
        /// @param v The value to subtract.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator-=(const CompensatedDouble& v)
        {
            (*this) -= v.hi;
            lo -= v.lo;
            return *this;
        }

        /// @brief Multiplies this number by a standard double in place.
        /// @param v The scalar to multiply by.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator*=(const double v)
        {
            const double c = lo * v;
            two_product(hi, lo, hi, v);
            *this += c;
            return *this;
        }

        /// @brief Multiplies this number by another CompensatedDouble in place.
        /// @param v The value to multiply by.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator*=(const CompensatedDouble& v)
        {
            const double c1 = hi * v.lo;
            const double c2 = lo * v.hi;
            two_product(hi, lo, hi, v.hi);
            *this += c1;
            *this += c2;
            return *this;
        }

        /// @brief Divides this number by a standard double in place.
        /// @param v The scalar divisor.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator/=(const double v)
        {
            const CompensatedDouble d(hi / v, lo / v);
            CompensatedDouble c = d * v - (*this);
            c.hi /= v;
            c.lo /= v;
            *this = d - c;
            return *this;
        }

        /// @brief Divides this number by another CompensatedDouble in place.
        /// @param v The divisor.
        /// @return Reference to this object.
        KALIX_FORCE_INLINE CompensatedDouble& operator/=(const CompensatedDouble& v)
        {
            const double vdbl = v.hi + v.lo;
            const CompensatedDouble d(hi / vdbl, lo / vdbl);
            CompensatedDouble c = d * v - (*this);
            c.hi /= vdbl;
            c.lo /= vdbl;
            *this = d - c;
            return *this;
        }

        /// @brief Unary negation.
        /// @return A new CompensatedDouble with negated components.
        KALIX_FORCE_INLINE CompensatedDouble operator-() const
        {
            return {-hi, -lo};
        }

        /// @brief Addition operator (Compensated + double).
        KALIX_FORCE_INLINE CompensatedDouble operator+(const double v) const
        {
            CompensatedDouble res{};
            two_sum(res.hi, res.lo, hi, v);
            res.lo += lo;
            return res;
        }

        /// @brief Addition operator (Compensated + Compensated).
        KALIX_FORCE_INLINE CompensatedDouble operator+(const CompensatedDouble& v) const
        {
            CompensatedDouble res = (*this) + v.hi;
            res.lo += v.lo;
            return res;
        }

        /// @brief Addition operator (double + Compensated).
        friend KALIX_FORCE_INLINE CompensatedDouble operator+(const double a, const CompensatedDouble& b)
        {
            return b + a;
        }

        /// @brief Subtraction operator (Compensated - double).
        KALIX_FORCE_INLINE CompensatedDouble operator-(const double v) const
        {
            CompensatedDouble res{};
            two_sum(res.hi, res.lo, hi, -v);
            res.lo += lo;
            return res;
        }

        /// @brief Subtraction operator (Compensated - Compensated).
        KALIX_FORCE_INLINE CompensatedDouble operator-(const CompensatedDouble& v) const
        {
            CompensatedDouble res = (*this) - v.hi;
            res.lo -= v.lo;
            return res;
        }

        /// @brief Subtraction operator (double - Compensated).
        friend KALIX_FORCE_INLINE CompensatedDouble operator-(const double a, const CompensatedDouble& b)
        {
            return -b + a;
        }

        /// @brief Multiplication operator (Compensated * double).
        KALIX_FORCE_INLINE CompensatedDouble operator*(const double v) const
        {
            CompensatedDouble res{};
            two_product(res.hi, res.lo, hi, v);
            res += lo * v;
            return res;
        }

        /// @brief Multiplication operator (Compensated * Compensated).
        KALIX_FORCE_INLINE CompensatedDouble operator*(const CompensatedDouble& v) const
        {
            CompensatedDouble res = (*this) * v.hi;
            res += hi * v.lo;
            return res;
        }

        /// @brief Multiplication operator (double * Compensated).
        friend KALIX_FORCE_INLINE CompensatedDouble operator*(const double a, const CompensatedDouble& b)
        {
            return b * a;
        }

        /// @brief Division operator (Compensated / double).
        KALIX_FORCE_INLINE CompensatedDouble operator/(const double v) const
        {
            CompensatedDouble res = *this;
            res /= v;
            return res;
        }

        /// @brief Division operator (Compensated / Compensated).
        KALIX_FORCE_INLINE CompensatedDouble operator/(const CompensatedDouble& v) const
        {
            CompensatedDouble res = (*this);
            res /= v;
            return res;
        }

        /// @brief Division operator (double / Compensated).
        friend KALIX_FORCE_INLINE CompensatedDouble operator/(const double a, const CompensatedDouble& b)
        {
            return CompensatedDouble(a) / b;
        }

        /// @brief Greater-than comparison (Compensated > Compensated).
        KALIX_FORCE_INLINE bool operator>(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) > static_cast<double>(other);
        }

        /// @brief Greater-than comparison (Compensated > double).
        KALIX_FORCE_INLINE bool operator>(const double other) const { return static_cast<double>(*this) > other; }

        /// @brief Greater-than comparison (double > Compensated).
        friend KALIX_FORCE_INLINE bool operator>(const double a, const CompensatedDouble& b)
        {
            return a > static_cast<double>(b);
        }

        /// @brief Less-than comparison (Compensated < Compensated).
        KALIX_FORCE_INLINE bool operator<(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) < static_cast<double>(other);
        }

        /// @brief Less-than comparison (Compensated < double).
        KALIX_FORCE_INLINE bool operator<(const double other) const { return static_cast<double>(*this) < other; }

        /// @brief Less-than comparison (double < Compensated).
        friend KALIX_FORCE_INLINE bool operator<(const double a, const CompensatedDouble& b)
        {
            return a < static_cast<double>(b);
        }

        /// @brief Greater-than-or-equal comparison (Compensated >= Compensated).
        KALIX_FORCE_INLINE bool operator>=(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) >= static_cast<double>(other);
        }

        /// @brief Greater-than-or-equal comparison (Compensated >= double).
        KALIX_FORCE_INLINE bool operator>=(const double other) const { return static_cast<double>(*this) >= other; }

        /// @brief Greater-than-or-equal comparison (double >= Compensated).
        friend KALIX_FORCE_INLINE bool operator>=(const double a, const CompensatedDouble& b)
        {
            return a >= static_cast<double>(b);
        }

        /// @brief Less-than-or-equal comparison (Compensated <= Compensated).
        KALIX_FORCE_INLINE bool operator<=(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) <= static_cast<double>(other);
        }

        /// @brief Less-than-or-equal comparison (Compensated <= double).
        KALIX_FORCE_INLINE bool operator<=(const double other) const { return static_cast<double>(*this) <= other; }

        /// @brief Less-than-or-equal comparison (double <= Compensated).
        friend KALIX_FORCE_INLINE bool operator<=(const double a, const CompensatedDouble& b)
        {
            return a <= static_cast<double>(b);
        }

        /// @brief Equality comparison (Compensated == Compensated).
        KALIX_FORCE_INLINE bool operator==(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) == static_cast<double>(other);
        }

        /// @brief Equality comparison (Compensated == double).
        KALIX_FORCE_INLINE bool operator==(const double other) const { return static_cast<double>(*this) == other; }

        /// @brief Equality comparison (double == Compensated).
        friend KALIX_FORCE_INLINE bool operator==(const double a, const CompensatedDouble& b)
        {
            return a == static_cast<double>(b);
        }

        /// @brief Inequality comparison (Compensated != Compensated).
        KALIX_FORCE_INLINE bool operator!=(const CompensatedDouble& other) const
        {
            return static_cast<double>(*this) != static_cast<double>(other);
        }

        /// @brief Inequality comparison (Compensated != double).
        KALIX_FORCE_INLINE bool operator!=(const double other) const { return static_cast<double>(*this) != other; }

        /// @brief Inequality comparison (double != Compensated).
        friend KALIX_FORCE_INLINE bool operator!=(const double a, const CompensatedDouble& b)
        {
            return a != static_cast<double>(b);
        }

        // =========================================================================
        // Utilities & Math Friends
        // =========================================================================

        /// @brief Renormalizes the internal components.
        ///
        /// Recalculates `hi` and `lo` such that the magnitude of `lo` is minimized
        /// relative to `hi`. This ensures the representation remains canonical.
        KALIX_FORCE_INLINE void renormalize()
        {
            two_sum(hi, lo, hi, lo);
        }

        /// @brief Computes the absolute value.
        /// @param v The input value.
        /// @return The absolute value of `v`.
        friend KALIX_FORCE_INLINE CompensatedDouble abs(const CompensatedDouble& v) { return v < 0 ? -v : v; }

        /// @brief Computes the square root with high precision.
        ///
        /// Uses an initial standard double-precision `sqrt` as a guess, followed by
        /// a Newton-Raphson iteration performed in compensated arithmetic to refine
        /// the result to full precision.
        ///
        /// @param v The input value (must be non-negative).
        /// @return The square root of `v`.
        friend KALIX_FORCE_INLINE CompensatedDouble sqrt(const CompensatedDouble& v)
        {
            const double c = std::sqrt(v.hi + v.lo);

            // guard against division by zero
            if (c == 0.0)
            {
                return CompensatedDouble(0.0);
            }

            // calculate precise square root by newton step
            CompensatedDouble res = v / c;
            res += c;
            // multiplication by 0.5 is exact
            res.hi *= 0.5;
            res.lo *= 0.5;
            return res;
        }

        /// @brief Computes the floor of the value (largest integer not greater than x).
        /// @note Includes special handling for values strictly between -1 and 1.
        friend KALIX_FORCE_INLINE CompensatedDouble floor(const CompensatedDouble& x)
        {
            // Treat |x| < 1 as special case, as per (for example)
            // https://github.com/shibatch/tlfloat: see #2041
            if (abs(x) < 1)
            {
                if (x == 0 || x > 0)
                {
                    return CompensatedDouble(0.0);
                }
                return CompensatedDouble(-1.0);
            }
            const double floor_x = std::floor(static_cast<double>(x));
            CompensatedDouble res{};

            two_sum(res.hi, res.lo, floor_x, std::floor(static_cast<double>(x - floor_x)));
            return res;
        }

        /// @brief Computes the ceil of the value (smallest integer not less than x).
        /// @note Includes special handling for values strictly between -1 and 1.
        friend KALIX_FORCE_INLINE CompensatedDouble ceil(const CompensatedDouble& x)
        {
            // Treat |x| < 1 as special case, as per (for example)
            // https://github.com/shibatch/tlfloat: see #2041
            if (abs(x) < 1)
            {
                if (x == 0 || x < 0)
                {
                    return CompensatedDouble(0.0);
                }
                return CompensatedDouble(1.0);
            }
            const double ceil_x = std::ceil(static_cast<double>(x));
            CompensatedDouble res{};

            two_sum(res.hi, res.lo, ceil_x, std::ceil(static_cast<double>(x - ceil_x)));
            return res;
        }

        /// @brief Rounds to the nearest integer.
        /// @note Rounds halfway cases away from zero.
        friend KALIX_FORCE_INLINE CompensatedDouble round(const CompensatedDouble& x) { return floor(x + 0.5); }

        /// @brief Multiplies a compensated number by an integral power of 2.
        /// @param v The value to scale.
        /// @param exp The exponent power of 2.
        /// @return \f$ v \cdot 2^{exp} \f$
        friend KALIX_FORCE_INLINE CompensatedDouble ldexp(const CompensatedDouble& v, const int exp)
        {
            return {std::ldexp(v.hi, exp), std::ldexp(v.lo, exp)};
        }

        /// @brief Stream insertion operator.
        /// @note Prints the double-precision approximation of the value.
        friend std::ostream& operator<<(std::ostream& os, const CompensatedDouble& v)
        {
            os << static_cast<double>(v);
            return os;
        }

    private:
        double hi;
        double lo;
    };
}

#endif // KALIX_BASE_COMPENSATED_DOUBLE_H_
