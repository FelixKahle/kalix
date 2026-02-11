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

#ifndef KALIX_BASE_CONFIG_H_
#define KALIX_BASE_CONFIG_H_

// Determine the C++ Standard Version
#ifndef KALIX_CPLUSPLUS_LANG
    #if defined(_MSVC_LANG)
        #define KALIX_CPLUSPLUS_LANG _MSVC_LANG
    #elif defined(__cplusplus)
        #define KALIX_CPLUSPLUS_LANG __cplusplus
    #else
        #define KALIX_CPLUSPLUS_LANG 0L
    #endif
#endif

// Enforce C++20
#define KALIX_CPP_20_STANDARD 202002L

#if KALIX_CPLUSPLUS_LANG < KALIX_CPP_20_STANDARD
    #error "Kalix requires at least C++20."
#endif

// Verify specific C++20 features (Double check compiler compliance)
#if !defined(__cpp_constexpr) || __cpp_constexpr < 201907L
    #error "Kalix requires full compiler support for C++20 constexpr."
#endif

#if !defined(__cpp_concepts) || __cpp_concepts < 201907L
    #error "Kalix requires compiler support for C++20 concepts."
#endif

#ifdef NDEBUG
    #define KALIX_BUILD_RELEASE 1
    #define KALIX_BUILD_DEBUG 0
#else
    #define KALIX_BUILD_RELEASE 0
    #define KALIX_BUILD_DEBUG 1
#endif

#if (KALIX_BUILD_DEBUG + KALIX_BUILD_RELEASE) != 1
    #error "Inconsistent build configuration: Both KALIX_BUILD_DEBUG and KALIX_BUILD_RELEASE are set or unset."
#endif

// Force Inline
#ifndef KALIX_ALLOW_FORCE_INLINE
    #define KALIX_ALLOW_FORCE_INLINE 1
#endif

#ifndef KALIX_FORCE_INLINE
    #if KALIX_ALLOW_FORCE_INLINE
        #if defined(_MSC_VER) || defined(__INTEL_COMPILER) || (defined(__INTEL_LLVM_COMPILER) && defined(_WIN32))
            #define KALIX_FORCE_INLINE __forceinline
        #elif defined(__GNUC__) || defined(__clang__) || defined(__IBMCPP__) || defined(__NVCOMPILER) || defined(__ARMCC_VERSION)
            #define KALIX_FORCE_INLINE inline __attribute__((always_inline))
        #else
            #define KALIX_FORCE_INLINE inline
        #endif
    #else
        #define KALIX_FORCE_INLINE inline
    #endif
#endif

// No Inline
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
    #define KALIX_NO_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER) || defined(__IBMCPP__) || defined(__ARMCC_VERSION)
    #define KALIX_NO_INLINE __attribute__((noinline))
#else
    #define KALIX_NO_INLINE
#endif

// Branch Prediction Hints
// Note: C++20 introduced [[likely]] and [[unlikely]] attributes.
// However, these macros are useful for wrapping expressions (e.g., inside if conditions).
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__NVCOMPILER) || defined(__IBMCPP__) || defined(__ARMCC_VERSION)
    #define KALIX_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define KALIX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define KALIX_LIKELY(x)   (x)
    #define KALIX_UNLIKELY(x) (x)
#endif

#if !defined(KALIX_SYMBOL_EXPORT) && !defined(KALIX_SYMBOL_IMPORT) && !defined(KALIX_SYMBOL_LOCAL)
    #if defined(_WIN32) || defined(__CYGWIN__)
        #define KALIX_SYMBOL_EXPORT __declspec(dllexport)
        #define KALIX_SYMBOL_IMPORT __declspec(dllimport)
        #define KALIX_SYMBOL_LOCAL
    #elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER) || defined(__NVCOMPILER)
        #define KALIX_SYMBOL_EXPORT __attribute__((visibility("default")))
        #define KALIX_SYMBOL_IMPORT __attribute__((visibility("default")))
        #define KALIX_SYMBOL_LOCAL  __attribute__((visibility("hidden")))
    #elif defined(__SUNPRO_CC)
        #define KALIX_SYMBOL_EXPORT __global
        #define KALIX_SYMBOL_IMPORT __global
        #define KALIX_SYMBOL_LOCAL  __hidden
    #else
        #define KALIX_SYMBOL_EXPORT
        #define KALIX_SYMBOL_IMPORT
        #define KALIX_SYMBOL_LOCAL
    #endif
#endif

#endif // KALIX_BASE_CONFIG_H_
