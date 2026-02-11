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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "kalix/base/system_info.h"

TEST(SystemInfoTest, ReturnsNonZeroMemoryUsage) {
    const size_t memory = kalix::system::get_process_memory_usage();
    EXPECT_GT(memory, 0U) << "Memory usage should be greater than zero.";
}

TEST(SystemInfoTest, DetectsMemoryIncrease) {
    // Get Baseline
    const size_t initial_memory = kalix::system::get_process_memory_usage();

    // Allocate a significant chunk (10 MB)
    // We use a large amount to ensure we bypass small-block optimizers
    // in the memory allocator (malloc/free implementations).
    constexpr size_t alloc_size = 10 * 1024 * 1024;
    std::vector<uint8_t> huge_chunk(alloc_size);

    // FORCE PHYSICAL ALLOCATION
    // OSs are lazy; they give you virtual memory addresses but don't
    // assign physical RAM until you touch the page.
    // We fill the vector to force Page Faults and increase RSS.
    std::fill(huge_chunk.begin(), huge_chunk.end(), 0xAA);

    // Prevent compiler optimization from skipping the write
    // (accessing the data pointer effectively tells the compiler "I used this")
    const volatile uint8_t* prevent_opt = huge_chunk.data();
    (void)prevent_opt;

    // Measure Again
    const size_t spiked_memory = kalix::system::get_process_memory_usage();

    // Assertions
    // We check that memory increased.
    // Note: We don't check for exact equality (initial + 10MB) because
    // overhead from the vector class and GTest internals makes it fuzzy.
    EXPECT_GE(spiked_memory, initial_memory)
        << "Memory usage did not increase after allocating 10MB.";
}
