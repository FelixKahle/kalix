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

#include "kalix/base/system_info.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
// Link with Psapi.lib on Windows usually required,
// but modern MSVC links it automatically for these functions.

#elif defined(__APPLE__)
#include <mach/mach.h>

#elif defined(__linux__) || defined(__linux)
#include <unistd.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>

#endif

namespace kalix::system
{
#if defined(_WIN32)

    size_t get_process_memory_usage()
    {
        PROCESS_MEMORY_COUNTERS pmc;
        // Use GetCurrentProcess() pseudo-handle. No OpenProcess/CloseHandle needed.
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        {
            return static_cast<size_t>(pmc.WorkingSetSize);
        }
        return 0;
    }

#elif defined(__APPLE__)

    size_t get_process_memory_usage()
    {
        // Use MACH_TASK_BASIC_INFO to support >4GB memory reporting on 64-bit systems
        mach_task_basic_info info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        {
            return info.resident_size;
        }
        return 0;
    }

#elif defined(__linux__) || defined(__linux)

    size_t get_process_memory_usage()
    {
        // /proc/self/statm is safer and faster (no PID parsing needed)
        FILE* file = std::fopen("/proc/self/statm", "r");
        if (!file)
        {
            return 0;
        }

        long rss_pages = 0;
        // statm format: size resident shared text lib data dt
        // We skip the first value (virtual size) and read the second (resident)
        if (std::fscanf(file, "%*s %ld", &rss_pages) != 1)
        {
            std::fclose(file);
            return 0;
        }
        std::fclose(file);

        // sysconf can theoretically fail (-1), but unlikely for PAGESIZE
        long page_size = sysconf(_SC_PAGESIZE);
        return static_cast<size_t>(rss_pages) * static_cast<size_t>(page_size);
    }

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)

    size_t get_process_memory_usage()
    {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            // ru_maxrss is in Kilobytes on BSD systems
            return static_cast<size_t>(usage.ru_maxrss) * 1024;
        }
        return 0;
    }

#else

    // Fallback for unknown platforms
    size_t get_process_memory_usage()
    {
        return 0;
    }

#endif
} // namespace kalix::system
