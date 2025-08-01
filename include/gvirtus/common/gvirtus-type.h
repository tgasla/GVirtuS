#pragma once

#include <stdint.h>

namespace gvirtus::common {
    #if defined(__ia64) || defined(__itanium__) || defined(_M_IA64) || \
        defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
        typedef uint64_t pointer_t;
    #else
    typedef uint32_t pointer_t;
    #endif

    typedef struct __mappedPointer {
        void *pointer;
        size_t size;
    } mappedPointer;

    typedef enum __funcs {
        RegisterFatBinary,
        UnregisterFatBinary,
        RegisterFunction,
        RegisterVar,
        RegisterShared,
        RegisterSharedVar
    } funcs;
}