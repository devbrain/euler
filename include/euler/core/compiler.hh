/**
 * @file compiler.hh
 * @brief Compiler detection, feature macros, and optimization hints
 * 
 * This header provides:
 * - Compiler detection macros
 * - Optimization pragmas and attributes
 * - Loop vectorization hints
 * - Function attributes for performance
 */
#pragma once

// Compiler detection
#if !defined(__EMSCRIPTEN__)
#if defined(__clang__)
#define EULER_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
#define EULER_COMPILER_GCC
#elif defined(_MSC_VER)
#define EULER_COMPILER_MSVC
#endif
#else
#define EULER_COMPILER_WASM
#endif

// Optimization attributes
#if defined(EULER_COMPILER_GCC) || defined(EULER_COMPILER_CLANG)
    // Force inline for critical functions
    #define EULER_ALWAYS_INLINE __attribute__((always_inline)) inline
    
    // Hot function - optimize aggressively
    #define EULER_HOT __attribute__((hot))
    
    // Cold function - optimize for size
    #define EULER_COLD __attribute__((cold))
    
    // Pure function - no side effects, result depends only on arguments
    #define EULER_PURE __attribute__((pure))
    
    // Const function - pure + doesn't access memory
    #define EULER_CONST __attribute__((const))
    
    // Restrict pointer aliasing
    #define EULER_RESTRICT __restrict__
    
    // Assume aligned memory
    #define EULER_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
    
    // Likely/unlikely branch hints
    #define EULER_LIKELY(x) __builtin_expect(!!(x), 1)
    #define EULER_UNLIKELY(x) __builtin_expect(!!(x), 0)
    
#elif defined(EULER_COMPILER_MSVC)
    #define EULER_ALWAYS_INLINE __forceinline
    #define EULER_HOT
    #define EULER_COLD
    #define EULER_PURE
    #define EULER_CONST
    #define EULER_RESTRICT __restrict
    #define EULER_ASSUME_ALIGNED(ptr, alignment) ptr
    #define EULER_LIKELY(x) (x)
    #define EULER_UNLIKELY(x) (x)
#else
    #define EULER_ALWAYS_INLINE inline
    #define EULER_HOT
    #define EULER_COLD
    #define EULER_PURE
    #define EULER_CONST
    #define EULER_RESTRICT
    #define EULER_ASSUME_ALIGNED(ptr, alignment) ptr
    #define EULER_LIKELY(x) (x)
    #define EULER_UNLIKELY(x) (x)
#endif

// Loop optimization pragmas
#if defined(EULER_COMPILER_GCC)
    #define EULER_PRAGMA(x) _Pragma(#x)
    #define EULER_LOOP_VECTORIZE EULER_PRAGMA(GCC ivdep)
    #define EULER_LOOP_UNROLL(n) EULER_PRAGMA(GCC unroll n)
    #define EULER_LOOP_UNROLL_FULL EULER_PRAGMA(GCC unroll 65534)
    
#elif defined(EULER_COMPILER_CLANG)
    #define EULER_PRAGMA(x) _Pragma(#x)
    #define EULER_LOOP_VECTORIZE EULER_PRAGMA(clang loop vectorize(enable))
    #define EULER_LOOP_UNROLL(n) EULER_PRAGMA(clang loop unroll_count(n))
    #define EULER_LOOP_UNROLL_FULL EULER_PRAGMA(clang loop unroll(full))
    
#elif defined(EULER_COMPILER_MSVC)
    #define EULER_PRAGMA(x) __pragma(x)
    #define EULER_LOOP_VECTORIZE EULER_PRAGMA(loop(ivdep))
    #define EULER_LOOP_UNROLL(n)
    #define EULER_LOOP_UNROLL_FULL
    
#else
    #define EULER_PRAGMA(x)
    #define EULER_LOOP_VECTORIZE
    #define EULER_LOOP_UNROLL(n)
    #define EULER_LOOP_UNROLL_FULL
#endif

// Additional loop hints
#if defined(EULER_COMPILER_CLANG)
    #define EULER_LOOP_VECTORIZE_WIDTH(n) EULER_PRAGMA(clang loop vectorize_width(n))
    #define EULER_LOOP_INTERLEAVE(n) EULER_PRAGMA(clang loop interleave_count(n))
    #define EULER_LOOP_DISTRIBUTE EULER_PRAGMA(clang loop distribute(enable))
#else
    #define EULER_LOOP_VECTORIZE_WIDTH(n)
    #define EULER_LOOP_INTERLEAVE(n)
    #define EULER_LOOP_DISTRIBUTE
#endif

// Prefetch hints (already used in pixel_batch.hh, but let's make it consistent)
#if defined(EULER_COMPILER_GCC) || defined(EULER_COMPILER_CLANG)
    #define EULER_PREFETCH_READ(addr) __builtin_prefetch(addr, 0, 3)
    #define EULER_PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
    #define EULER_PREFETCH_READ_NTA(addr) __builtin_prefetch(addr, 0, 0)
#elif defined(EULER_COMPILER_MSVC)
    #include <intrin.h>
    #define EULER_PREFETCH_READ(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define EULER_PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define EULER_PREFETCH_READ_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#else
    #define EULER_PREFETCH_READ(addr) ((void)0)
    #define EULER_PREFETCH_WRITE(addr) ((void)0)
    #define EULER_PREFETCH_READ_NTA(addr) ((void)0)
#endif
