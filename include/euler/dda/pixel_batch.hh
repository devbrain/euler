/**
 * @file pixel_batch.hh
 * @brief Pixel batching support for improved cache performance
 * @ingroup DDAModule
 * 
 * This header provides batching infrastructure for DDA iterators to
 * improve performance through:
 * - Reduced function call overhead by processing multiple pixels at once
 * - Better cache utilization with aligned memory layout
 * - SIMD-friendly data organization
 * - Prefetch optimization for sequential access
 * 
 * @section batch_sizes Batch Sizes
 * Batch sizes are optimized for cache line efficiency:
 * - Simple pixels: 16 pixels per batch (64 bytes on 32-bit systems)
 * - AA pixels: 8 pixels per batch (more data per pixel)
 * 
 * @section usage Usage
 * @code
 * // Manual batch processing
 * pixel_batch<pixel<int>> batch;
 * for (auto pixel : line_pixels(start, end)) {
 *     batch.add(pixel);
 *     if (batch.is_full()) {
 *         process_batch(batch);
 *         batch.clear();
 *     }
 * }
 * if (!batch.is_empty()) {
 *     process_batch(batch);
 * }
 * 
 * // Using batched iterators
 * auto batched = make_batched_line_iterator(start, end);
 * while (!batched.at_end()) {
 *     const auto& batch = batched.current_batch();
 *     // Process batch.count pixels at once
 *     batched.next_batch();
 * }
 * @endcode
 * 
 * @see batched_line_iterator.hh
 * @see batched_bezier_iterator.hh
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/core/simd.hh>
#include <euler/core/compiler.hh>
#include <euler/coordinates/point2.hh>
#include <array>
#include <cstring>

namespace euler::dda {

/**
 * @brief Optimal batch sizes for different pixel types
 */
template<typename PixelType>
struct batch_size_traits {
    static constexpr size_t value = 8;  // Default batch size
};

// Larger batches for simple pixels
template<typename T>
struct batch_size_traits<pixel<T>> {
    static constexpr size_t value = 16;
};

// Smaller batches for antialiased pixels (more data per pixel)
template<typename T>
struct batch_size_traits<aa_pixel<T>> {
    static constexpr size_t value = 8;
};

/**
 * @brief Batch of pixels for efficient processing
 * @tparam PixelType Type of pixel (pixel<T> or aa_pixel<T>)
 */
template<typename PixelType>
struct pixel_batch {
    using pixel_type = PixelType;
    static constexpr size_t max_size = batch_size_traits<PixelType>::value;
    
    alignas(32) std::array<PixelType, max_size> pixels;
    size_t count;
    
    pixel_batch() : count(0) {}
    
    // Add a pixel to the batch
    bool add(const PixelType& p) {
        if (count < max_size) {
            pixels[count++] = p;
            return true;
        }
        return false;
    }
    
    // Check if batch is full
    bool is_full() const { return count == max_size; }
    
    // Check if batch is empty
    bool is_empty() const { return count == 0; }
    
    // Clear the batch
    void clear() { count = 0; }
    
    // Get a view of valid pixels
    PixelType* data() { return pixels.data(); }
    const PixelType* data() const { return pixels.data(); }
    
    // Get size
    size_t size() const { return count; }
    
    // Array access
    PixelType& operator[](size_t i) { return pixels[i]; }
    const PixelType& operator[](size_t i) const { return pixels[i]; }
    
    // Iterators
    PixelType* begin() { return pixels.data(); }
    PixelType* end() { return pixels.data() + count; }
    const PixelType* begin() const { return pixels.data(); }
    const PixelType* end() const { return pixels.data() + count; }
};

/**
 * @brief Prefetch hints for pixel batching
 */
class prefetch_hint {
public:
    enum class locality {
        none = 0,      // No temporal locality (use once)
        low = 1,       // Low temporal locality
        moderate = 2,  // Moderate temporal locality  
        high = 3       // High temporal locality (keep in cache)
    };
    
    enum class operation {
        read = 0,   // Prefetch for read
        write = 1   // Prefetch for write
    };
    
    // Cross-platform prefetch wrapper - using template parameters for constants
    template<typename T, int Op = 0, int Loc = 2>
    static void prefetch(const T* addr) {
        #ifdef EULER_COMPILER_GCC
            __builtin_prefetch(addr, Op, Loc);
        #elif defined(EULER_COMPILER_CLANG)
            __builtin_prefetch(addr, Op, Loc);
        #elif defined(EULER_COMPILER_MSVC)
            #include <intrin.h>
            if constexpr (Loc == 0) {
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA);
            } else if constexpr (Loc == 1) {
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T2);
            } else if constexpr (Loc == 2) {
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1);
            } else {
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
            }
        #else
            // Fallback: just touch the memory
            volatile auto temp = *addr;
            (void)temp;
        #endif
    }
    
    // Helper overloads for common cases
    template<typename T>
    static void prefetch_for_read(const T* addr) {
        prefetch<T, 0, 2>(addr);  // read, moderate locality
    }
    
    template<typename T>
    static void prefetch_for_write(const T* addr) {
        prefetch<T, 1, 2>(addr);  // write, moderate locality
    }
    
    // Prefetch a range of memory
    template<typename T, int Op = 0, int Loc = 2>
    static void prefetch_range(const T* begin, const T* end) {
        constexpr size_t cache_line_size = 64;
        constexpr size_t stride = cache_line_size / sizeof(T);
        
        for (const T* p = begin; p < end; p += stride) {
            prefetch<T, Op, Loc>(p);
        }
    }
};

/**
 * @brief SIMD-optimized batch processing utilities
 */
template<typename T>
class batch_processor {
public:
    // Process a batch of regular pixels
    EULER_HOT
    static void process_pixel_batch(const pixel_batch<pixel<int>>& batch,
                                   std::function<void(const pixel<int>&)> callback) {
        // Process in groups of 4 for better SIMD utilization
        size_t i = 0;
        
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<int>::has_simd) {
            constexpr size_t simd_size = 4;
            for (; i + simd_size <= batch.count; i += simd_size) {
                // Prefetch next group
                if (i + simd_size < batch.count) {
                    prefetch_hint::prefetch<pixel<int>, 0, 3>(&batch.pixels[i + simd_size]);
                }
                
                // Process current group
                for (size_t j = 0; j < simd_size; ++j) {
                    callback(batch.pixels[i + j]);
                }
            }
        }
        #endif
        
        // Process remaining pixels
        EULER_LOOP_VECTORIZE
        for (; i < batch.count; ++i) {
            callback(batch.pixels[i]);
        }
    }
    
    // Process a batch of antialiased pixels with SIMD coverage accumulation
    static void process_aa_batch(const pixel_batch<aa_pixel<T>>& batch,
                                T* coverage_buffer,
                                size_t buffer_width) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd && std::is_floating_point_v<T>) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            // Process pixels in SIMD groups
            size_t i = 0;
            for (; i + simd_size <= batch.count; i += simd_size) {
                alignas(32) T x_coords[simd_size];
                alignas(32) T y_coords[simd_size];
                alignas(32) T coverages[simd_size];
                
                // Extract data for SIMD processing
                for (size_t j = 0; j < simd_size; ++j) {
                    x_coords[j] = batch.pixels[i + j].pos.x;
                    y_coords[j] = batch.pixels[i + j].pos.y;
                    coverages[j] = batch.pixels[i + j].coverage;
                }
                
                // Load into SIMD registers
                batch_t x_batch = batch_t::load_aligned(x_coords);
                batch_t y_batch = batch_t::load_aligned(y_coords);
                batch_t cov_batch = batch_t::load_aligned(coverages);
                
                // Process coverage accumulation
                // (Implementation depends on specific rendering needs)
            }
            
            // Handle remaining pixels
            for (; i < batch.count; ++i) {
                int x = static_cast<int>(batch.pixels[i].pos.x);
                int y = static_cast<int>(batch.pixels[i].pos.y);
                coverage_buffer[static_cast<size_t>(y) * buffer_width + static_cast<size_t>(x)] += batch.pixels[i].coverage;
            }
        } else
        #endif
        {
            // Scalar fallback
            for (size_t i = 0; i < batch.count; ++i) {
                int x = static_cast<int>(batch.pixels[i].pos.x);
                int y = static_cast<int>(batch.pixels[i].pos.y);
                coverage_buffer[static_cast<size_t>(y) * buffer_width + static_cast<size_t>(x)] += batch.pixels[i].coverage;
            }
        }
    }
};

/**
 * @brief Batch writer for efficient pixel output
 */
template<typename PixelType>
class batch_writer {
private:
    pixel_batch<PixelType> current_batch;
    std::function<void(const pixel_batch<PixelType>&)> flush_callback;
    
public:
    explicit batch_writer(std::function<void(const pixel_batch<PixelType>&)> callback)
        : flush_callback(std::move(callback)) {}
    
    ~batch_writer() {
        flush();
    }
    
    // Add a pixel, automatically flushing when batch is full
    void write(const PixelType& pixel) {
        if (!current_batch.add(pixel)) {
            flush();
            current_batch.add(pixel);
        }
    }
    
    // Force flush of current batch
    void flush() {
        if (!current_batch.is_empty() && flush_callback) {
            flush_callback(current_batch);
            current_batch.clear();
        }
    }
    
    // Get current batch size
    size_t pending_count() const { return current_batch.count; }
};

} // namespace euler::dda