/**
 * @file batched_iterators.hh
 * @brief Consolidated header for all batched DDA iterators
 * @ingroup DDAModule
 * 
 * This header provides a convenient way to include all batched DDA iterators
 * and documents their consistent API.
 * 
 * @section api_overview Batched Iterator API
 * 
 * All batched iterators follow a consistent API pattern:
 * 
 * 1. **Type Definitions:**
 *    - `coord_type`: The coordinate type (template parameter T)
 *    - `point_type`: point2<T>
 *    - `pixel_type`: pixel<int> or span type
 *    - `batch_type`: pixel_batch<pixel_type>
 * 
 * 2. **Core Methods:**
 *    - `current_batch()`: Get the current batch of pixels
 *    - `at_end()`: Check if iteration is complete
 *    - `next_batch()`: Advance to the next batch
 *    - `process_all(callback)`: Process all pixels with a callback
 * 
 * 3. **Callback Signature:**
 *    The callback for `process_all()` takes a const reference to the batch:
 *    ```cpp
 *    void callback(const pixel_batch<pixel<int>>& batch);
 *    ```
 * 
 * @section usage_example Usage Example
 * 
 * @code
 * // Drawing a batched circle
 * auto circle = make_batched_circle(center, radius);
 * circle.process_all([&](const auto& batch) {
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_pixel(batch.pixels[i].pos);
 *     }
 * });
 * 
 * // Manual iteration
 * auto line = make_batched_line(start, end);
 * while (!line.at_end()) {
 *     const auto& batch = line.current_batch();
 *     // Process batch...
 *     line.next_batch();
 * }
 * @endcode
 * 
 * @section performance Performance Benefits
 * 
 * Batched iterators provide several performance benefits:
 * - Reduced function call overhead
 * - Better cache utilization
 * - SIMD-friendly data layout
 * - Prefetching opportunities
 * - Amortized computation costs
 * 
 * @section available_iterators Available Batched Iterators
 * 
 * - **Lines:**
 *   - batched_line_iterator
 *   - batched_aa_line_iterator
 *   - batched_thick_line_iterator
 * 
 * - **Curves:**
 *   - batched_bezier_iterator
 *   - batched_cubic_bezier_iterator
 *   - batched_bspline_iterator
 *   - batched_catmull_rom_iterator
 * 
 * - **Shapes:**
 *   - batched_circle_iterator
 *   - batched_ellipse_iterator
 *   - batched_filled_circle_iterator
 *   - batched_filled_ellipse_iterator
 */
#pragma once

#include <type_traits>
#include <functional>

// Line iterators
#include <euler/dda/batched_line_iterator.hh>
#include <euler/dda/batched_thick_line_iterator.hh>

// Curve iterators
#include <euler/dda/batched_bezier_iterator.hh>
#include <euler/dda/batched_bspline_iterator.hh>

// Shape iterators
#include <euler/dda/batched_circle_iterator.hh>
#include <euler/dda/batched_ellipse_iterator.hh>

namespace euler::dda {

/**
 * @brief Type trait to check if a type is a batched iterator
 * @tparam T The type to check
 * 
 * Provides a compile-time check for batched iterator interface.
 * Usage: static_assert(is_batched_iterator_v<MyIterator>);
 */
template<typename T, typename = void>
struct is_batched_iterator : std::false_type {};

template<typename T>
struct is_batched_iterator<T, std::void_t<
    typename T::coord_type,
    typename T::point_type,
    typename T::pixel_type,
    typename T::batch_type,
    decltype(std::declval<const T&>().current_batch()),
    decltype(std::declval<const T&>().at_end()),
    decltype(std::declval<T&>().next_batch()),
    decltype(std::declval<T&>().process_all(
        std::declval<std::function<void(const typename T::batch_type&)>>()))
>> : std::true_type {};

template<typename T>
inline constexpr bool is_batched_iterator_v = is_batched_iterator<T>::value;

/**
 * @brief Generic batch processing function
 * @tparam Iterator Batched iterator type
 * @tparam Callback Callback function type
 * 
 * Provides a uniform way to process all pixels from any batched iterator.
 */
template<typename Iterator, typename Callback>
void process_batched_pixels(Iterator& iter, Callback&& callback) {
    static_assert(is_batched_iterator_v<Iterator>, 
                  "Iterator must be a batched iterator type");
    iter.process_all(std::forward<Callback>(callback));
}

/**
 * @brief Batch size recommendation based on pixel type
 * @tparam PixelType The pixel type
 * @return Recommended batch size for optimal performance
 */
template<typename PixelType>
constexpr size_t recommended_batch_size() {
    if constexpr (std::is_same_v<PixelType, pixel<int>>) {
        return 16; // Simple pixels - larger batches
    } else if constexpr (std::is_same_v<PixelType, aa_pixel<float>>) {
        return 8;  // AA pixels - more data per pixel
    } else {
        return 8;  // Default for custom types
    }
}

} // namespace euler::dda