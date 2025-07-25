/**
 * @file dda_traits.hh
 * @brief Common traits, concepts, and types for DDA module
 * @ingroup DDAModule
 * 
 * This header defines the fundamental types and traits used throughout
 * the DDA module. It provides:
 * - Basic pixel types for rasterization output
 * - Type traits for iterator validation
 * - Common enumerations for algorithm configuration
 * - Base classes for iterator implementation
 * 
 * @see dda.hh for the main module interface
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/core/types.hh>
#include <iterator>
#include <type_traits>

namespace euler::dda {

/**
 * @brief Basic pixel output for DDA iterators
 * @tparam T Coordinate type (typically int)
 * 
 * This is the fundamental output type for all non-antialiased DDA
 * iterators. It contains only the pixel position, making it lightweight
 * and efficient for basic rasterization tasks.
 * 
 * @note The implicit conversion to point2<T> allows pixels to be used
 *       directly in point-based operations.
 */
template<typename T>
struct pixel {
    point2<T> pos;  ///< Pixel position in screen coordinates
    
    /// Implicit conversion to point for convenience
    constexpr operator point2<T>() const { return pos; }
    
    /// Element access
    constexpr T& x() { return pos.x; }
    constexpr T& y() { return pos.y; }
    constexpr const T& x() const { return pos.x; }
    constexpr const T& y() const { return pos.y; }
};

/**
 * @brief Antialiased pixel output with coverage information
 * @tparam T Coordinate type
 */
template<typename T>
struct aa_pixel {
    point2<T> pos;      ///< Pixel position
    float coverage;     ///< Coverage/alpha value [0.0, 1.0]
    float distance;     ///< Distance to ideal curve (optional, for advanced AA)
    
    /// Implicit conversion to point
    constexpr operator point2<T>() const { return pos; }
};

/**
 * @brief Horizontal span for filled shapes
 */
struct span {
    int y;          ///< Y coordinate
    int x_start;    ///< Starting X coordinate
    int x_end;      ///< Ending X coordinate (inclusive)
    
    /// Get span width
    constexpr int width() const { return x_end - x_start + 1; }
};

/**
 * @brief Type trait for DDA iterators
 */
template<typename Iterator, typename = void>
struct is_dda_iterator : std::false_type {};

template<typename Iterator>
struct is_dda_iterator<Iterator, std::void_t<
    typename Iterator::value_type,
    typename Iterator::point_type,
    typename Iterator::coord_type,
    typename Iterator::iterator_category,
    decltype(*std::declval<Iterator>()),
    decltype(++std::declval<Iterator&>()),
    decltype(std::declval<Iterator>() != std::declval<Iterator>())
>> : std::true_type {};

template<typename Iterator>
inline constexpr bool is_dda_iterator_v = is_dda_iterator<Iterator>::value;

/**
 * @brief Type trait for curve functions
 */
template<typename F, typename T, typename = void>
struct is_curve_function : std::false_type {};

template<typename F, typename T>
struct is_curve_function<F, T, std::void_t<
    decltype(std::declval<F>()(std::declval<T>()))
>> : std::conditional_t<
    std::is_convertible_v<decltype(std::declval<F>()(std::declval<T>())), point2<T>>,
    std::true_type,
    std::false_type
> {};

template<typename F, typename T>
inline constexpr bool is_curve_function_v = is_curve_function<F, T>::value;

/**
 * @brief Type trait for Cartesian curve functions (y = f(x))
 */
template<typename F, typename T, typename = void>
struct is_cartesian_curve_function : std::false_type {};

template<typename F, typename T>
struct is_cartesian_curve_function<F, T, std::void_t<
    decltype(std::declval<F>()(std::declval<T>()))
>> : std::conditional_t<
    std::is_convertible_v<decltype(std::declval<F>()(std::declval<T>())), T>,
    std::true_type,
    std::false_type
> {};

template<typename F, typename T>
inline constexpr bool is_cartesian_curve_function_v = is_cartesian_curve_function<F, T>::value;

/**
 * @brief Type trait for polar curve functions (r = f(theta))
 */
template<typename F, typename T, typename = void>
struct is_polar_curve_function : std::false_type {};

template<typename F, typename T>
struct is_polar_curve_function<F, T, std::void_t<
    decltype(std::declval<F>()(std::declval<T>()))
>> : std::conditional_t<
    std::is_convertible_v<decltype(std::declval<F>()(std::declval<T>())), T>,
    std::true_type,
    std::false_type
> {};

template<typename F, typename T>
inline constexpr bool is_polar_curve_function_v = is_polar_curve_function<F, T>::value;

/**
 * @brief Curve representation types
 */
enum class curve_type {
    parametric,  ///< (x,y) = f(t)
    cartesian,   ///< y = f(x)
    polar        ///< r = f(theta)
};

/**
 * @brief Line end cap styles
 */
enum class cap_style {
    butt,    ///< Square end at exact endpoint
    round,   ///< Rounded end cap
    square   ///< Square end cap extending beyond endpoint
};

/**
 * @brief Antialiasing algorithm selection
 */
enum class aa_algorithm {
    wu,             ///< Wu's algorithm (fast, good quality)
    gupta_sproull,  ///< Gupta-Sproull (distance-based)
    supersampling   ///< Supersampling (highest quality, slower)
};

/**
 * @brief Default tolerance values for different types
 */
template<typename T>
constexpr T default_tolerance() {
    if constexpr (std::is_same_v<T, float>) {
        return T(0.5);
    } else if constexpr (std::is_same_v<T, double>) {
        return T(0.25);
    } else {
        return T(1);
    }
}

/**
 * @brief Sentinel type for DDA iterators
 */
struct dda_sentinel {};

/**
 * @brief Base class for DDA iterators (CRTP)
 */
template<typename Derived, typename ValueType, typename CoordType>
class dda_iterator_base {
public:
    using value_type = ValueType;
    using coord_type = CoordType;
    using point_type = point2<CoordType>;
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;
    
protected:
    bool done_ = false;
    
public:
    /// Check if iterator is at end
    constexpr bool is_done() const { return done_; }
    
    /// Comparison with sentinel
    friend constexpr bool operator==(const Derived& lhs, dda_sentinel) {
        return lhs.is_done();
    }
    
    friend constexpr bool operator!=(const Derived& lhs, dda_sentinel) {
        return !lhs.is_done();
    }
    
    /// Comparison with other iterator
    friend constexpr bool operator==(const Derived& lhs, const Derived& rhs) {
        return lhs.done_ == rhs.done_;
    }
    
    friend constexpr bool operator!=(const Derived& lhs, const Derived& rhs) {
        return !(lhs == rhs);
    }
};

/**
 * @brief Rectangle for clipping operations
 */
template<typename T>
struct rectangle {
    point2<T> min;  ///< Top-left corner
    point2<T> max;  ///< Bottom-right corner
    
    /// Check if point is inside rectangle
    constexpr bool contains(const point2<T>& p) const {
        return p.x >= min.x && p.x <= max.x && 
               p.y >= min.y && p.y <= max.y;
    }
    
    /// Clip line segment to rectangle (Cohen-Sutherland)
    bool clip_line(point2<T>& p1, point2<T>& p2) const;
};

} // namespace euler::dda