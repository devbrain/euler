/**
 * @file angle_traits.hh
 * @brief Type traits and metaprogramming utilities for angle types
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides compile-time type traits and metaprogramming utilities
 * for working with angle types. It includes:
 * - Type detection traits (is_angle, is_degree, is_radian)
 * - Type extraction utilities (value_type, unit_type)
 * - Type manipulation utilities (common_type, enable_if helpers)
 * - Factory functions for creating angles with deduced types
 * 
 * These traits are essential for writing generic code that works with
 * angle types without knowing the specific unit at compile time.
 * 
 * @section angle_traits_usage Usage Example
 * @code{.cpp}
 * #include <euler/angles/angle_traits.hh>
 * 
 * using namespace euler;
 * 
 * // Type detection
 * static_assert(is_angle_v<degree<float>>);  // true
 * static_assert(is_degree_v<degree<float>>); // true
 * static_assert(is_radian_v<degree<float>>); // false
 * 
 * // Extract value type
 * using value_t = angle_value_type_t<degree<float>>;  // float
 * 
 * // Extract unit type
 * using unit_t = angle_unit_type_t<radian<double>>;  // radian_tag
 * 
 * // Generic function that works with any angle type
 * template<typename AngleType>
 * auto double_angle(const AngleType& a) -> std::enable_if_t<is_angle_v<AngleType>, AngleType> {
 *     return a * 2;
 * }
 * 
 * // Create angle with same unit as reference
 * auto ref = degree<float>(45.0f);
 * auto same_unit = make_angle_like(ref, 90.0);  // degree<double>(90.0)
 * @endcode
 */
#pragma once

#include <euler/angles/angle.hh>
#include <type_traits>

namespace euler {

// Forward declarations are not needed since degree and radian are type aliases

/**
 * @brief Type trait to check if a type is an angle
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Primary template that evaluates to std::false_type for non-angle types.
 * Specializations for angle types evaluate to std::true_type.
 * 
 * @see is_angle_v for the convenient variable template
 */
template<typename T>
struct is_angle : std::false_type {};

/**
 * @brief Specialization for angle types
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 */
template<typename T, typename Unit>
struct is_angle<angle<T, Unit>> : std::true_type {};

/**
 * @brief Helper variable template for is_angle
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Convenient variable template that provides the value of is_angle<T>.
 * Use this in if constexpr or template constraints.
 * 
 * @example
 * @code
 * if constexpr (is_angle_v<T>) {
 *     // T is an angle type
 * }
 * @endcode
 */
template<typename T>
inline constexpr bool is_angle_v = is_angle<T>::value;

/**
 * @brief Type trait to check if a type is a degree angle
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Primary template that evaluates to std::false_type for non-degree types.
 * The specialization for degree angles evaluates to std::true_type.
 * 
 * @see is_degree_v for the convenient variable template
 */
template<typename T>
struct is_degree : std::false_type {};

/**
 * @brief Specialization for degree angle types
 * @ingroup AnglesModule
 * @tparam T The numeric type
 */
template<typename T>
struct is_degree<angle<T, degree_tag>> : std::true_type {};

/**
 * @brief Helper variable template for is_degree
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Convenient variable template that provides the value of is_degree<T>.
 * Useful for compile-time branching based on angle units.
 * 
 * @example
 * @code
 * template<typename AngleType>
 * void process_angle(const AngleType& a) {
 *     if constexpr (is_degree_v<AngleType>) {
 *         // Special handling for degrees
 *     }
 * }
 * @endcode
 */
template<typename T>
inline constexpr bool is_degree_v = is_degree<T>::value;

/**
 * @brief Type trait to check if a type is a radian angle
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Primary template that evaluates to std::false_type for non-radian types.
 * The specialization for radian angles evaluates to std::true_type.
 * 
 * @see is_radian_v for the convenient variable template
 */
template<typename T>
struct is_radian : std::false_type {};

/**
 * @brief Specialization for radian angle types
 * @ingroup AnglesModule
 * @tparam T The numeric type
 */
template<typename T>
struct is_radian<angle<T, radian_tag>> : std::true_type {};

/**
 * @brief Helper variable template for is_radian
 * @ingroup AnglesModule
 * @tparam T The type to check
 * 
 * @details
 * Convenient variable template that provides the value of is_radian<T>.
 * Useful for compile-time branching based on angle units.
 * 
 * @example
 * @code
 * template<typename AngleType>
 * auto sin_angle(const AngleType& a) {
 *     if constexpr (is_radian_v<AngleType>) {
 *         return std::sin(a.value());  // Direct computation
 *     } else {
 *         return std::sin(to_radians(a).value());  // Convert first
 *     }
 * }
 * @endcode
 */
template<typename T>
inline constexpr bool is_radian_v = is_radian<T>::value;

/**
 * @brief Extract numeric value type from angle or other types
 * @ingroup AnglesModule
 * @tparam T The type to extract from
 * 
 * @details
 * For angle types, extracts the underlying numeric type (e.g., float, double).
 * For non-angle types, returns the type itself. This allows generic code to
 * work with both angle types and raw numeric types.
 * 
 * @see angle_value_type_t for the convenient alias template
 */
template<typename T>
struct angle_value_type {
    /** @brief The extracted type (T itself for non-angles) */
    using type = T;
};

/**
 * @brief Specialization for angle types
 * @ingroup AnglesModule
 * @tparam T The numeric type of the angle
 * @tparam Unit The angle unit tag
 */
template<typename T, typename Unit>
struct angle_value_type<angle<T, Unit>> {
    /** @brief The underlying numeric type */
    using type = T;
};

/**
 * @brief Helper alias template for angle_value_type
 * @ingroup AnglesModule
 * @tparam T The type to extract from
 * 
 * @details
 * Convenient alias template that provides the type from angle_value_type<T>.
 * 
 * @example
 * @code
 * using float_t = angle_value_type_t<degree<float>>;   // float
 * using double_t = angle_value_type_t<radian<double>>; // double
 * using int_t = angle_value_type_t<int>;               // int
 * @endcode
 */
template<typename T>
using angle_value_type_t = typename angle_value_type<T>::type;

/**
 * @brief Extract unit tag type from angle or other types
 * @ingroup AnglesModule
 * @tparam T The type to extract from
 * 
 * @details
 * For angle types, extracts the unit tag (degree_tag or radian_tag).
 * For non-angle types, the type member is void. This allows generic code
 * to detect and work with angle units.
 * 
 * @see angle_unit_type_t for the convenient alias template
 */
template<typename T>
struct angle_unit_type {
    /** @brief void for non-angle types */
    using type = void;
};

/**
 * @brief Specialization for angle types
 * @ingroup AnglesModule
 * @tparam T The numeric type of the angle
 * @tparam Unit The angle unit tag
 */
template<typename T, typename Unit>
struct angle_unit_type<angle<T, Unit>> {
    /** @brief The unit tag type (degree_tag or radian_tag) */
    using type = Unit;
};

/**
 * @brief Helper alias template for angle_unit_type
 * @ingroup AnglesModule
 * @tparam T The type to extract from
 * 
 * @details
 * Convenient alias template that provides the type from angle_unit_type<T>.
 * 
 * @example
 * @code
 * using deg_unit = angle_unit_type_t<degree<float>>;   // degree_tag
 * using rad_unit = angle_unit_type_t<radian<double>>;  // radian_tag
 * using no_unit = angle_unit_type_t<float>;            // void
 * @endcode
 */
template<typename T>
using angle_unit_type_t = typename angle_unit_type<T>::type;

/**
 * @brief Check if two angle types have the same unit
 * @ingroup AnglesModule
 * @tparam T1 First type to compare
 * @tparam T2 Second type to compare
 * 
 * @details
 * This trait checks if two angle types use the same unit tag (both degrees
 * or both radians). The numeric types can be different. Non-angle types
 * always evaluate to false.
 * 
 * @note This trait is more restrictive than necessary - it requires the same
 *       numeric type. Consider using std::is_same on angle_unit_type_t instead.
 * 
 * @see have_same_angle_unit_v for the convenient variable template
 */
template<typename T1, typename T2>
struct have_same_angle_unit : std::false_type {};

/**
 * @brief Specialization for angles with same numeric type and unit
 * @ingroup AnglesModule
 * @tparam T The numeric type (must be the same for both)
 * @tparam Unit The angle unit tag
 */
template<typename T, typename Unit>
struct have_same_angle_unit<angle<T, Unit>, angle<T, Unit>> : std::true_type {};

/**
 * @brief Helper variable template for have_same_angle_unit
 * @ingroup AnglesModule
 * @tparam T1 First type to compare
 * @tparam T2 Second type to compare
 * 
 * @warning This trait requires both the numeric type and unit to match.
 *          For checking only unit compatibility, consider:
 *          `std::is_same_v<angle_unit_type_t<T1>, angle_unit_type_t<T2>>`
 */
template<typename T1, typename T2>
inline constexpr bool have_same_angle_unit_v = have_same_angle_unit<T1, T2>::value;

/**
 * @brief Determine common type for mixed-precision angle arithmetic
 * @ingroup AnglesModule
 * @tparam T1 First angle type
 * @tparam T2 Second angle type
 * 
 * @details
 * This trait determines the result type when performing arithmetic operations
 * on angles with different numeric precision. It uses std::common_type to
 * find the appropriate numeric type while preserving the angle unit.
 * 
 * The primary template is left undefined to catch incompatible angle units
 * at compile time.
 * 
 * @see angle_common_type_t for the convenient alias template
 */
template<typename T1, typename T2>
struct angle_common_type {};

/**
 * @brief Specialization for angles with same unit but different precision
 * @ingroup AnglesModule
 * @tparam T1 First numeric type
 * @tparam T2 Second numeric type  
 * @tparam Unit The shared angle unit tag
 */
template<typename T1, typename T2, typename Unit>
struct angle_common_type<angle<T1, Unit>, angle<T2, Unit>> {
    /** @brief Angle type with promoted numeric type */
    using type = angle<std::common_type_t<T1, T2>, Unit>;
};

/**
 * @brief Helper alias template for angle_common_type
 * @ingroup AnglesModule
 * @tparam T1 First angle type
 * @tparam T2 Second angle type
 * 
 * @details
 * Convenient alias template that provides the type from angle_common_type<T1, T2>.
 * 
 * @example
 * @code
 * auto a = degree<float>(45.0f);
 * auto b = degree<double>(30.0);
 * using result_t = angle_common_type_t<decltype(a), decltype(b)>;
 * // result_t is degree<double>
 * @endcode
 */
template<typename T1, typename T2>
using angle_common_type_t = typename angle_common_type<T1, T2>::type;

/**
 * @brief Enable arithmetic operations only for compatible angle types
 * @ingroup AnglesModule
 * @tparam T1 First angle type
 * @tparam T2 Second angle type
 * 
 * @details
 * This trait is used to constrain angle arithmetic operations to compatible
 * types. It ensures that operations like addition and subtraction are only
 * allowed between angles with the same unit.
 * 
 * @note This trait is overly restrictive - it requires the same numeric type.
 *       Consider using SFINAE with angle_unit_type_t comparison instead.
 * 
 * @see enable_angle_arithmetic_v for the convenient variable template
 */
template<typename T1, typename T2>
struct enable_angle_arithmetic : std::false_type {};

/**
 * @brief Specialization for angles with same type and unit
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 */
template<typename T, typename Unit>
struct enable_angle_arithmetic<angle<T, Unit>, angle<T, Unit>> : std::true_type {};

/**
 * @brief Helper variable template for enable_angle_arithmetic
 * @ingroup AnglesModule
 * @tparam T1 First angle type
 * @tparam T2 Second angle type
 * 
 * @details
 * Use this in template constraints or if constexpr to enable operations
 * only for compatible angle types.
 * 
 * @example
 * @code
 * template<typename T1, typename T2>
 * auto add_angles(const T1& a, const T2& b) 
 *     -> std::enable_if_t<enable_angle_arithmetic_v<T1, T2>, T1> {
 *     return a + b;
 * }
 * @endcode
 */
template<typename T1, typename T2>
inline constexpr bool enable_angle_arithmetic_v = enable_angle_arithmetic<T1, T2>::value;

/**
 * @brief Create an angle with the same unit as a reference angle
 * @ingroup AnglesModule
 * @tparam AngleType The reference angle type (must satisfy is_angle_v)
 * @tparam T The numeric type for the new angle
 * @param reference Reference angle (only used for type deduction)
 * @param value The numeric value for the new angle
 * @return New angle with same unit as reference but potentially different numeric type
 * 
 * @details
 * This factory function creates a new angle with the same unit (degrees or radians)
 * as a reference angle, but allows using a different numeric type. The reference
 * angle parameter is not used at runtime - it exists only for type deduction.
 * 
 * This is useful in generic code where you need to create angles with a consistent
 * unit but varying precision.
 * 
 * @note The function is SFINAE-enabled and only participates in overload resolution
 *       when AngleType is actually an angle type.
 * 
 * @example
 * @code
 * auto ref = degree<float>(45.0f);
 * auto same_unit_double = make_angle_like(ref, 90.0);  // degree<double>(90.0)
 * auto same_unit_int = make_angle_like(ref, 180);      // degree<int>(180)
 * 
 * // In generic code:
 * template<typename AngleType>
 * auto double_angle(const AngleType& a) {
 *     auto value = a.value();
 *     return make_angle_like(a, value * 2);
 * }
 * @endcode
 */
template<typename AngleType, typename T>
constexpr auto make_angle_like(const AngleType& /*reference*/, T value) 
    -> std::enable_if_t<is_angle_v<AngleType>, 
                        angle<T, angle_unit_type_t<AngleType>>> {
    return angle<T, angle_unit_type_t<AngleType>>(value);
}

} // namespace euler