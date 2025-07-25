#pragma once

#include <euler/angles/angle.hh>
#include <type_traits>

namespace euler {

// Forward declarations are not needed since degree and radian are type aliases

// Trait to check if a type is an angle
template<typename T>
struct is_angle : std::false_type {};

template<typename T, typename Unit>
struct is_angle<angle<T, Unit>> : std::true_type {};

template<typename T>
inline constexpr bool is_angle_v = is_angle<T>::value;

// Trait to check if a type is a degree angle
template<typename T>
struct is_degree : std::false_type {};

template<typename T>
struct is_degree<angle<T, degree_tag>> : std::true_type {};

template<typename T>
inline constexpr bool is_degree_v = is_degree<T>::value;

// Trait to check if a type is a radian angle
template<typename T>
struct is_radian : std::false_type {};

template<typename T>
struct is_radian<angle<T, radian_tag>> : std::true_type {};

template<typename T>
inline constexpr bool is_radian_v = is_radian<T>::value;

// Extract value type from angle
template<typename T>
struct angle_value_type {
    using type = T;
};

template<typename T, typename Unit>
struct angle_value_type<angle<T, Unit>> {
    using type = T;
};

template<typename T>
using angle_value_type_t = typename angle_value_type<T>::type;

// Extract unit type from angle
template<typename T>
struct angle_unit_type {
    using type = void;
};

template<typename T, typename Unit>
struct angle_unit_type<angle<T, Unit>> {
    using type = Unit;
};

template<typename T>
using angle_unit_type_t = typename angle_unit_type<T>::type;

// Check if two types have the same angle unit
template<typename T1, typename T2>
struct have_same_angle_unit : std::false_type {};

template<typename T, typename Unit>
struct have_same_angle_unit<angle<T, Unit>, angle<T, Unit>> : std::true_type {};

template<typename T1, typename T2>
inline constexpr bool have_same_angle_unit_v = have_same_angle_unit<T1, T2>::value;

// Angle type promotion rules (for mixed precision)
template<typename T1, typename T2>
struct angle_common_type {};

template<typename T1, typename T2, typename Unit>
struct angle_common_type<angle<T1, Unit>, angle<T2, Unit>> {
    using type = angle<std::common_type_t<T1, T2>, Unit>;
};

template<typename T1, typename T2>
using angle_common_type_t = typename angle_common_type<T1, T2>::type;

// Enable arithmetic operations only for same unit types
template<typename T1, typename T2>
struct enable_angle_arithmetic : std::false_type {};

template<typename T, typename Unit>
struct enable_angle_arithmetic<angle<T, Unit>, angle<T, Unit>> : std::true_type {};

template<typename T1, typename T2>
inline constexpr bool enable_angle_arithmetic_v = enable_angle_arithmetic<T1, T2>::value;

// Helper to make angle from raw value with same unit
template<typename AngleType, typename T>
constexpr auto make_angle_like(const AngleType& /*reference*/, T value) 
    -> std::enable_if_t<is_angle_v<AngleType>, 
                        angle<T, angle_unit_type_t<AngleType>>> {
    return angle<T, angle_unit_type_t<AngleType>>(value);
}

} // namespace euler