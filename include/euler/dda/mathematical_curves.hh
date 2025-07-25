/**
 * @file mathematical_curves.hh
 * @brief Common mathematical curves for use with curve iterators
 * @ingroup DDAModule
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/core/types.hh>
#include <euler/math/trigonometry.hh>
#include <euler/math/basic.hh>
#include <euler/dda/dda_math.hh>
#include <optional>
#include <functional>

namespace euler::dda::curves {

/**
 * @brief Cardioid curve: r = a(1 - cos(θ))
 * @param a Scale parameter
 * @param center Center point (default origin)
 * @return Polar curve function
 */
template<typename T>
auto cardioid(T a, point2<T> center = {0, 0}) {
    (void)center; // Center is used by polar_curve wrapper
    return [=](T theta) -> T {
        return a * (T(1) - cos(theta));
    };
}

/**
 * @brief Witch of Agnesi: y = 8a³/(x² + 4a²)
 * @param a Shape parameter
 * @return Cartesian curve function
 */
template<typename T>
auto agnesi_witch(T a) {
    return [=](T x) -> T {
        T a2 = a * a;
        return T(8) * a * a2 / (x * x + T(4) * a2);
    };
}

/**
 * @brief Lemniscate of Bernoulli: (x² + y²)² = a²(x² - y²)
 * @param a Scale parameter
 * @return Polar curve function (returns negative r for invalid angles)
 */
template<typename T>
auto lemniscate(T a) {
    return [=](T theta) -> T {
        T cos2theta = cos(T(2) * theta);
        if (cos2theta < 0) {
            return T(0);  // No point at this angle
        }
        return a * sqrt(cos2theta);
    };
}

/**
 * @brief Rose curve: r = a*cos(k*θ) or r = a*sin(k*θ)
 * @param a Amplitude
 * @param k Number of petals (k petals if k is odd, 2k if k is even)
 * @param use_sine Use sine instead of cosine
 * @return Polar curve function
 */
template<typename T>
std::function<T(T)> rose(T a, T k, bool use_sine = false) {
    if (use_sine) {
        return [=](T theta) -> T {
            return a * sin(k * theta);
        };
    } else {
        return [=](T theta) -> T {
            return a * cos(k * theta);
        };
    }
}

/**
 * @brief Archimedean spiral: r = a + b*θ
 * @param a Starting radius
 * @param b Growth rate
 * @return Polar curve function
 */
template<typename T>
auto archimedean_spiral(T a, T b) {
    return [=](T theta) -> T {
        return a + b * theta;
    };
}

/**
 * @brief Logarithmic spiral: r = a*e^(b*θ)
 * @param a Scale parameter
 * @param b Growth rate
 * @return Polar curve function
 */
template<typename T>
auto logarithmic_spiral(T a, T b) {
    return [=](T theta) -> T {
        return a * exp(b * theta);
    };
}

/**
 * @brief Fermat's spiral: r² = a²*θ
 * @param a Scale parameter
 * @param positive Take positive or negative branch
 * @return Polar curve function
 */
template<typename T>
auto fermat_spiral(T a, bool positive = true) {
    return [=](T theta) -> T {
        if (theta < 0) return T(0);
        T r2 = a * a * theta;
        return positive ? sqrt(r2) : -sqrt(r2);
    };
}

/**
 * @brief Astroid: x^(2/3) + y^(2/3) = a^(2/3)
 * @param a Radius parameter
 * @return Parametric curve function
 */
template<typename T>
auto astroid(T a, point2<T> center = {0, 0}) {
    return [=](T t) -> point2<T> {
        T cos3 = cos(t) * cos(t) * cos(t);
        T sin3 = sin(t) * sin(t) * sin(t);
        return point2<T>{center.x + a * cos3, center.y + a * sin3};
    };
}

/**
 * @brief Cycloid: curve traced by a point on a rolling circle
 * @param r Radius of rolling circle
 * @return Parametric curve function
 */
template<typename T>
auto cycloid(T r, point2<T> start = {0, 0}) {
    return [=](T t) -> point2<T> {
        return start + point2<T>{
            r * (t - sin(t)),
            r * (T(1) - cos(t))
        };
    };
}

/**
 * @brief Epicycloid: curve traced by a point on a circle rolling outside another circle
 * @param R Radius of fixed circle
 * @param r Radius of rolling circle
 * @return Parametric curve function
 */
template<typename T>
auto epicycloid(T R, T r, point2<T> center = {0, 0}) {
    return [=](T t) -> point2<T> {
        T ratio = (R + r) / r;
        return point2<T>{
            center.x + (R + r) * cos(t) - r * cos(ratio * t),
            center.y + (R + r) * sin(t) - r * sin(ratio * t)
        };
    };
}

/**
 * @brief Hypocycloid: curve traced by a point on a circle rolling inside another circle
 * @param R Radius of fixed circle
 * @param r Radius of rolling circle
 * @return Parametric curve function
 */
template<typename T>
auto hypocycloid(T R, T r, point2<T> center = {0, 0}) {
    return [=](T t) -> point2<T> {
        T ratio = (R - r) / r;
        return point2<T>{
            center.x + (R - r) * cos(t) + r * cos(ratio * t),
            center.y + (R - r) * sin(t) - r * sin(ratio * t)
        };
    };
}

/**
 * @brief Folium of Descartes: x³ + y³ = 3axy
 * @param a Shape parameter
 * @return Parametric curve function
 */
template<typename T>
auto folium_of_descartes(T a) {
    return [=](T t) -> point2<T> {
        T t3 = t * t * t;
        T denom = T(1) + t3;
        return {
            T(3) * a * t / denom,
            T(3) * a * t * t / denom
        };
    };
}

/**
 * @brief Limacon: r = a + b*cos(θ)
 * @param a Base radius
 * @param b Variation amplitude
 * @return Polar curve function
 */
template<typename T>
auto limacon(T a, T b) {
    return [=](T theta) -> T {
        return a + b * cos(theta);
    };
}

/**
 * @brief Parabola: y = ax²
 * @param a Shape parameter
 * @return Cartesian curve function
 */
template<typename T>
auto parabola(T a) {
    return [=](T x) -> T {
        return a * x * x;
    };
}

/**
 * @brief Hyperbola: y = a/x
 * @param a Shape parameter
 * @return Cartesian curve function
 */
template<typename T>
auto hyperbola(T a) {
    return [=](T x) -> T {
        if (abs(x) < std::numeric_limits<T>::epsilon()) {
            return std::numeric_limits<T>::infinity();
        }
        return a / x;
    };
}

/**
 * @brief Cissoid of Diocles: r = 2a*sin(θ)*tan(θ)
 * @param a Shape parameter
 * @return Polar curve function
 */
template<typename T>
auto cissoid_of_diocles(T a) {
    return [=](T theta) -> T {
        if (abs(cos(theta)) < std::numeric_limits<T>::epsilon()) {
            return T(0);
        }
        return T(2) * a * sin(theta) * tan(theta);
    };
}

/**
 * @brief Conchoid of Nicomedes: r = a + b*sec(θ)
 * @param a Base distance
 * @param b Loop size
 * @return Polar curve function
 */
template<typename T>
auto conchoid(T a, T b) {
    return [=](T theta) -> T {
        T costheta = cos(theta);
        if (abs(costheta) < std::numeric_limits<T>::epsilon()) {
            return std::numeric_limits<T>::infinity();
        }
        return a + b / costheta;
    };
}

/**
 * @brief Sine wave: y = a*sin(b*x + c)
 * @param amplitude Wave amplitude
 * @param frequency Wave frequency
 * @param phase Phase shift
 * @return Cartesian curve function
 */
template<typename T>
auto sine_wave(T amplitude, T frequency, T phase = T(0)) {
    return [=](T x) -> T {
        return amplitude * sin(frequency * x + phase);
    };
}

/**
 * @brief Damped sine wave: y = a*e^(-b*x)*sin(c*x)
 * @param amplitude Initial amplitude
 * @param damping Damping factor
 * @param frequency Oscillation frequency
 * @return Cartesian curve function
 */
template<typename T>
auto damped_sine_wave(T amplitude, T damping, T frequency) {
    return [=](T x) -> T {
        return amplitude * exp(-damping * x) * sin(frequency * x);
    };
}

} // namespace euler::dda::curves