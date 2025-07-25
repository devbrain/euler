#pragma once

#include <euler/core/types.hh>
#include <euler/complex/complex.hh>
#include <euler/random/random.hh>
#include <euler/random/random_angle.hh>
#include <euler/math/trigonometry.hh>
#include <euler/math/basic.hh>

namespace euler {

// ============================================================================
// Random Complex Number Generation
// ============================================================================

// Generate random complex number on unit circle
template<typename T, typename Generator>
complex<T> random_complex_unit(Generator& g) {
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
    return complex<T>(cos(theta), sin(theta));
}

// Generate random complex number in rectangle [real_min, real_max] × [imag_min, imag_max]
template<typename T, typename Generator>
complex<T> random_complex(Generator& g, T real_min, T real_max, T imag_min, T imag_max) {
    T real = g.uniform(real_min, real_max);
    T imag = g.uniform(imag_min, imag_max);
    return complex<T>(real, imag);
}

// Generate random complex number in square [-size, size] × [-size, size]
template<typename T, typename Generator>
complex<T> random_complex(Generator& g, T size = T(1)) {
    return random_complex(g, -size, size, -size, size);
}

// Generate random complex number in disk of given radius
template<typename T, typename Generator>
complex<T> random_complex_disk(Generator& g, T radius = T(1)) {
    // Use rejection sampling for uniform distribution
    T x, y;
    do {
        x = g.uniform(T(-1), T(1));
        y = g.uniform(T(-1), T(1));
    } while (x*x + y*y > T(1));
    
    return complex<T>(x * radius, y * radius);
}

// Generate random complex number in annulus (ring)
template<typename T, typename Generator>
complex<T> random_complex_annulus(Generator& g, T inner_radius, T outer_radius) {
    EULER_CHECK(inner_radius >= T(0) && inner_radius <= outer_radius,
                error_code::invalid_argument,
                "random_complex_annulus: invalid radius range");
    
    // Generate radius with proper distribution
    T r_squared = g.uniform(inner_radius * inner_radius, outer_radius * outer_radius);
    T r = sqrt(r_squared);
    
    // Generate angle
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
    
    return complex<T>::polar(r, theta);
}

// Generate random complex number with normal distribution
template<typename T, typename Generator>
complex<T> random_complex_normal(Generator& g, 
                                const complex<T>& mean = complex<T>(0, 0),
                                T stddev = T(1)) {
    T real = g.normal(mean.real(), stddev);
    T imag = g.normal(mean.imag(), stddev);
    return complex<T>(real, imag);
}

// Generate random complex number with independent normal distributions for real/imag
template<typename T, typename Generator>
complex<T> random_complex_normal(Generator& g,
                                const complex<T>& mean,
                                T real_stddev,
                                T imag_stddev) {
    T real = g.normal(mean.real(), real_stddev);
    T imag = g.normal(mean.imag(), imag_stddev);
    return complex<T>(real, imag);
}

// Generate random complex number with given magnitude and random phase
template<typename T, typename Generator>
complex<T> random_complex_fixed_magnitude(Generator& g, T magnitude) {
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
    return complex<T>::polar(magnitude, theta);
}

// Generate random complex number with given phase and random magnitude
template<typename T, typename Generator>
complex<T> random_complex_fixed_phase(Generator& g, 
                                     const angle<T, radian_tag>& phase,
                                     T min_magnitude = T(0),
                                     T max_magnitude = T(1)) {
    T magnitude = g.uniform(min_magnitude, max_magnitude);
    return complex<T>::polar(magnitude, phase);
}

// Generate random roots of unity (n-th roots of 1)
template<typename T, typename Generator>
complex<T> random_root_of_unity(Generator& g, int n) {
    EULER_CHECK(n > 0, error_code::invalid_argument,
                "random_root_of_unity: n must be positive");
    
    int k = g.uniform(0, n - 1);
    T angle_rad = T(2) * constants<T>::pi * T(k) / T(n);
    return complex<T>(cos(radian<T>(angle_rad)), sin(radian<T>(angle_rad)));
}

// Generate complex number with random magnitude (log-normal distribution)
template<typename T, typename Generator>
complex<T> random_complex_log_normal(Generator& g,
                                    T log_mean = T(0),
                                    T log_stddev = T(1)) {
    T log_magnitude = g.normal(log_mean, log_stddev);
    T magnitude = exp(log_magnitude);
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
    return complex<T>::polar(magnitude, theta);
}

// ============================================================================
// Convenience functions using thread-local generator
// ============================================================================

template<typename T>
inline complex<T> random_complex_unit() {
    return random_complex_unit<T>(thread_local_rng());
}

template<typename T>
inline complex<T> random_complex(T size = T(1)) {
    return random_complex(thread_local_rng(), size);
}

template<typename T>
inline complex<T> random_complex_disk(T radius = T(1)) {
    return random_complex_disk(thread_local_rng(), radius);
}

template<typename T>
inline complex<T> random_complex_normal(const complex<T>& mean = complex<T>(0, 0),
                                       T stddev = T(1)) {
    return random_complex_normal(thread_local_rng(), mean, stddev);
}

} // namespace euler