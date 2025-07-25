#pragma once

#include <euler/core/types.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/degree.hh>
#include <cmath>
#include <complex>

namespace euler {

template<typename T>
class complex {
    static_assert(std::is_arithmetic_v<T>, 
                  "Complex value type must be arithmetic type");
public:
    using value_type = T;
    
    // Constructors
    constexpr complex() = default;
    constexpr complex(T real) : real_(real), imag_(0) {}
    constexpr complex(T real, T imag) : real_(real), imag_(imag) {}
    
    // Copy and move constructors
    constexpr complex(const complex&) = default;
    constexpr complex(complex&&) = default;
    
    // Assignment operators
    complex& operator=(const complex&) = default;
    complex& operator=(complex&&) = default;
    
    // Polar construction with angles
    template<typename Unit>
    static complex polar(T magnitude, const angle<T, Unit>& phase) {
        radian<T> phase_rad(phase);
        return complex(
            magnitude * std::cos(phase_rad.value()),
            magnitude * std::sin(phase_rad.value())
        );
    }
    
    // Polar construction with raw radians
    static complex polar(T magnitude, T phase_radians) {
        return complex(
            magnitude * std::cos(phase_radians),
            magnitude * std::sin(phase_radians)
        );
    }
    
    // Component access
    constexpr T real() const { return real_; }
    constexpr T imag() const { return imag_; }
    
    T& real() { return real_; }
    T& imag() { return imag_; }
    
    // Polar access
    T abs() const { 
        return std::sqrt(real_ * real_ + imag_ * imag_); 
    }
    
    T norm() const { 
        return real_ * real_ + imag_ * imag_; 
    }
    
    radian<T> arg() const { 
        return radian<T>(std::atan2(imag_, real_)); 
    }
    
    degree<T> arg_deg() const { 
        return degree<T>(arg()); 
    }
    
    // Arithmetic operators
    complex& operator+=(const complex& rhs) {
        real_ += rhs.real_;
        imag_ += rhs.imag_;
        return *this;
    }
    
    complex& operator-=(const complex& rhs) {
        real_ -= rhs.real_;
        imag_ -= rhs.imag_;
        return *this;
    }
    
    complex& operator*=(const complex& rhs) {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        T new_real = real_ * rhs.real_ - imag_ * rhs.imag_;
        T new_imag = real_ * rhs.imag_ + imag_ * rhs.real_;
        real_ = new_real;
        imag_ = new_imag;
        return *this;
    }
    
    complex& operator/=(const complex& rhs) {
        // (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
        T denominator = rhs.real_ * rhs.real_ + rhs.imag_ * rhs.imag_;
        T new_real = (real_ * rhs.real_ + imag_ * rhs.imag_) / denominator;
        T new_imag = (imag_ * rhs.real_ - real_ * rhs.imag_) / denominator;
        real_ = new_real;
        imag_ = new_imag;
        return *this;
    }
    
    // Scalar multiplication/division
    complex& operator*=(T ascalar) {
        real_ *= ascalar;
        imag_ *= ascalar;
        return *this;
    }
    
    complex& operator/=(T ascalar) {
        real_ /= ascalar;
        imag_ /= ascalar;
        return *this;
    }
    
    // Unary operators
    constexpr complex operator+() const { return *this; }
    constexpr complex operator-() const { return complex(-real_, -imag_); }
    
    // Conversion from std::complex
    template<typename U>
    complex(const std::complex<U>& c) 
        : real_(static_cast<T>(c.real()))
        , imag_(static_cast<T>(c.imag())) {}
    
    // Conversion to std::complex
    operator std::complex<T>() const {
        return std::complex<T>(real_, imag_);
    }
    
    // Comparison operators
    constexpr bool operator==(const complex& rhs) const {
        return real_ == rhs.real_ && imag_ == rhs.imag_;
    }
    
    constexpr bool operator!=(const complex& rhs) const {
        return !(*this == rhs);
    }
    
private:
    T real_{};
    T imag_{};
};

// Type aliases
using complexf = complex<float>;
using complexd = complex<double>;

// Literal operators
inline namespace literals {
    constexpr complexf operator""_i(long double value) {
        return complexf(0, static_cast<float>(value));
    }
    
    constexpr complexf operator""_if(long double value) {
        return complexf(0, static_cast<float>(value));
    }
    
    constexpr complexd operator""_id(long double value) {
        return complexd(0, static_cast<double>(value));
    }
    
    constexpr complexf operator""_i(unsigned long long value) {
        return complexf(0, static_cast<float>(value));
    }
}

// Binary arithmetic operators
template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
    return complex<T>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
    return complex<T>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
}

template<typename T>
complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
    complex<T> result = lhs;
    result *= rhs;
    return result;
}

template<typename T>
complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
    complex<T> result = lhs;
    result /= rhs;
    return result;
}

// Mixed real-complex operations
template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() + rhs, lhs.imag());
}

template<typename T>
constexpr complex<T> operator+(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs + rhs.real(), rhs.imag());
}

template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() - rhs, lhs.imag());
}

template<typename T>
constexpr complex<T> operator-(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs - rhs.real(), -rhs.imag());
}

template<typename T>
complex<T> operator*(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() * rhs, lhs.imag() * rhs);
}

template<typename T>
complex<T> operator*(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs * rhs.real(), lhs * rhs.imag());
}

template<typename T>
complex<T> operator/(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() / rhs, lhs.imag() / rhs);
}

template<typename T>
complex<T> operator/(T lhs, const complex<T>& rhs) {
    // lhs / (a + bi) = lhs * (a - bi) / (a² + b²)
    T denominator = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
    return complex<T>(
        lhs * rhs.real() / denominator,
        -lhs * rhs.imag() / denominator
    );
}

} // namespace euler