#pragma once

namespace euler {

// Operation types for scalar-first operations
struct scalar_div_op {
    template<typename T1, typename T2>
    static constexpr auto apply(const T1& s, const T2& value) {
        return s / value;
    }
};

struct scalar_sub_op {
    template<typename T1, typename T2>
    static constexpr auto apply(const T1& s, const T2& value) {
        return s - value;
    }
};

struct scalar_pow_op {
    template<typename T1, typename T2>
    static auto apply(const T1& base, const T2& exponent) {
        using std::pow;
        return pow(base, exponent);
    }
};

} // namespace euler