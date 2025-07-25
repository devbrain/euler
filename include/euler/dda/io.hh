/**
 * @file io.hh
 * @brief I/O stream operators for DDA types with pretty printing
 * @ingroup DDAModule
 * 
 * This header provides stream output operators for all DDA-specific types
 * with support for the pretty printing infrastructure from io/io.hh.
 * The operators respect stream formatting flags like width and precision.
 * 
 * @section usage Usage
 * @code
 * pixel<int> p{10, 20};
 * std::cout << p;                           // Output: (10, 20)
 * 
 * aa_pixel<float> aa{10.5f, 20.5f, 0.75f, 0.25f};
 * std::cout << aa;                          // Output: (10.5, 20.5) [coverage: 0.75, distance: 0.25]
 * 
 * span s{100, 10, 50};
 * std::cout << s;                           // Output: span(y: 100, x: [10, 50])
 * 
 * rectangle<int> rect{{0, 0}, {100, 100}};
 * std::cout << rect;                        // Output: [(0, 0) - (100, 100)]
 * @endcode
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/coordinates/io.hh>
#include <euler/io/io.hh>
#include <ostream>
#include <iomanip>

namespace euler::dda {

// Stream output for pixel type
template<typename T>
std::ostream& operator<<(std::ostream& os, const pixel<T>& p) {
    // Pixels output the same as points since they implicitly convert
    return os << p.pos;
}

// Stream output for antialiased pixel type
template<typename T>
std::ostream& operator<<(std::ostream& os, const aa_pixel<T>& p) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    
    // Output position
    os << p.pos << " [";
    
    // Output coverage with appropriate precision
    os << "coverage: ";
    if (precision > 0) {
        os << std::fixed << std::setprecision(static_cast<int>(precision));
    }
    os << p.coverage;
    
    // Output distance if non-zero
    if (p.distance > 0.001f) {
        os << ", distance: " << p.distance;
    }
    
    os << ']';
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

// Stream output for span type
std::ostream& operator<<(std::ostream& os, const span& s) {
    return os << "span(y: " << s.y << ", x: [" << s.x_start << ", " << s.x_end << "])";
}

// Stream output for rectangle type
template<typename T>
std::ostream& operator<<(std::ostream& os, const rectangle<T>& r) {
    return os << '[' << r.min << " - " << r.max << ']';
}

// Stream output for curve_type enum
inline std::ostream& operator<<(std::ostream& os, curve_type type) {
    switch (type) {
        case curve_type::parametric:
            return os << "parametric";
        case curve_type::cartesian:
            return os << "cartesian";
        case curve_type::polar:
            return os << "polar";
        default:
            return os << "unknown";
    }
}

// Stream output for cap_style enum
inline std::ostream& operator<<(std::ostream& os, cap_style style) {
    switch (style) {
        case cap_style::butt:
            return os << "butt";
        case cap_style::round:
            return os << "round";
        case cap_style::square:
            return os << "square";
        default:
            return os << "unknown";
    }
}

// Stream output for aa_algorithm enum
inline std::ostream& operator<<(std::ostream& os, aa_algorithm algo) {
    switch (algo) {
        case aa_algorithm::wu:
            return os << "wu";
        case aa_algorithm::gupta_sproull:
            return os << "gupta_sproull";
        case aa_algorithm::supersampling:
            return os << "supersampling";
        default:
            return os << "unknown";
    }
}

} // namespace euler::dda