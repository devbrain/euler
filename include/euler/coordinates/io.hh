/**
 * @file io.hh
 * @brief I/O stream operators for coordinate types with pretty printing
 * @ingroup CoordinatesModule
 * 
 * This header provides stream output operators for all coordinate types
 * with support for the pretty printing infrastructure from io/io.hh.
 * The operators respect stream formatting flags like width and precision.
 * 
 * @section usage Usage
 * @code
 * point2f p{1.5f, 2.5f};
 * std::cout << p;                           // Output: (1.5, 2.5)
 * std::cout << std::setw(10) << p;         // Output: (      1.5,       2.5)
 * std::cout << std::fixed << std::setprecision(3) << p;  // Output: (1.500, 2.500)
 * 
 * projective3f proj{100, 200, 300, 2};
 * std::cout << proj;                        // Output: [100, 200, 300, 2]
 * @endcode
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/projective3.hh>
#include <euler/io/io.hh>
#include <ostream>
#include <iomanip>
#include <sstream>

namespace euler {

// Stream output for 2D points with pretty printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const point2<T>& p) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    os << '(';
    if (width > 0) {
        std::string x_str = detail::format_value(os, p.x);
        std::string y_str = detail::format_value(os, p.y);
        os << std::setw(width) << x_str << ", " << std::setw(width) << y_str;
    } else {
        os << p.x << ", " << p.y;
    }
    os << ')';
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

// Stream output for 3D points with pretty printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const point3<T>& p) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    os << '(';
    if (width > 0) {
        std::string x_str = detail::format_value(os, p.x);
        std::string y_str = detail::format_value(os, p.y);
        std::string z_str = detail::format_value(os, p.z);
        os << std::setw(width) << x_str << ", " 
           << std::setw(width) << y_str << ", " 
           << std::setw(width) << z_str;
    } else {
        os << p.x << ", " << p.y << ", " << p.z;
    }
    os << ')';
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

// Stream output for 2D projective coordinates with pretty printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const projective2<T>& p) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    os << '[';
    if (width > 0) {
        std::string x_str = detail::format_value(os, p.x);
        std::string y_str = detail::format_value(os, p.y);
        std::string w_str = detail::format_value(os, p.w);
        os << std::setw(width) << x_str << ", " 
           << std::setw(width) << y_str << ", " 
           << std::setw(width) << w_str;
    } else {
        os << p.x << ", " << p.y << ", " << p.w;
    }
    os << ']';
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

// Stream output for 3D projective coordinates with pretty printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const projective3<T>& p) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    os << '[';
    if (width > 0) {
        std::string x_str = detail::format_value(os, p.x);
        std::string y_str = detail::format_value(os, p.y);
        std::string z_str = detail::format_value(os, p.z);
        std::string w_str = detail::format_value(os, p.w);
        os << std::setw(width) << x_str << ", " 
           << std::setw(width) << y_str << ", " 
           << std::setw(width) << z_str << ", "
           << std::setw(width) << w_str;
    } else {
        os << p.x << ", " << p.y << ", " << p.z << ", " << p.w;
    }
    os << ']';
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

} // namespace euler