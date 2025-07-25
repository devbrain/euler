/**
 * @file dda.hh
 * @brief Main header for DDA (Digital Differential Analyzer) module
 * @ingroup DDAModule
 * 
 * The DDA module provides high-performance rasterization algorithms for
 * geometric primitives and curves. All algorithms are implemented as
 * iterators for efficient, cache-friendly pixel generation.
 * 
 * @section Features
 * - Iterator-based design for lazy evaluation
 * - Sub-pixel precision with antialiasing support
 * - Transparent SIMD acceleration (no API changes)
 * - Pixel batching for improved cache utilization
 * - Expression template integration
 * - Comprehensive 2D rasterization support
 * 
 * @section Performance
 * The DDA module is optimized for performance:
 * - Automatic SIMD vectorization when available (4-8x speedup)
 * - Pixel batching reduces function call overhead
 * - Prefetch hints for predictable access patterns
 * - Adaptive stepping balances quality and speed
 * 
 * @note SIMD optimization is completely transparent. All iterators
 *       automatically use SIMD instructions when available without
 *       any API changes or special iterator types.
 * 
 * @section Coordinates
 * The DDA module supports multiple coordinate systems:
 * 
 * @subsection screen_coords Screen Coordinates
 * - Integer pixel coordinates (0,0) at top-left
 * - Y-axis points downward (standard raster convention)
 * - Sub-pixel precision via floating-point coordinates
 * - Automatic rounding for pixel generation
 * 
 * @subsection world_coords World Coordinates
 * - Floating-point coordinates for curves and primitives
 * - Supports arbitrary coordinate ranges
 * - Automatic conversion to screen space during rasterization
 * 
 * @subsection projective_coords Projective Coordinates
 * - Homogeneous coordinates supported via point3<T>
 * - Automatic perspective division when needed
 * - Useful for perspective-correct interpolation
 * 
 * @subsection clipping Clipping Support
 * - Rectangle-based clipping for all iterators
 * - Cohen-Sutherland algorithm for line clipping
 * - Sutherland-Hodgman for polygon clipping
 * - Automatic handling of partially visible primitives
 * 
 * @code
 * // Example: Clipping a line to screen bounds
 * rectangle<int> screen{{0, 0}, {1920, 1080}};
 * point2f p1{-100, 500}, p2{2000, 600};
 * 
 * if (screen.clip_line(p1, p2)) {
 *     // Line is at least partially visible
 *     for (auto pixel : line_pixels(p1, p2)) {
 *         draw_pixel(pixel.pos);
 *     }
 * }
 * 
 * // Example: Projective coordinates
 * point3f proj_p1{100, 200, 2};  // w = 2
 * point3f proj_p2{300, 400, 4};  // w = 4
 * 
 * // Convert to screen coordinates
 * point2f screen_p1{proj_p1.x / proj_p1.z, proj_p1.y / proj_p1.z};
 * point2f screen_p2{proj_p2.x / proj_p2.z, proj_p2.y / proj_p2.z};
 * @endcode
 * 
 * @author Euler Development Team
 * @date 2024
 */
#pragma once

// Core traits and types
#include <euler/dda/dda_traits.hh>

// Line iterators
#include <euler/dda/line_iterator.hh>
#include <euler/dda/aa_line_iterator.hh>
#include <euler/dda/thick_line_iterator.hh>

// Circle iterators
#include <euler/dda/circle_iterator.hh>

// Ellipse iterators
#include <euler/dda/ellipse_iterator.hh>

// Arc iterators (filled and antialiased arcs)
#include <euler/dda/arc_iterator.hh>

// Curve iterators
#include <euler/dda/curve_iterator.hh>
#include <euler/dda/aa_curve_iterator.hh>
#include <euler/dda/bezier_iterator.hh>
#include <euler/dda/bspline_iterator.hh>
#include <euler/dda/aa_bspline_iterator.hh>

// Batched iterators for performance
#include <euler/dda/pixel_batch.hh>
#include <euler/dda/batched_line_iterator.hh>
#include <euler/dda/batched_bezier_iterator.hh>

// Mathematical curves library
#include <euler/dda/mathematical_curves.hh>

/**
 * @defgroup DDAModule DDA Module
 * @brief Digital Differential Analyzer for rasterization
 * 
 * The DDA module provides:
 * - Line rasterization (basic, antialiased, thick)
 * - Circle and ellipse rasterization
 * - Bezier and B-spline curves
 * - Generic curve rasterization with adaptive stepping
 * - Mathematical curves library
 * 
 * @section dda_usage Usage Examples
 * 
 * @subsection dda_lines Lines
 * @code
 * // Basic line
 * for (auto pixel : euler::dda::line_pixels(point2{0, 0}, point2{100, 50})) {
 *     draw_pixel(pixel.pos.x, pixel.pos.y);
 * }
 * 
 * // Antialiased line
 * auto line = euler::dda::make_aa_line_iterator(point2{0.0f, 0.0f}, point2{100.0f, 50.0f});
 * for (; line != euler::dda::aa_line_iterator<float>::end(); ++line) {
 *     auto pixel = *line;
 *     draw_pixel(pixel.pos.x, pixel.pos.y, pixel.coverage);
 * }
 * 
 * // Thick line
 * for (auto pixel : euler::dda::make_thick_line_iterator(start, end, 5.0f)) {
 *     draw_pixel(pixel.pos.x, pixel.pos.y);
 * }
 * @endcode
 * 
 * @subsection dda_circles Circles and Ellipses
 * @code
 * // Circle
 * for (auto pixel : euler::dda::circle_pixels(center, radius)) {
 *     draw_pixel(pixel.pos.x, pixel.pos.y);
 * }
 * 
 * // Filled circle using spans
 * auto filled = euler::dda::make_filled_circle_iterator(center, radius);
 * for (; filled != euler::dda::filled_circle_iterator<float>::end(); ++filled) {
 *     auto span = *filled;
 *     draw_horizontal_line(span.y, span.x_start, span.x_end);
 * }
 * 
 * // Ellipse
 * auto ellipse = euler::dda::make_ellipse_iterator(center, 50.0f, 30.0f);
 * for (; ellipse != euler::dda::ellipse_iterator<float>::end(); ++ellipse) {
 *     draw_pixel((*ellipse).pos.x, (*ellipse).pos.y);
 * }
 * @endcode
 * 
 * @subsection dda_curves Curves
 * @code
 * // Parametric curve
 * auto spiral = [](float t) { 
 *     float r = t * 10;
 *     return point2{r * cos(t), r * sin(t)}; 
 * };
 * auto curve = euler::dda::make_curve_iterator(spiral, 0.0f, 6.28f);
 * 
 * // Cartesian curve (y = f(x))
 * auto parabola = [](float x) { return x * x / 100; };
 * auto cart = euler::dda::curve_iterator<float>::cartesian(parabola, -50.0f, 50.0f);
 * 
 * // Polar curve (r = f(theta))
 * auto rose = euler::dda::curves::rose(50.0f, 3.0f);
 * auto polar = euler::dda::curve_iterator<float>::polar(rose, 0.0f, 2*pi);
 * 
 * // Mathematical curves
 * auto cardioid = euler::dda::curves::cardioid(30.0f);
 * auto witch = euler::dda::curves::agnesi_witch(20.0f);
 * @endcode
 * 
 * @subsection dda_bezier Bezier and B-Splines
 * @code
 * // Quadratic Bezier
 * auto quad = euler::dda::make_quadratic_bezier(p0, p1, p2);
 * 
 * // Cubic Bezier
 * auto cubic = euler::dda::make_cubic_bezier(p0, p1, p2, p3);
 * 
 * // B-spline
 * std::vector<point2<float>> control_points = {...};
 * auto spline = euler::dda::make_bspline(control_points, 3); // degree 3
 * 
 * // Catmull-Rom (interpolating spline)
 * auto catmull = euler::dda::make_catmull_rom(points);
 * 
 * // Antialiased variants
 * auto aa_curve = euler::dda::make_aa_curve_iterator(spiral, 0.0f, 6.28f);
 * auto aa_spline = euler::dda::make_aa_bspline(control_points, 3);
 * @endcode
 */

namespace euler::dda {

/**
 * @brief Quick reference for common operations
 */

// Lines
using euler::dda::make_line_iterator;
using euler::dda::make_aa_line_iterator;
using euler::dda::make_thick_line_iterator;
using euler::dda::line_pixels;

// Circles
using euler::dda::make_circle_iterator;
using euler::dda::make_arc_iterator;
using euler::dda::make_filled_circle_iterator;
using euler::dda::make_filled_arc_iterator;
using euler::dda::make_aa_circle_iterator;
using euler::dda::make_aa_arc_iterator;
using euler::dda::circle_pixels;

// Ellipses
using euler::dda::make_ellipse_iterator;
using euler::dda::make_ellipse_arc_iterator;
using euler::dda::make_filled_ellipse_iterator;
using euler::dda::make_filled_ellipse_arc_iterator;
using euler::dda::make_aa_ellipse_iterator;
using euler::dda::make_aa_ellipse_arc_iterator;

// Curves
using euler::dda::make_curve_iterator;
using euler::dda::make_cartesian_curve;
using euler::dda::make_polar_curve;
using euler::dda::make_aa_curve_iterator;
using euler::dda::make_aa_cartesian_curve;
using euler::dda::make_aa_polar_curve;

// Bezier
using euler::dda::make_quadratic_bezier;
using euler::dda::make_cubic_bezier;
using euler::dda::make_bezier;
using euler::dda::make_aa_cubic_bezier;

// B-splines
using euler::dda::make_bspline;
using euler::dda::make_catmull_rom;
using euler::dda::make_aa_bspline;
using euler::dda::make_aa_catmull_rom;

} // namespace euler::dda