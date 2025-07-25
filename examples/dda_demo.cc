#include <euler/euler.hh>
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <iomanip>

using namespace euler;
using namespace euler::dda;

// Simple ASCII canvas for visualization
class Canvas {
    static constexpr int WIDTH = 120;
    static constexpr int HEIGHT = 40;
    std::array<std::array<char, WIDTH>, HEIGHT> buffer;
    
public:
    Canvas() { clear(); }
    
    void clear() {
        for (auto& row : buffer) {
            std::fill(row.begin(), row.end(), ' ');
        }
    }
    
    void set_pixel(int x, int y, char c = '*') {
        // Center the coordinate system
        x += WIDTH / 2;
        y = HEIGHT / 2 - y;
        
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            buffer[y][x] = c;
        }
    }
    
    void set_pixel(point2i p, char c = '*') {
        set_pixel(p.x, p.y, c);
    }
    
    void draw_horizontal_line(int y, int x_start, int x_end, char c = '=') {
        for (int x = x_start; x <= x_end; ++x) {
            set_pixel(x, y, c);
        }
    }
    
    void print() const {
        // Top border
        std::cout << "+";
        for (int i = 0; i < WIDTH; ++i) std::cout << "-";
        std::cout << "+\n";
        
        // Content
        for (const auto& row : buffer) {
            std::cout << "|";
            for (char c : row) std::cout << c;
            std::cout << "|\n";
        }
        
        // Bottom border
        std::cout << "+";
        for (int i = 0; i < WIDTH; ++i) std::cout << "-";
        std::cout << "+\n";
    }
};

void demo_lines() {
    std::cout << "\n=== Line Rasterization Demo ===\n";
    Canvas canvas;
    
    // Draw coordinate axes
    for (auto pixel : line_pixels(point2i{-50, 0}, point2i{50, 0})) {
        canvas.set_pixel(pixel.pos, '-');
    }
    for (auto pixel : line_pixels(point2i{0, -18}, point2i{0, 18})) {
        canvas.set_pixel(pixel.pos, '|');
    }
    canvas.set_pixel(0, 0, '+');
    
    // Draw various lines
    for (auto pixel : line_pixels(point2i{-40, -15}, point2i{40, 15})) {
        canvas.set_pixel(pixel.pos, '/');
    }
    
    for (auto pixel : line_pixels(point2i{-40, 15}, point2i{40, -15})) {
        canvas.set_pixel(pixel.pos, '\\');
    }
    
    // Thick line
    auto thick = make_thick_line_iterator(point2{-30.0f, 10.0f}, point2{30.0f, 10.0f}, 3.0f);
    for (; thick != decltype(thick)::end(); ++thick) {
        canvas.set_pixel((*thick).pos, '#');
    }
    
    canvas.print();
    std::cout << "Legend: - horizontal, | vertical, / diagonal, \\ diagonal, # thick line\n";
}

void demo_circles() {
    std::cout << "\n=== Circle Rasterization Demo ===\n";
    Canvas canvas;
    
    // Concentric circles
    for (auto pixel : circle_pixels(point2i{0, 0}, 5)) {
        canvas.set_pixel(pixel.pos, '.');
    }
    
    for (auto pixel : circle_pixels(point2i{0, 0}, 10)) {
        canvas.set_pixel(pixel.pos, 'o');
    }
    
    for (auto pixel : circle_pixels(point2i{0, 0}, 15)) {
        canvas.set_pixel(pixel.pos, 'O');
    }
    
    // Arc
    auto arc = make_arc_iterator(point2{25.0f, 0.0f}, 8.0f, 
                                degree<float>(30), degree<float>(150));
    for (; arc != decltype(arc)::end(); ++arc) {
        canvas.set_pixel((*arc).pos, '^');
    }
    
    // Filled circle
    auto filled = make_filled_circle_iterator(point2{-25.0f, 0.0f}, 6.0f);
    for (; filled != decltype(filled)::end(); ++filled) {
        auto span = *filled;
        canvas.draw_horizontal_line(span.y, span.x_start, span.x_end, '*');
    }
    
    canvas.print();
    std::cout << "Legend: . small circle, o medium circle, O large circle\n";
    std::cout << "        ^ arc, * filled circle\n";
}

void demo_ellipses() {
    std::cout << "\n=== Ellipse Rasterization Demo ===\n";
    Canvas canvas;
    
    // Horizontal ellipse
    auto h_ellipse = make_ellipse_iterator(point2{0.0f, 0.0f}, 20.0f, 10.0f);
    for (; h_ellipse != decltype(h_ellipse)::end(); ++h_ellipse) {
        canvas.set_pixel((*h_ellipse).pos, '-');
    }
    
    // Vertical ellipse
    auto v_ellipse = make_ellipse_iterator(point2{30.0f, 0.0f}, 8.0f, 15.0f);
    for (; v_ellipse != decltype(v_ellipse)::end(); ++v_ellipse) {
        canvas.set_pixel((*v_ellipse).pos, '|');
    }
    
    // Filled ellipse
    auto filled = make_filled_ellipse_iterator(point2{-30.0f, 0.0f}, 10.0f, 6.0f);
    for (; filled != decltype(filled)::end(); ++filled) {
        auto span = *filled;
        canvas.draw_horizontal_line(span.y, span.x_start, span.x_end, '#');
    }
    
    canvas.print();
    std::cout << "Legend: - horizontal ellipse, | vertical ellipse, # filled ellipse\n";
}

void demo_curves() {
    std::cout << "\n=== Mathematical Curves Demo ===\n";
    Canvas canvas;
    
    // Sine wave
    auto sine = curves::sine_wave(8.0f, 0.15f);
    auto sine_curve = curve_iterator<float, decltype(sine)>::cartesian(
        sine, -50.0f, 50.0f);
    for (; sine_curve != decltype(sine_curve)::end(); ++sine_curve) {
        canvas.set_pixel((*sine_curve).pos, '~');
    }
    
    // Parabola
    auto parabola = [](float x) { return 0.02f * x * x - 10.0f; };
    auto para_curve = curve_iterator<float, decltype(parabola)>::cartesian(
        parabola, -30.0f, 30.0f);
    for (; para_curve != decltype(para_curve)::end(); ++para_curve) {
        canvas.set_pixel((*para_curve).pos, 'U');
    }
    
    // Rose curve (polar)
    auto rose = curves::rose(12.0f, 5.0f);
    auto rose_curve = curve_iterator<float, decltype(rose)>::polar(
        rose, 0.0f, 2.0f * constants<float>::pi, point2{0.0f, 0.0f});
    for (; rose_curve != decltype(rose_curve)::end(); ++rose_curve) {
        canvas.set_pixel((*rose_curve).pos, '*');
    }
    
    canvas.print();
    std::cout << "Legend: ~ sine wave, U parabola, * rose curve (5 petals)\n";
}

void demo_bezier() {
    std::cout << "\n=== Bezier Curves Demo ===\n";
    Canvas canvas;
    
    // Quadratic Bezier
    point2f q0{-40, -10}, q1{-20, 15}, q2{0, -10};
    auto quad = make_quadratic_bezier(q0, q1, q2);
    for (; quad != decltype(quad)::end(); ++quad) {
        canvas.set_pixel((*quad).pos, '2');
    }
    
    // Cubic Bezier S-curve
    point2f c0{10, -15}, c1{30, -15}, c2{10, 15}, c3{30, 15};
    auto cubic = make_cubic_bezier(c0, c1, c2, c3);
    for (; cubic != decltype(cubic)::end(); ++cubic) {
        canvas.set_pixel((*cubic).pos, '3');
    }
    
    // Mark control points
    canvas.set_pixel(round(q0), 'o');
    canvas.set_pixel(round(q1), '+');
    canvas.set_pixel(round(q2), 'o');
    
    canvas.set_pixel(round(c0), 'o');
    canvas.set_pixel(round(c1), '+');
    canvas.set_pixel(round(c2), '+');
    canvas.set_pixel(round(c3), 'o');
    
    canvas.print();
    std::cout << "Legend: 2 quadratic Bezier, 3 cubic Bezier\n";
    std::cout << "        o endpoints, + control points\n";
}

void demo_splines() {
    std::cout << "\n=== B-Splines Demo ===\n";
    Canvas canvas;
    
    // B-spline
    std::vector<point2f> control_points = {
        {-40, 0}, {-30, 10}, {-20, -5}, {-10, 15}, {0, 0},
        {10, -10}, {20, 5}, {30, -15}, {40, 0}
    };
    
    auto spline = make_bspline(control_points, 3);
    for (; spline != decltype(spline)::end(); ++spline) {
        canvas.set_pixel((*spline).pos, 'B');
    }
    
    // Catmull-Rom (interpolating)
    std::vector<point2f> interp_points = {
        {-35, -10}, {-20, 5}, {-5, -10}, {10, 5}, {25, -10}
    };
    
    auto catmull = make_catmull_rom(interp_points);
    for (; catmull != decltype(catmull)::end(); ++catmull) {
        canvas.set_pixel((*catmull).pos, 'C');
    }
    
    // Mark control/interpolation points
    for (const auto& p : control_points) {
        canvas.set_pixel(round(p), '.');
    }
    for (const auto& p : interp_points) {
        canvas.set_pixel(round(p), 'o');
    }
    
    canvas.print();
    std::cout << "Legend: B B-spline, C Catmull-Rom spline\n";
    std::cout << "        . B-spline control points, o Catmull-Rom interpolation points\n";
}

void demo_antialiasing() {
    std::cout << "\n=== Antialiasing Demo ===\n";
    std::cout << "(Coverage values shown as ASCII intensity)\n\n";
    
    // Create a small canvas to show AA pixels
    constexpr int SIZE = 20;
    std::array<std::array<float, SIZE>, SIZE> coverage{};
    
    // Draw antialiased line
    auto aa_line = make_aa_line_iterator(
        point2{2.0f, 2.0f}, point2{17.0f, 12.0f});
    
    for (; aa_line != decltype(aa_line)::end(); ++aa_line) {
        auto pixel = *aa_line;
        if (pixel.pos.x >= 0 && pixel.pos.x < SIZE &&
            pixel.pos.y >= 0 && pixel.pos.y < SIZE) {
            coverage[pixel.pos.y][pixel.pos.x] = 
                std::max(coverage[pixel.pos.y][pixel.pos.x], pixel.coverage);
        }
    }
    
    // Print with intensity
    const char* intensity = " .:-=+*#%@";
    for (int y = SIZE - 1; y >= 0; --y) {
        for (int x = 0; x < SIZE; ++x) {
            float c = coverage[y][x];
            int idx = static_cast<int>(c * 9.99f);
            std::cout << intensity[idx] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nCoverage intensity: [space]=0% @ = 100%\n";
}

int main() {
    std::cout << "Euler DDA Module Demonstration\n";
    std::cout << "==============================\n";
    
    demo_lines();
    demo_circles();
    demo_ellipses();
    demo_curves();
    demo_bezier();
    demo_splines();
    demo_antialiasing();
    
    std::cout << "\nDemonstration complete!\n";
    return 0;
}