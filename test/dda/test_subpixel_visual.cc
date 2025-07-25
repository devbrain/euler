#include <euler/dda/dda.hh>
#include <euler/coordinates/point2.hh>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>

using namespace euler;
using namespace euler::dda;

// Simple grid to visualize pixels
class PixelGrid {
    std::array<std::array<float, 40>, 20> grid;
    
public:
    PixelGrid() {
        for (auto& row : grid) {
            row.fill(0.0f);
        }
    }
    
    void set_pixel(int x, int y, float value = 1.0f) {
        if (x >= 0 && x < 40 && y >= 0 && y < 20) {
            grid[static_cast<size_t>(y)][static_cast<size_t>(x)] = 
                std::min(1.0f, grid[static_cast<size_t>(y)][static_cast<size_t>(x)] + value);
        }
    }
    
    void print() const {
        // Print top border
        std::cout << "┌";
        for (int i = 0; i < 40; ++i) std::cout << "─";
        std::cout << "┐\n";
        
        // Print grid
        for (const auto& row : grid) {
            std::cout << "│";
            for (float val : row) {
                if (val == 0) {
                    std::cout << " ";
                } else if (val < 0.25f) {
                    std::cout << "░";
                } else if (val < 0.5f) {
                    std::cout << "▒";
                } else if (val < 0.75f) {
                    std::cout << "▓";
                } else {
                    std::cout << "█";
                }
            }
            std::cout << "│\n";
        }
        
        // Print bottom border
        std::cout << "└";
        for (int i = 0; i < 40; ++i) std::cout << "─";
        std::cout << "┘\n";
    }
};

int main() {
    std::cout << "Subpixel Accuracy Visual Comparison\n";
    std::cout << "===================================\n\n";
    
    // Test diagonal line with non-integer endpoints
    point2f start{5.7f, 3.3f};
    point2f end{35.2f, 16.8f};
    
    std::cout << "Drawing line from (" << start.x << ", " << start.y 
              << ") to (" << end.x << ", " << end.y << ")\n\n";
    
    // Integer rasterization
    {
        std::cout << "Integer rasterization (basic line_iterator):\n";
        PixelGrid grid;
        
        auto line = make_line_iterator(start, end);
        for (; line != decltype(line)::end(); ++line) {
            auto p = *line;
            grid.set_pixel(p.pos.x, p.pos.y);
        }
        
        grid.print();
        std::cout << "\n";
    }
    
    // Antialiased rasterization
    {
        std::cout << "Antialiased rasterization (aa_line_iterator):\n";
        PixelGrid grid;
        
        auto aa_line = make_aa_line_iterator(start, end);
        for (; aa_line != decltype(aa_line)::end(); ++aa_line) {
            auto p = *aa_line;
            grid.set_pixel(static_cast<int>(p.pos.x), 
                          static_cast<int>(p.pos.y), 
                          p.coverage);
        }
        
        grid.print();
        std::cout << "\n";
    }
    
    // Test circle with non-integer center and radius
    point2f center{20.5f, 10.5f};
    float radius = 7.3f;
    
    std::cout << "Drawing circle at (" << center.x << ", " << center.y 
              << ") with radius " << radius << "\n\n";
    
    // Integer circle
    {
        std::cout << "Integer circle (basic circle_iterator):\n";
        PixelGrid grid;
        
        auto circle = make_circle_iterator(center, radius);
        for (; circle != decltype(circle)::end(); ++circle) {
            auto p = *circle;
            grid.set_pixel(p.pos.x, p.pos.y);
        }
        
        grid.print();
        std::cout << "\n";
    }
    
    // Antialiased circle
    {
        std::cout << "Antialiased circle (aa_circle_iterator):\n";
        PixelGrid grid;
        
        auto aa_circle = make_aa_circle_iterator(center, radius);
        for (; aa_circle != decltype(aa_circle)::end(); ++aa_circle) {
            auto p = *aa_circle;
            grid.set_pixel(static_cast<int>(p.pos.x), 
                          static_cast<int>(p.pos.y), 
                          p.coverage);
        }
        
        grid.print();
        std::cout << "\n";
    }
    
    std::cout << "Legend: █ = full coverage, ▓ = 75%, ▒ = 50%, ░ = 25%, space = 0%\n";
    std::cout << "\nNote: Antialiased versions show smoother edges with partial coverage\n";
    
    return 0;
}