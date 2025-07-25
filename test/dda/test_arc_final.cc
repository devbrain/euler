#include <euler/dda/arc_iterator.hh>
#include <euler/angles/angle.hh>
#include <euler/coordinates/point2.hh>
#include <iostream>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::dda;

int main() {
    point2f center{50, 50};
    float radius = 20;
    
    std::cout << "Testing various arc configurations:\n\n";
    
    // Test 1: Quarter arc (0-90 degrees)
    {
        std::cout << "1. Quarter arc (0-90 degrees):\n";
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(0), degree<float>(90));
        int count = 0;
        for (; filled != decltype(filled)::end(); ++filled) {
            count++;
        }
        std::cout << "   Total spans: " << count << " (expected: ~21)\n\n";
    }
    
    // Test 2: Half arc (0-180 degrees)
    {
        std::cout << "2. Half arc (0-180 degrees):\n";
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(0), degree<float>(180));
        int count = 0;
        for (; filled != decltype(filled)::end(); ++filled) {
            count++;
        }
        std::cout << "   Total spans: " << count << " (expected: ~41)\n\n";
    }
    
    // Test 3: Arc crossing 0 degrees (270-90)
    {
        std::cout << "3. Arc crossing 0 degrees (270-90):\n";
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(270), degree<float>(90));
        int count = 0;
        int right_side_count = 0;
        for (; filled != decltype(filled)::end(); ++filled) {
            auto span = *filled;
            count++;
            // Check if span is on the right side (x >= center.x)
            if (span.x_start >= center.x) {
                right_side_count++;
            }
        }
        std::cout << "   Total spans: " << count << " (expected: ~41)\n";
        std::cout << "   Right-side spans: " << right_side_count << "\n\n";
    }
    
    // Test 4: Small arc (45-60 degrees)
    {
        std::cout << "4. Small arc (45-60 degrees):\n";
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(45), degree<float>(60));
        int count = 0;
        for (; filled != decltype(filled)::end(); ++filled) {
            count++;
        }
        std::cout << "   Total spans: " << count << "\n\n";
    }
    
    // Test 5: Ellipse arc
    {
        std::cout << "5. Ellipse arc (0-90 degrees):\n";
        float a = 30, b = 20;
        auto filled = make_filled_ellipse_arc_iterator(center, a, b, degree<float>(0), degree<float>(90));
        int count = 0;
        for (; filled != decltype(filled)::end(); ++filled) {
            count++;
        }
        std::cout << "   Total spans: " << count << " (expected: ~21)\n\n";
    }
    
    // Test 6: Check for vertical line bug
    {
        std::cout << "6. Checking for vertical line bug:\n";
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(0), degree<float>(90));
        bool found_x_zero = false;
        for (; filled != decltype(filled)::end(); ++filled) {
            auto span = *filled;
            if (span.x_start == 0 || span.x_end == 0) {
                std::cout << "   Found x=0 at y=" << span.y << "!\n";
                found_x_zero = true;
            }
        }
        if (!found_x_zero) {
            std::cout << "   No vertical line bug detected - SUCCESS!\n";
        }
    }
    
    return 0;
}