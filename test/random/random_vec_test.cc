#include <euler/random/random_vec.hh>
#include <euler/random/random.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>
#include <algorithm>

using vec2f = euler::vec2<float>;
using vec2d = euler::vec2<double>;
using vec3f = euler::vec3<float>;
using vec3d = euler::vec3<double>;

using namespace euler;

TEST_CASE("Random Vector - Uniform Components") {
    random_generator rng(12345);
    
    SUBCASE("Default range [0, 1]") {
        const int N = 10000;
        std::vector<float> all_components;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_vector<float, 3>(rng);
            
            for (size_t j = 0; j < 3; ++j) {
                CHECK(v[j] >= 0.0f);
                CHECK(v[j] <= 1.0f);
                all_components.push_back(v[j]);
            }
        }
        
        // Check uniform distribution
        float mean = 0.0f;
        for (float x : all_components) mean += x;
        mean /= static_cast<float>(all_components.size());
        CHECK(std::abs(mean - 0.5f) < 0.01f);
        
        // Check variance (should be 1/12 for uniform [0,1])
        float var = 0.0f;
        for (float x : all_components) {
            var += (x - mean) * (x - mean);
        }
        var /= static_cast<float>(all_components.size());
        CHECK(std::abs(var - 1.0f/12.0f) < 0.005f);
    }
    
    SUBCASE("Custom range") {
        const int N = 1000;
        double min_val = -2.0;
        double max_val = 3.0;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_vector<double, 4>(rng, min_val, max_val);
            
            for (size_t j = 0; j < 4; ++j) {
                CHECK(v[j] >= min_val);
                CHECK(v[j] <= max_val);
            }
        }
    }
    
    SUBCASE("Different dimensions") {
        auto v2 = random_vector<float, 2>(rng, -1.0f, 1.0f);
        CHECK(v2.size == 2);
        
        auto v5 = random_vector<double, 5>(rng, 0.0, 10.0);
        CHECK(v5.size == 5);
        
        auto v10 = random_vector<float, 10>(rng);
        CHECK(v10.size == 10);
    }
}

TEST_CASE("Random Unit Vector") {
    random_generator rng(54321);
    
    SUBCASE("2D unit vectors") {
        const int N = 10000;
        float sum_x = 0.0f, sum_y = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<float, 2>(rng);
            
            // Check unit length
            CHECK(std::abs(v.length() - 1.0f) < 1e-6f);
            
            sum_x += v[0];
            sum_y += v[1];
        }
        
        // Mean should be near zero
        CHECK(std::abs(sum_x / N) < 0.02f);
        CHECK(std::abs(sum_y / N) < 0.02f);
        
        // Check angle distribution
        std::vector<float> angles;
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<double, 2>(rng);
            float angle = static_cast<float>(std::atan2(v[1], v[0]));
            if (angle < 0) angle += 2.0f * constants<float>::pi;
            angles.push_back(angle);
        }
        
        // Should be uniform in [0, 2π]
        float mean_angle = 0.0f;
        for (float a : angles) mean_angle += a;
        mean_angle /= static_cast<float>(angles.size());
        CHECK(std::abs(mean_angle - constants<float>::pi) < 0.1f);
    }
    
    SUBCASE("3D unit vectors - Marsaglia method") {
        const int N = 10000;
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<float, 3>(rng);
            
            // Check unit length
            CHECK(std::abs(v.length() - 1.0f) < 1e-6f);
            
            sum_x += v[0];
            sum_y += v[1];
            sum_z += v[2];
        }
        
        // Mean should be near zero
        CHECK(std::abs(sum_x / N) < 0.02f);
        CHECK(std::abs(sum_y / N) < 0.02f);
        CHECK(std::abs(sum_z / N) < 0.02f);
        
        // Check that all octants are covered
        int octant_counts[8] = {0};
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<double, 3>(rng);
            
            int octant = 0;
            if (v[0] > 0) octant |= 1;
            if (v[1] > 0) octant |= 2;
            if (v[2] > 0) octant |= 4;
            
            octant_counts[octant]++;
        }
        
        // Each octant should have roughly N/8 points
        for (int i = 0; i < 8; ++i) {
            CHECK(std::abs(octant_counts[i] - N/8) < N/20);
        }
    }
    
    SUBCASE("Higher dimensions - Gaussian method") {
        const int N = 1000;
        
        // Test 4D
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<float, 4>(rng);
            CHECK(std::abs(v.length() - 1.0f) < 1e-6f);
        }
        
        // Test 10D
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<double, 10>(rng);
            CHECK(std::abs(v.length() - 1.0) < 1e-6);
        }
    }
}

TEST_CASE("Random Vector - In Sphere") {
    random_generator rng(99999);
    
    SUBCASE("2D disk") {
        const int N = 10000;
        int inside_half = 0;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_in_sphere<float, 2>(rng);
            float len = v.length();
            CHECK(len <= 1.0f);
            
            if (len <= 0.5f) {
                inside_half++;
            }
        }
        
        // Area ratio: π(0.5)²/π(1)² = 0.25
        float ratio = float(inside_half) / N;
        CHECK(std::abs(ratio - 0.25f) < 0.02f);
    }
    
    SUBCASE("3D ball") {
        const int N = 10000;
        int inside_half = 0;
        float sum_radius = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_in_sphere<double, 3>(rng);
            double len = v.length();
            CHECK(len <= 1.0);
            
            if (len <= 0.5) {
                inside_half++;
            }
            sum_radius += static_cast<float>(len);
        }
        
        // Volume ratio: (4/3)π(0.5)³/(4/3)π(1)³ = 0.125
        float ratio = float(inside_half) / N;
        CHECK(std::abs(ratio - 0.125f) < 0.02f);
        
        // Mean radius for uniform distribution in 3D ball is 3/4
        float mean_radius = sum_radius / N;
        CHECK(std::abs(mean_radius - 0.75f) < 0.02f);
    }
    
    SUBCASE("Higher dimensions") {
        // In N dimensions, if uniformly distributed in unit ball,
        // the mean radius is N/(N+1)
        
        // 4D
        const int N4 = 1000;
        float sum_r4 = 0.0f;
        for (int i = 0; i < N4; ++i) {
            auto v = random_in_sphere<float, 4>(rng);
            sum_r4 += v.length();
        }
        float mean_r4 = sum_r4 / N4;
        CHECK(std::abs(mean_r4 - 4.0f/5.0f) < 0.02f);
        
        // 5D
        const int N5 = 1000;
        float sum_r5 = 0.0f;
        for (int i = 0; i < N5; ++i) {
            auto v = random_in_sphere<double, 5>(rng);
            sum_r5 += static_cast<float>(v.length());
        }
        float mean_r5 = sum_r5 / static_cast<float>(N5);
        CHECK(std::abs(mean_r5 - 5.0f/6.0f) < 0.02f);
    }
}

TEST_CASE("Random Vector - On Sphere") {
    random_generator rng(11111);
    
    SUBCASE("Custom radius") {
        float radius = 2.5f;
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_on_sphere<float, 3>(rng, radius);
            CHECK(std::abs(v.length() - radius) < 1e-6f);
        }
    }
    
    SUBCASE("Distribution check") {
        const int N = 10000;
        
        // Generate points on unit 4-sphere
        std::vector<vector<double, 4>> points;
        for (int i = 0; i < N; ++i) {
            points.push_back(random_on_sphere<double, 4>(rng));
        }
        
        // Check that mean is near zero
        vector<double, 4> mean = vector<double, 4>::zero();
        for (const auto& p : points) {
            mean = mean + p;
        }
        mean = mean / double(N);
        
        CHECK(mean.length() < 0.02);
    }
}

TEST_CASE("Random Vector - Normal Distribution") {
    random_generator rng(22222);
    
    SUBCASE("Zero mean, unit variance") {
        const int N = 10000;
        std::vector<vector<float, 3>> samples;
        
        for (int i = 0; i < N; ++i) {
            samples.push_back(random_vector_normal<float, 3>(rng));
        }
        
        // Check each component
        for (size_t comp = 0; comp < 3; ++comp) {
            float mean = 0.0f;
            float var = 0.0f;
            
            for (const auto& v : samples) {
                mean += v[comp];
            }
            mean /= N;
            
            for (const auto& v : samples) {
                var += (v[comp] - mean) * (v[comp] - mean);
            }
            var /= N;
            
            CHECK(std::abs(mean) < 0.02f);
            CHECK(std::abs(var - 1.0f) < 0.05f);
        }
    }
    
    SUBCASE("Custom mean and stddev") {
        vector<double, 2> mean(10.0, -5.0);
        double stddev = 2.0;
        
        const int N = 10000;
        vector<double, 2> sum = vector<double, 2>::zero();
        
        for (int i = 0; i < N; ++i) {
            auto v = random_vector_normal<double, 2>(rng, mean, stddev);
            sum = sum + v;
        }
        
        vector<double, 2> computed_mean = sum / double(N);
        CHECK(std::abs(computed_mean[0] - mean[0]) < 0.05);
        CHECK(std::abs(computed_mean[1] - mean[1]) < 0.05);
    }
}

TEST_CASE("Random Vector - In Box") {
    random_generator rng(33333);
    
    SUBCASE("Box constraints") {
        vector<float, 3> min_corner(-1.0f, -2.0f, -3.0f);
        vector<float, 3> max_corner(1.0f, 2.0f, 3.0f);
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_in_box<float, 3>(rng, min_corner, max_corner);
            
            for (size_t j = 0; j < 3; ++j) {
                CHECK(v[j] >= min_corner[j]);
                CHECK(v[j] <= max_corner[j]);
            }
        }
    }
    
    SUBCASE("Uniform distribution") {
        vector<double, 2> min_corner(0.0, 0.0);
        vector<double, 2> max_corner(10.0, 20.0);
        
        const int N = 10000;
        double sum_x = 0.0, sum_y = 0.0;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_in_box<double, 2>(rng, min_corner, max_corner);
            sum_x += v[0];
            sum_y += v[1];
        }
        
        CHECK(std::abs(sum_x / N - 5.0) < 0.1);
        CHECK(std::abs(sum_y / N - 10.0) < 0.2);
    }
}

TEST_CASE("Random Vector - On Simplex") {
    random_generator rng(44444);
    
    SUBCASE("Sum to one") {
        // Test different dimensions
        for (size_t dim : {2u, 3u, 4u, 5u, 10u}) {
            for (int i = 0; i < 100; ++i) {
                if (dim == 2) {
                    auto v = random_on_simplex<float, 2>(rng);
                    CHECK(std::abs(v[0] + v[1] - 1.0f) < 1e-5f);
                    CHECK(v[0] >= 0.0f);
                    CHECK(v[1] >= 0.0f);
                } else if (dim == 3) {
                    auto v = random_on_simplex<double, 3>(rng);
                    double sum = v[0] + v[1] + v[2];
                    CHECK(std::abs(sum - 1.0) < 1e-10);
                    for (size_t j = 0; j < 3; ++j) {
                        CHECK(v[j] >= 0.0);
                    }
                }
            }
        }
    }
    
    SUBCASE("Uniform distribution") {
        const int N = 10000;
        std::vector<float> first_components;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_on_simplex<float, 3>(rng);
            first_components.push_back(v[0]);
        }
        
        // For uniform distribution on 2-simplex (triangle),
        // each component has mean 1/3
        float mean = 0.0f;
        for (float x : first_components) mean += x;
        mean /= N;
        
        CHECK(std::abs(mean - 1.0f/3.0f) < 0.01f);
    }
}

TEST_CASE("Random Matrix - Uniform Elements") {
    random_generator rng(55555);
    
    SUBCASE("Default range") {
        auto m = random_matrix<float, 3, 4>(rng);
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                CHECK(m(i, j) >= 0.0f);
                CHECK(m(i, j) <= 1.0f);
            }
        }
    }
    
    SUBCASE("Custom range") {
        const int N = 100;
        double min_val = -5.0;
        double max_val = 10.0;
        
        double sum = 0.0;
        int count = 0;
        
        for (int n = 0; n < N; ++n) {
            auto m = random_matrix<double, 2, 2>(rng, min_val, max_val);
            
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    CHECK(m(i, j) >= min_val);
                    CHECK(m(i, j) <= max_val);
                    sum += m(i, j);
                    count++;
                }
            }
        }
        
        double mean = sum / static_cast<double>(count);
        CHECK(std::abs(mean - (min_val + max_val)/2.0) < 0.5);
    }
}

TEST_CASE("Random Matrix - Rotation Matrices") {
    random_generator rng(66666);
    
    SUBCASE("2D rotation") {
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto R = random_rotation_matrix<float, 2>(rng);
            
            // Check orthogonality: R * R^T = I
            // Force evaluation to avoid expression template issues
            matrix<float, 2, 2> RRt = R * transpose(R);
            CHECK(approx_equal(RRt, matrix<float, 2, 2>::identity(), 1e-6f));
            
            // Check determinant = 1
            CHECK(std::abs(determinant(R) - 1.0f) < 1e-6f);
            
            // Check that columns are unit vectors
            vec2f col0(R(0,0), R(1,0));
            vec2f col1(R(0,1), R(1,1));
            CHECK(std::abs(col0.length() - 1.0f) < 1e-6f);
            CHECK(std::abs(col1.length() - 1.0f) < 1e-6f);
            
            // Check that columns are orthogonal
            CHECK(std::abs(dot(col0, col1)) < 1e-6f);
        }
    }
    
    SUBCASE("3D rotation") {
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto R = random_rotation_matrix<double, 3>(rng);
            
            // Check orthogonality
            // Force evaluation to avoid expression template issues
            matrix<double, 3, 3> RRt = R * transpose(R);
            CHECK(approx_equal(RRt, matrix<double, 3, 3>::identity(), 1e-10));
            
            // Check determinant = 1
            CHECK(std::abs(determinant(R) - 1.0) < 1e-10);
            
            // Check preservation of vector length
            vec3d v(1.0, 2.0, 3.0);
            vec3d Rv = R * v;
            CHECK(std::abs(Rv.length() - v.length()) < 1e-10);
        }
    }
}

TEST_CASE("Random Matrix - Orthogonal Matrices") {
    random_generator rng(77777);
    
    SUBCASE("Determinant distribution") {
        const int N = 1000;
        int pos_det = 0;
        int neg_det = 0;
        
        for (int i = 0; i < N; ++i) {
            auto O = random_orthogonal_matrix<float, 3>(rng);
            
            // Check orthogonality
            // Force evaluation to avoid expression template issues
            matrix<float, 3, 3> OOt = O * transpose(O);
            CHECK(approx_equal(OOt, matrix<float, 3, 3>::identity(), 1e-6f));
            
            float det = determinant(O);
            CHECK(std::abs(std::abs(det) - 1.0f) < 1e-6f);
            
            if (det > 0) pos_det++;
            else neg_det++;
        }
        
        // Should be roughly 50/50
        CHECK(std::abs(pos_det - neg_det) < N/10);
    }
}

TEST_CASE("Random Matrix - Symmetric Matrices") {
    random_generator rng(88888);
    
    SUBCASE("Symmetry check") {
        const int N = 100;
        
        for (int i = 0; i < N; ++i) {
            auto S = random_symmetric_matrix<double, 4>(rng, -2.0, 2.0);
            
            // Check symmetry
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    CHECK(S(r, c) == S(c, r));
                    CHECK(S(r, c) >= -2.0);
                    CHECK(S(r, c) <= 2.0);
                }
            }
            
            // Check that S = S^T
            CHECK(approx_equal(S, transpose(S), 1e-15));
        }
    }
}

TEST_CASE("Random Matrix - Positive Definite") {
    random_generator rng(99999);
    
    SUBCASE("Eigenvalue properties") {
        float condition_number = 5.0f;
        auto A = random_positive_definite_matrix<float, 3>(rng, condition_number);
        
        // Check symmetry
        CHECK(approx_equal(A, transpose(A), 1e-6f));
        
        // Check positive definiteness by testing with random vectors
        for (int i = 0; i < 100; ++i) {
            auto v = random_vector<float, 3>(rng, -1.0f, 1.0f);
            if (v.length() > 1e-6f) {
                float vTAv = dot(v, A * v);
                CHECK(vTAv > 0.0f);
            }
        }
    }
    
    SUBCASE("Different condition numbers") {
        // Well-conditioned
        auto A1 = random_positive_definite_matrix<double, 2>(rng, 2.0);
        
        // Ill-conditioned
        auto A2 = random_positive_definite_matrix<double, 2>(rng, 100.0);
        
        // Both should be positive definite
        vec2d v(1.0, 1.0);
        CHECK(dot(v, A1 * v) > 0.0);
        CHECK(dot(v, A2 * v) > 0.0);
    }
}

TEST_CASE("Random Matrix - Thread-local Functions") {
    SUBCASE("All convenience functions") {
        // Vectors
        auto v1 = random_vector<float, 3>(-1.0f, 1.0f);
        CHECK(v1.size == 3);
        
        auto v2 = random_unit_vector<double, 4>();
        CHECK(std::abs(v2.length() - 1.0) < 1e-10);
        
        auto v3 = random_in_sphere<float, 2>();
        CHECK(v3.length() <= 1.0f);
        
        // Matrices
        auto m1 = random_matrix<double, 3, 3>(0.0, 1.0);
        CHECK(m1.rows == 3);
        CHECK(m1.cols == 3);
        
        auto m2 = random_rotation_matrix<float, 2>();
        CHECK(std::abs(determinant(m2) - 1.0f) < 1e-6f);
    }
}