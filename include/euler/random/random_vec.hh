#pragma once

#include <euler/core/types.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/random/random.hh>
#include <euler/random/random_angle.hh>
#include <euler/math/trigonometry.hh>
#include <euler/math/basic.hh>
#include <euler/matrix/matrix_ops.hh>
#include <algorithm>

namespace euler {

// Forward declarations to avoid circular dependency
template<typename T> class quaternion;
template<typename T, typename Generator>
quaternion<T> random_quaternion(Generator& g);

// ============================================================================
// Random Vector Generation
// ============================================================================

// Random vector with uniform distribution in each component
template<typename T, size_t N, typename Generator>
vector<T, N> random_vector(Generator& g, T min = T(0), T max = T(1)) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = g.uniform(min, max);
    }
    return result;
}

// Random unit vector (uniform on sphere)
template<typename T, size_t N, typename Generator>
vector<T, N> random_unit_vector(Generator& g) {
    vector<T, N> v;
    
    if constexpr (N == 2) {
        // 2D: Simple angle method
        angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
        v[0] = cos(theta);
        v[1] = sin(theta);
    } else if constexpr (N == 3) {
        // 3D: Marsaglia method for efficiency
        T x1, x2, sq;
        do {
            x1 = g.uniform(T(-1), T(1));
            x2 = g.uniform(T(-1), T(1));
            sq = x1*x1 + x2*x2;
        } while (sq >= T(1));
        
        T factor = T(2) * sqrt(T(1) - sq);
        v[0] = x1 * factor;
        v[1] = x2 * factor;
        v[2] = T(1) - T(2) * sq;
    } else {
        // N-D: Gaussian method
        for (size_t i = 0; i < N; ++i) {
            v[i] = g.normal(T(0), T(1));
        }
        v = normalize(v);
    }
    
    return v;
}

// Random point in unit ball (uniform distribution)
template<typename T, size_t N, typename Generator>
vector<T, N> random_in_sphere(Generator& g) {
    // Generate point on sphere and scale by random radius
    vector<T, N> v = random_unit_vector<T, N>(g);
    
    // For uniform distribution in N-D ball, radius^N should be uniform
    T u = g.template uniform<T>(T(0), T(1));
    T radius = pow(u, T(1)/T(N));
    
    return v * radius;
}

// Random point on sphere surface with given radius
template<typename T, size_t N, typename Generator>
vector<T, N> random_on_sphere(Generator& g, T radius = T(1)) {
    return random_unit_vector<T, N>(g) * radius;
}

// Random vector with normal distribution
template<typename T, size_t N, typename Generator>
vector<T, N> random_vector_normal(Generator& g, 
                                 const vector<T, N>& mean = vector<T, N>::zero(),
                                 T stddev = T(1)) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = g.normal(mean[i], stddev);
    }
    return result;
}

// Random vector in box [min, max]^N
template<typename T, size_t N, typename Generator>
vector<T, N> random_in_box(Generator& g, 
                          const vector<T, N>& min,
                          const vector<T, N>& max) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = g.uniform(min[i], max[i]);
    }
    return result;
}

// Random vector on simplex (components sum to 1, all non-negative)
template<typename T, size_t N, typename Generator>
vector<T, N> random_on_simplex(Generator& g) {
    // Generate N exponential random variables
    vector<T, N> v;
    T sum = T(0);
    
    for (size_t i = 0; i < N; ++i) {
        v[i] = -log(g.template uniform<T>(constants<T>::epsilon, T(1)));
        sum += v[i];
    }
    
    // Normalize to sum to 1
    return v / sum;
}

// ============================================================================
// Random Matrix Generation
// ============================================================================

// Random matrix with uniform distribution
template<typename T, size_t M, size_t N, typename Generator>
matrix<T, M, N> random_matrix(Generator& g, T min = T(0), T max = T(1)) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = g.uniform(min, max);
        }
    }
    return result;
}

// Random rotation matrix (SO(N))
template<typename T, size_t N, typename Generator>
matrix<T, N, N> random_rotation_matrix(Generator& g) {
    if constexpr (N == 2) {
        // 2D rotation
        angle<T, radian_tag> theta = random_angle<T, radian_tag>(g);
        T c = cos(theta);
        T s = sin(theta);
        matrix<T, 2, 2> m;
        m(0, 0) = c; m(0, 1) = -s;
        m(1, 0) = s; m(1, 1) = c;
        return m;
    } else if constexpr (N == 3) {
        // 3D rotation via axis-angle representation
        // Generate random axis
        vector<T, 3> axis = random_unit_vector<T, 3>(g);
        
        // Generate random angle
        T angle = g.uniform(T(0), T(2) * constants<T>::pi);
        
        // Build rotation matrix using Rodrigues' formula
        T c = cos(angle);
        T s = sin(angle);
        T t = T(1) - c;
        
        matrix<T, 3, 3> R;
        R(0, 0) = t * axis[0] * axis[0] + c;
        R(0, 1) = t * axis[0] * axis[1] - s * axis[2];
        R(0, 2) = t * axis[0] * axis[2] + s * axis[1];
        
        R(1, 0) = t * axis[0] * axis[1] + s * axis[2];
        R(1, 1) = t * axis[1] * axis[1] + c;
        R(1, 2) = t * axis[1] * axis[2] - s * axis[0];
        
        R(2, 0) = t * axis[0] * axis[2] - s * axis[1];
        R(2, 1) = t * axis[1] * axis[2] + s * axis[0];
        R(2, 2) = t * axis[2] * axis[2] + c;
        
        return R;
    } else {
        // General case: use QR decomposition of random matrix
        // Generate random matrix
        matrix<T, N, N> A = random_matrix<T, N, N>(g, T(-1), T(1));
        
        // Simple Gram-Schmidt orthogonalization
        matrix<T, N, N> Q;
        for (size_t j = 0; j < N; ++j) {
            // Extract column j
            vector<T, N> v;
            for (size_t i = 0; i < N; ++i) {
                v[i] = A(i, j);
            }
            
            // Orthogonalize against previous columns
            for (size_t k = 0; k < j; ++k) {
                vector<T, N> u;
                for (size_t i = 0; i < N; ++i) {
                    u[i] = Q(i, k);
                }
                v = v - dot(v, u) * u;
            }
            
            // Normalize
            v = normalize(v);
            
            // Store in Q
            for (size_t i = 0; i < N; ++i) {
                Q(i, j) = v[i];
            }
        }
        
        // Ensure determinant is +1
        if (determinant(Q) < T(0)) {
            // Flip one column
            for (size_t i = 0; i < N; ++i) {
                Q(i, 0) = -Q(i, 0);
            }
        }
        
        return Q;
    }
}

// Random orthogonal matrix (O(N))
template<typename T, size_t N, typename Generator>
matrix<T, N, N> random_orthogonal_matrix(Generator& g) {
    matrix<T, N, N> m = random_rotation_matrix<T, N>(g);
    
    // Randomly negate determinant
    if (g.template uniform<T>(T(0), T(1)) < T(0.5)) {
        // Flip one row
        for (size_t j = 0; j < N; ++j) {
            m(0, j) = -m(0, j);
        }
    }
    
    return m;
}

// Random symmetric matrix
template<typename T, size_t N, typename Generator>
matrix<T, N, N> random_symmetric_matrix(Generator& g, T min = T(0), T max = T(1)) {
    matrix<T, N, N> m;
    
    // Fill upper triangle and diagonal
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            T value = g.uniform(min, max);
            m(i, j) = value;
            m(j, i) = value;
        }
    }
    
    return m;
}

// Random positive definite matrix
template<typename T, size_t N, typename Generator>
matrix<T, N, N> random_positive_definite_matrix(Generator& g, T condition_number = T(10)) {
    // Generate random orthogonal matrix
    matrix<T, N, N> Q = random_orthogonal_matrix<T, N>(g);
    
    // Generate random eigenvalues with specified condition number
    vector<T, N> eigenvalues;
    T max_eigenvalue = T(1);
    T min_eigenvalue = max_eigenvalue / condition_number;
    
    for (size_t i = 0; i < N; ++i) {
        eigenvalues[i] = g.uniform(min_eigenvalue, max_eigenvalue);
    }
    
    // Construct A = Q * D * Q^T
    matrix<T, N, N> D = matrix<T, N, N>::zero();
    for (size_t i = 0; i < N; ++i) {
        D(i, i) = eigenvalues[i];
    }
    
    return Q * D * transpose(Q);
}

// Random correlation matrix (positive definite with diagonal 1)
template<typename T, size_t N, typename Generator>
matrix<T, N, N> random_correlation_matrix(Generator& g) {
    // Use vine method for generating random correlation matrices
    matrix<T, N, N> R = matrix<T, N, N>::identity();
    
    // Generate partial correlations
    for (size_t k = 0; k < N - 1; ++k) {
        for (size_t i = k + 1; i < N; ++i) {
            // Generate partial correlation
            T r = g.uniform(T(-1), T(1));
            
            // Update correlation matrix
            T factor = sqrt((T(1) - R(k, k)) * (T(1) - R(i, i)));
            R(k, i) = R(i, k) = r * factor;
            
            // Update diagonal elements
            R(i, i) -= r * r * (T(1) - R(k, k));
        }
    }
    
    return R;
}

// ============================================================================
// Convenience functions using thread-local generator
// ============================================================================

template<typename T, size_t N>
inline vector<T, N> random_vector(T min = T(0), T max = T(1)) {
    return random_vector<T, N>(thread_local_rng(), min, max);
}

template<typename T, size_t N>
inline vector<T, N> random_unit_vector() {
    return random_unit_vector<T, N>(thread_local_rng());
}

template<typename T, size_t N>
inline vector<T, N> random_in_sphere() {
    return random_in_sphere<T, N>(thread_local_rng());
}

template<typename T, size_t M, size_t N>
inline matrix<T, M, N> random_matrix(T min = T(0), T max = T(1)) {
    return random_matrix<T, M, N>(thread_local_rng(), min, max);
}

template<typename T, size_t N>
inline matrix<T, N, N> random_rotation_matrix() {
    return random_rotation_matrix<T, N>(thread_local_rng());
}

} // namespace euler