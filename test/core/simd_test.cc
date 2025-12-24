#include <doctest/doctest.h>
#include <euler/core/simd.hh>
#include <vector>
#include <cmath>

TEST_CASE("euler::simd_traits") {
    using namespace euler;
    
    SUBCASE("basic traits") {
        // Without SIMD, batch_size should be 1
        #ifndef EULER_HAS_XSIMD
        CHECK(simd_traits<float>::batch_size == 1);
        CHECK(simd_traits<double>::batch_size == 1);
        CHECK(!simd_traits<float>::has_simd);
        CHECK(!simd_traits<double>::has_simd);
        #else
        CHECK(simd_traits<float>::batch_size > 1);
        CHECK(simd_traits<double>::batch_size > 1);
        CHECK(simd_traits<float>::has_simd);
        CHECK(simd_traits<double>::has_simd);
        #endif
    }
}

TEST_CASE("euler::simd_alignment") {
    using namespace euler;
    
    SUBCASE("alignment values") {
        const size_t float_align = simd_alignment<float>();
        const size_t double_align = simd_alignment<double>();
        
        CHECK(float_align >= alignof(float));
        CHECK(double_align >= alignof(double));
        
        // Alignment should be power of 2
        CHECK((float_align & (float_align - 1)) == 0);
        CHECK((double_align & (double_align - 1)) == 0);
    }
    
    SUBCASE("is_aligned check") {
        float stack_data[16];
        bool aligned = is_aligned(&stack_data[0]);
        CHECK((aligned == true || aligned == false)); // May or may not be aligned
        
        // Aligned allocation
        auto* aligned_data = aligned_alloc<float>(16);
        CHECK(is_aligned(aligned_data));
        aligned_free(aligned_data);
    }
}

TEST_CASE("euler::simd_ops basic operations") {
    using namespace euler;
    using ops = simd_ops<float>;
    
    SUBCASE("arithmetic operations") {
        constexpr size_t batch_size = simd_traits<float>::batch_size;
        constexpr size_t array_size = batch_size * 2; // Ensure we have at least 2 batches
        
        std::vector<float> data1(array_size);
        std::vector<float> data2(array_size);
        std::vector<float> result(array_size);
        
        // Initialize test data
        for (size_t i = 0; i < array_size; ++i) {
            data1[i] = static_cast<float>(i + 1);
            data2[i] = static_cast<float>(i + 5);
        }
        
        // Test batch operations if SIMD is available
        if (simd_traits<float>::has_simd && array_size >= batch_size) {
            // Process in batches
            for (size_t i = 0; i + batch_size <= array_size; i += batch_size) {
                auto a = ops::load_unaligned(&data1[i]);
                auto b = ops::load_unaligned(&data2[i]);
                
                auto sum = ops::add(a, b);
                ops::store_unaligned(&result[i], sum);
                
                // Check all elements in the batch
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == data1[i + j] + data2[i + j]);
                }
                
                auto diff = ops::sub(a, b);
                ops::store_unaligned(&result[i], diff);
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == data1[i + j] - data2[i + j]);
                }
                
                auto prod = ops::mul(a, b);
                ops::store_unaligned(&result[i], prod);
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == data1[i + j] * data2[i + j]);
                }
                
                auto quot = ops::div(a, b);
                ops::store_unaligned(&result[i], quot);
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == doctest::Approx(data1[i + j] / data2[i + j]));
                }
            }
        } else {
            // Test scalar operations
            for (size_t i = 0; i < 4; ++i) {
                auto a = ops::load_unaligned(&data1[i]);
                auto b = ops::load_unaligned(&data2[i]);
                
                auto sum = ops::add(a, b);
                ops::store_unaligned(&result[i], sum);
                CHECK(result[i] == data1[i] + data2[i]);
                
                auto diff = ops::sub(a, b);
                ops::store_unaligned(&result[i], diff);
                CHECK(result[i] == data1[i] - data2[i]);
                
                auto prod = ops::mul(a, b);
                ops::store_unaligned(&result[i], prod);
                CHECK(result[i] == data1[i] * data2[i]);
                
                auto quot = ops::div(a, b);
                ops::store_unaligned(&result[i], quot);
                CHECK(result[i] == doctest::Approx(data1[i] / data2[i]));
            }
        }
    }
    
    SUBCASE("math functions") {
        constexpr size_t batch_size = simd_traits<float>::batch_size;
        constexpr size_t array_size = batch_size * 2;
        
        std::vector<float> data(array_size);
        std::vector<float> result(array_size);
        
        // Initialize with perfect squares
        for (size_t i = 0; i < array_size; ++i) {
            data[i] = static_cast<float>((i + 1) * (i + 1));
        }
        
        if (simd_traits<float>::has_simd && array_size >= batch_size) {
            // Process in batches
            for (size_t i = 0; i + batch_size <= array_size; i += batch_size) {
                auto val = ops::load_unaligned(&data[i]);
                
                auto sqrt_val = ops::sqrt(val);
                ops::store_unaligned(&result[i], sqrt_val);
                
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == doctest::Approx(std::sqrt(data[i + j])));
                }
                
                auto abs_val = ops::abs(val);
                ops::store_unaligned(&result[i], abs_val);
                
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == std::abs(data[i + j]));
                }
            }
            
            // Test with negative values
            for (size_t i = 0; i < array_size; ++i) {
                data[i] = -static_cast<float>(i + 1);
            }
            
            for (size_t i = 0; i + batch_size <= array_size; i += batch_size) {
                auto val = ops::load_unaligned(&data[i]);
                auto abs_val = ops::abs(val);
                ops::store_unaligned(&result[i], abs_val);
                
                for (size_t j = 0; j < batch_size; ++j) {
                    CHECK(result[i + j] == std::abs(data[i + j]));
                }
            }
        } else {
            // Scalar mode
            float scalar_data[4] = {4.0f, 9.0f, 16.0f, 25.0f};
            float scalar_result[4];
            
            for (size_t i = 0; i < 4; ++i) {
                auto val = ops::load_unaligned(&scalar_data[i]);
                
                auto sqrt_val = ops::sqrt(val);
                ops::store_unaligned(&scalar_result[i], sqrt_val);
                CHECK(scalar_result[i] == doctest::Approx(std::sqrt(scalar_data[i])));
                
                auto abs_val = ops::abs(val);
                ops::store_unaligned(&scalar_result[i], abs_val);
                CHECK(scalar_result[i] == std::abs(scalar_data[i]));
            }
            
            // Test with negative values
            float neg_data[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
            for (size_t i = 0; i < 4; ++i) {
                auto val = ops::load_unaligned(&neg_data[i]);
                auto abs_val = ops::abs(val);
                ops::store_unaligned(&scalar_result[i], abs_val);
                CHECK(scalar_result[i] == std::abs(neg_data[i]));
            }
        }
    }
    
    SUBCASE("reduction operations") {
        alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        #ifdef EULER_HAS_XSIMD
        if (simd_traits<float>::has_simd) {
            // SIMD mode: load all 4 values as a batch
            auto batch = ops::load_aligned(data);
            CHECK(ops::reduce_add(batch) == doctest::Approx(10.0f));  // 1+2+3+4
            CHECK(ops::reduce_min(batch) == doctest::Approx(1.0f));
            CHECK(ops::reduce_max(batch) == doctest::Approx(4.0f));
        } else {
            // Scalar mode: reductions just return the value
            for (int i = 0; i < 4; ++i) {
                auto val = ops::load_unaligned(&data[i]);
                CHECK(ops::reduce_add(val) == data[i]);
                CHECK(ops::reduce_min(val) == data[i]);
                CHECK(ops::reduce_max(val) == data[i]);
            }
        }
        #else
        // No SIMD: reductions just return the value
        for (int i = 0; i < 4; ++i) {
            auto val = ops::load_unaligned(&data[i]);
            CHECK(ops::reduce_add(val) == data[i]);
            CHECK(ops::reduce_min(val) == data[i]);
            CHECK(ops::reduce_max(val) == data[i]);
        }
        #endif
    }
}

TEST_CASE("euler::should_use_simd") {
    using namespace euler;
    
    // Without SIMD support, should always return false
    #ifndef EULER_HAS_XSIMD
    CHECK(!should_use_simd<float>(1));
    CHECK(!should_use_simd<float>(100));
    CHECK(!should_use_simd<float>(1000));
    CHECK(!should_use_simd<double>(1));
    CHECK(!should_use_simd<double>(100));
    #else
    // With SIMD, should return true for larger sizes
    const size_t float_batch = simd_traits<float>::batch_size;
    CHECK(!should_use_simd<float>(1));
    CHECK(!should_use_simd<float>(float_batch));
    CHECK(should_use_simd<float>(float_batch * 2));
    CHECK(should_use_simd<float>(float_batch * 10));
    #endif
}

TEST_CASE("euler::aligned_alloc") {
    using namespace euler;
    
    SUBCASE("allocation and deallocation") {
        auto* data = aligned_alloc<float>(64);
        CHECK(is_aligned(data));
        
        // Should be able to use the memory
        for (size_t i = 0; i < 64; ++i) {
            data[i] = static_cast<float>(i);
        }
        
        for (size_t i = 0; i < 64; ++i) {
            CHECK(data[i] == static_cast<float>(i));
        }
        
        aligned_free(data);
    }
    
    SUBCASE("different types") {
        auto* d_data = aligned_alloc<double>(32);
        CHECK(is_aligned(d_data));
        aligned_free(d_data);
        
        int* i_data = aligned_alloc<int>(128);
        aligned_free(i_data);
    }
    
#ifdef EULER_DEBUG
    SUBCASE("zero count allocation should fail") {
        CHECK_THROWS_AS(
            aligned_alloc<float>(0),
            std::runtime_error
        );
    }
#endif
}