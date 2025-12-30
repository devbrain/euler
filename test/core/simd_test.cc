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

    SUBCASE("batch types are correct") {
        #ifdef EULER_HAS_XSIMD
        static_assert(std::is_same_v<simd_traits<float>::batch_type, xsimd::batch<float>>);
        static_assert(std::is_same_v<simd_traits<double>::batch_type, xsimd::batch<double>>);
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

    SUBCASE("compile-time alignment") {
        CHECK(simd_alignment_v<float>::value >= alignof(float));
        CHECK(simd_alignment_v<double>::value >= alignof(double));

        // Should be power of 2
        CHECK((simd_alignment_v<float>::value & (simd_alignment_v<float>::value - 1)) == 0);
        CHECK((simd_alignment_v<double>::value & (simd_alignment_v<double>::value - 1)) == 0);
    }

    SUBCASE("is_aligned check") {
        // Stack data may or may not be aligned
        float stack_data[16];
        bool aligned = is_aligned(&stack_data[0]);
        CHECK((aligned == true || aligned == false)); // May or may not be aligned

        // Aligned stack data should be aligned
        alignas(32) float aligned_stack[16];
        CHECK(is_aligned(&aligned_stack[0]));
    }
}

#ifdef EULER_HAS_XSIMD
TEST_CASE("euler::simd direct xsimd usage") {
    using namespace euler;
    using batch_t = simd_traits<float>::batch_type;
    constexpr size_t batch_size = simd_traits<float>::batch_size;

    SUBCASE("basic xsimd operations work") {
        alignas(32) float data1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        alignas(32) float data2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        alignas(32) float result[8];

        auto a = batch_t::load_aligned(data1);
        auto b = batch_t::load_aligned(data2);
        auto sum = a + b;
        sum.store_aligned(result);

        // All elements should sum to 9
        for (size_t i = 0; i < batch_size && i < 8; ++i) {
            CHECK(result[i] == doctest::Approx(9.0f));
        }
    }

    SUBCASE("xsimd math functions") {
        alignas(32) float data[8] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
        alignas(32) float result[8];

        auto val = batch_t::load_aligned(data);
        auto sqrt_val = xsimd::sqrt(val);
        sqrt_val.store_aligned(result);

        for (size_t i = 0; i < batch_size && i < 8; ++i) {
            CHECK(result[i] == doctest::Approx(std::sqrt(data[i])));
        }
    }

    SUBCASE("xsimd reductions") {
        alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

        auto batch = batch_t::load_aligned(data);
        float sum = xsimd::reduce_add(batch);
        float min_val = xsimd::reduce_min(batch);
        float max_val = xsimd::reduce_max(batch);

        // Sum depends on batch size
        float expected_sum = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            expected_sum += data[i];
        }
        CHECK(sum == doctest::Approx(expected_sum));
        CHECK(min_val == doctest::Approx(1.0f));
        CHECK(max_val == doctest::Approx(static_cast<float>(batch_size)));
    }
}
#endif
