//set number of fractional bits for fixed point numbers here
#define fixed_frac_bits 36
#include "AVX2Ext.h"
#include <climits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <immintrin.h>

const double ULongMaxValueSD = (double)ULLONG_MAX;

struct fixed64
{
    union
    {
        int64_t sbits;
        uint64_t ubits;
    };

    explicit fixed64(double fp)
    {
        sbits = (int64_t)(fp * pow(2.0, fixed_frac_bits));
    }

    explicit operator double()
    {
        return sbits * pow(2.0, -fixed_frac_bits);
    }

    operator int64_t()
    {
        return sbits;
    }

    operator uint64_t()
    {
        return ubits;
    }
};

#define fixed64_maxvalue pow(2.0, -fixed_frac_bits) * LLONG_MAX
#define fixed64_minvalue pow(2.0, -fixed_frac_bits) * LLONG_MIN

struct fixed64x4
{
    __m256i bits;

    fixed64x4(double fp)
    {
        fp *= pow(2.0, fixed_frac_bits);
        auto vfp = _mm256_set1_pd(fp);
        bits = double_to_int64_fast(vfp);
    }

    fixed64x4(int64_t i)
    {
        i <<= fixed_frac_bits;
        bits = _mm256_set1_epi64x(i);
    }
    
    fixed64x4(const __m256i& i)
    {
        bits = _mm256_slli_epi64(i, fixed_frac_bits);
    }

    fixed64x4(__m256d vfp)
    {
        const auto vfactor = _mm256_set1_pd(pow(2.0, fixed_frac_bits));
        vfp = _mm256_mul_pd(vfp, vfactor);
        bits = double_to_int64_fast(vfp);
    }

    fixed64x4(__m128 vfp32)
    {
        const auto vfactor = _mm_set_ps1(std::pow(2.0f, fixed_frac_bits));
        vfp32 = _mm_mul_ps(vfp32, vfactor);
        auto vfp = _mm256_cvtps_pd(vfp32);
        bits = double_to_int64_fast(vfp);
    }

    operator __m256d()
    {
        const auto vfactor = _mm256_set1_pd(pow(2.0, -fixed_frac_bits));
        auto vfp = int64_to_double_fast_precise(bits);
        return _mm256_mul_pd(vfp, vfactor);
    }

    operator __m128()
    {
        const auto vfactor = _mm_set_ps1(pow(2.0f, -fixed_frac_bits));
        auto vfp = int64_to_double_fast_precise(bits);
        auto vfp32 = _mm256_cvtpd_ps(vfp);
        return _mm_mul_ps(vfp32, vfactor);
    }
};

struct fixed128
{
    uint64_t upper, lower;

    explicit fixed128(double fp)
    {
        fp *= pow(2.0, fixed_frac_bits);
        double abs = std::abs(fp);
        double div = abs * (1.0 / ULongMaxValueSD);
        double c1trunc = std::trunc(div); //maybe trunc better?
        uint64_t c1 = static_cast<uint64_t>(div);
        uint64_t c0 = static_cast<uint64_t>(std::fma(c1trunc, -ULongMaxValueSD, abs));
        if (fp < 0.0)
        {
            c1 = 0 - c1;
            c1 -= c0 > 0;
            c0 = 0 - c0;
        }
        upper = c1;
        lower = c0;
    }
    //warn: no value shift, direct bit assign
    explicit fixed128(uint64_t _upper, uint64_t _lower)
    {
        upper = _upper;
        lower = _lower;
    }

    fixed128(fixed64 _lower)
    {
        upper = 0;
        lower = *(uint64_t*)&_lower;
    }

    fixed128()
    {
        upper = 0;
        lower = 0;
    }

    bool is_negative()
    {
        return upper >> 63;
    }

    explicit operator double()
    {
        auto c1 = upper;
        auto c0 = lower;
        if (upper >> 63)
        {
            c1 = 0 - c1;
            c1 -= c0 > 0;
            c0 = 0 - c0;
            return -std::fma(c1, ULongMaxValueSD, c0) * pow(2.0, -fixed_frac_bits);
        }
        else return std::fma(c1, ULongMaxValueSD, c0) * pow(2.0, -fixed_frac_bits);
    }
    
    fixed128& operator+=(fixed128& value)
    {
        auto carry = _addcarry_u64(0, lower, value.lower, &lower);
        _addcarry_u64(carry, upper, value.upper, &upper);
        return *this;
    }

    fixed128& operator+=(fixed64 value)
    {
        upper += _addcarry_u64(0, lower, value, &lower);
        return *this;
    }

    fixed128& operator+=(double value)
    {
        if (value < 0.0)
        {
            value = abs(value) * pow(2.0, fixed_frac_bits);//==bit_cast<double>(bit_cast<uint64_t>(value) & LLONG_MIN)
            double div = value * (1.0 / ULongMaxValueSD);
            double upperTrunc = std::floor(div); //maybe trunc better?
            uint64_t targetUpper = static_cast<uint64_t>(div);
            uint64_t targetLower = static_cast<uint64_t>(std::fma(upperTrunc, -ULongMaxValueSD, value));
            //减法
            lower -= lower < targetLower;
            upper -= targetUpper;
            lower -= targetLower;
        }
        else if(value > 0.0)
        {
            value *= pow(2.0, fixed_frac_bits);
            double div = value * (1.0 / ULongMaxValueSD);
            double upperTrunc = std::floor(div); //maybe trunc better?
            uint64_t targetUpper = static_cast<uint64_t>(div);
            uint64_t targetLower = static_cast<uint64_t>(std::fma(upperTrunc, -ULongMaxValueSD, value));
            //加法
            lower += targetLower;
            upper = upper + targetUpper + (lower < targetLower);
        }
        return *this;
    }

    fixed128& operator-=(fixed128& value)
    {
        auto borrow = _subborrow_u64(0, lower, value.lower, &lower);
        _subborrow_u64(borrow, upper, value.upper, &upper);
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const fixed128& i128);
};

std::ostream& operator<<(std::ostream& os, const fixed128& i128) {
    return os << std::hex << i128.upper << std::setw(16) << std::setfill('0') << i128.lower;
}

fixed128 operator+(const fixed128& a, const fixed128& b)
{
    fixed128 res;
    auto carry = _addcarry_u64(0, a.lower, b.lower, &res.lower);
    _addcarry_u64(carry, a.upper, b.upper, &res.upper);
    return res;
}

fixed128 operator-(const fixed128& a, const fixed128& b)
{
    fixed128 res;
    auto borrow = _subborrow_u64(0, a.lower, b.lower, &res.lower);
    _subborrow_u64(borrow, a.upper, b.upper, &res.upper);
    return res;
}

double sub_to_double(const fixed128& a, const fixed128& b)
{
    //减法
    uint64_t lower = a.lower - (a.lower < b.lower);
    uint64_t upper = a.upper - b.upper;
    lower -= b.lower;

    //如果是负数,则反转
    if (upper > 0x7FFFFFFFFFFFFFFFu)
    {
        uint64_t neglower = (lower ^ (-1LL)) + 1;
        uint64_t borrow = lower == 0 ? -1LL : 0;

        uint64_t negupper = (upper ^ (-1LL)) - borrow;
        return std::fma(negupper, -ULongMaxValueSD, -static_cast<double>(neglower));// *TwoP32_InvSD;
    }
    else return std::fma(upper, ULongMaxValueSD, lower);// *TwoP32_InvSD;
}

#define SignMaskPI _mm256_set1_epi64x(LLONG_MIN)

struct fixed128x4
{
    __m256i upper, lower;

    explicit fixed128x4(const __m256d& x)
    {
        double_to_fixed128_full(x, upper, lower);
    }

    explicit fixed128x4(double x)
    {
        auto x4 = _mm256_set1_pd(x);
        *this = fixed128x4{ x4 };
    }

    fixed128x4()
    {
        upper = _mm256_setzero_si256();
        lower = _mm256_setzero_si256();
    }

    fixed128x4(fixed64 x, fixed64 y, fixed64 z, fixed64 w)
    {
        upper = _mm256_setzero_si256();
        lower = _mm256_set_epi64x(x, y, z, w);
    }

    fixed128x4(fixed64 x)
    {
        upper = _mm256_setzero_si256();
        lower = _mm256_set1_epi64x(x.sbits);
    }

    fixed128x4(fixed128 x)
    {
        upper = _mm256_set1_epi64x(x.upper);
        lower = _mm256_set1_epi64x(x.lower);
    }

    fixed128x4(fixed128 x, fixed128 y, fixed128 z, fixed128 w)
    {
        upper = _mm256_set_epi64x(x.upper, y.upper, z.upper, w.upper);
        lower = _mm256_set_epi64x(x.lower, y.lower, z.lower, w.lower);
    }

    fixed128x4(__m256i upper, __m256i lower)
    {
        this->upper = upper;
        this->lower = lower;
    }

    explicit operator __m256d()
    {
        return fixed128_to_double(upper, lower);
    }

    void operator+=(fixed128x4 value)
    {
        this->lower = _mm256_add_epi64(this->lower, value.lower);
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(value.lower, SignMaskPI), _mm256_add_epi64(this->lower, SignMaskPI));
        this->upper = _mm256_sub_epi64(_mm256_add_epi64(this->upper, value.upper), carry);
    }

    void operator+=(__m256i lower)
    {
        this->lower = _mm256_add_epi64(this->lower, lower);
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(lower, SignMaskPI), _mm256_add_epi64(this->lower, SignMaskPI));
        this->upper = _mm256_sub_epi64(this->upper, carry);
    }
    
    void operator-=(fixed128x4 value)
    {
        __m256i up = _mm256_sub_epi64(this->upper, value.upper);
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(value.lower, SignMaskPI), _mm256_add_epi64(this->lower, SignMaskPI));
        this->lower = _mm256_sub_epi64(this->lower, value.lower);
        this->upper = _mm256_add_epi64(up, carry);
    }

    void operator-=(__m256i value)
    {
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(value, SignMaskPI), _mm256_add_epi64(this->lower, SignMaskPI));
        this->lower = _mm256_sub_epi64(this->lower, value);
        this->upper = _mm256_add_epi64(this->upper, carry);
    }
};

fixed128x4 operator+(fixed128x4 a, const fixed128x4& b)
{
    a += b;
    return a;
}

fixed128x4 operator-(fixed128x4 a, const fixed128x4& b)
{
    a -= b;
    return a;
}

fixed128x4 gather_3xfix128x4_last_elem_into1(const fixed128x4& in1, const fixed128x4& in2, const fixed128x4& in3)
{
    fixed128x4 out{};
    //operations:
    //out.lower.m256i_i64[0] = in1.lower.m256i_i64[3];
    //out.lower.m256i_i64[1] = in2.lower.m256i_i64[3];
    //out.lower.m256i_i64[2] = in3.lower.m256i_i64[3];
    //out.upper.m256i_i64[0] = in1.upper.m256i_i64[3];
    //out.upper.m256i_i64[1] = in2.upper.m256i_i64[3];
    //out.upper.m256i_i64[2] = in3.upper.m256i_i64[3];
    auto c = _mm256_permute4x64_epi64(in1.lower, 0b11);
    auto d = _mm256_permute4x64_epi64(in2.lower, 0b1100);
    c = _mm256_blend_epi32(c, d, 0b1100);
    auto lo = _mm256_srli_si256(in3.lower, 8);
    out.lower = _mm256_blend_epi32(c, lo, 0b11110000);
    auto a = _mm256_permute4x64_epi64(in1.upper, 0b11);
    auto b = _mm256_permute4x64_epi64(in2.upper, 0b1100);
    a = _mm256_blend_epi32(a, b, 0b1100);
    auto hi = _mm256_srli_si256(in3.upper, 8);
    out.upper = _mm256_blend_epi32(a, hi, 0b11110000);
    return out;
}

void distribute_in4xyz_to_3xfix128x4_last_elem(fixed128x4& inout1, fixed128x4& inout2, fixed128x4& inout3, const fixed128x4& in4)
{
    //operations:
    //inout1.lower.m256i_i64[3] = in4.lower.m256i_i64[0];
    //inout2.lower.m256i_i64[3] = in4.lower.m256i_i64[1];
    //inout3.lower.m256i_i64[3] = in4.lower.m256i_i64[2];
    //inout1.upper.m256i_i64[3] = in4.upper.m256i_i64[0];
    //inout2.upper.m256i_i64[3] = in4.upper.m256i_i64[1];
    //inout3.upper.m256i_i64[3] = in4.upper.m256i_i64[2];
    auto a = _mm256_permute4x64_epi64(in4.lower, 0);
    inout1.lower = _mm256_blend_epi32(inout1.lower, a, 0b11000000);
    auto b = _mm256_permute4x64_epi64(in4.lower, 0b01000000);
    inout2.lower = _mm256_blend_epi32(inout2.lower, b, 0b11000000);
    auto c = _mm256_slli_si256(in4.lower, 8);
    inout3.lower = _mm256_blend_epi32(inout3.lower, c, 0b11000000);
    auto d = _mm256_permute4x64_epi64(in4.upper, 0);
    inout1.upper = _mm256_blend_epi32(inout1.upper, d, 0b11000000);
    auto e = _mm256_permute4x64_epi64(in4.upper, 0b01000000);
    inout2.upper = _mm256_blend_epi32(inout2.upper, e, 0b11000000);
    auto f = _mm256_slli_si256(in4.upper, 8);
    inout3.upper = _mm256_blend_epi32(inout3.upper, f, 0b11000000);
}