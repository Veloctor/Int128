#include "../include/int128.hpp"
#include <benchmark/benchmark.h>
#include <bitset>
#include <random>
#include <iostream>

using namespace benchmark;
using namespace std;

#define DoBenchmarkOrTest true

#pragma region Helpers
double Sqr(double d) { return d * d; }

template <typename T>
static T RandomBinary()
{
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<uint32_t> dis(CHAR_MIN, CHAR_MAX);
	T obj;
	auto bytePtr = reinterpret_cast<char*>(&obj);
	for (size_t i = 0; i < sizeof(T); ++i) {
		bytePtr[i] = static_cast<char>(dis(gen));
	}
	return obj;
}

static double RandomDouble(double min = 0, double max = 1)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution dis(min, max);
	return dis(gen);
}

static float RandomSingle(float min = 0, float max = 1)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution dis(min, max);
	return dis(gen);
}
#pragma endregion 

#if (DoBenchmarkOrTest)
static void cvt_fp32_int64(State& state)
{
	float a = RandomSingle(INT64_MIN, INT64_MAX);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		int64_t res = a;
		DoNotOptimize(res);
	}
}

static void cvt_int64_fp32(State& state)
{
	auto a = RandomBinary<int64_t>();
	for (auto _ : state)
	{
		DoNotOptimize(a);
		float res = a;
		DoNotOptimize(res);
	}
}

static void cvt_fp32_uint64(State& state)
{
	float a = RandomSingle(0, UINT64_MAX);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		uint64_t res = a;
		DoNotOptimize(res);
	}
}

static void cvt_uint64_fp32(State& state)
{
	auto a = RandomBinary<uint64_t>();
	for (auto _ : state)
	{
		DoNotOptimize(a);
		float res = a;
		DoNotOptimize(res);
	}
}

static void cvt_fp64_int64(State& state)
{
	double a = RandomDouble(INT64_MIN, INT64_MAX);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		int64_t res = a;
		DoNotOptimize(res);
	}
}

static void cvt_int64_fp64(State& state)
{
	int64_t a = RandomBinary<int64_t>();
	for (auto _ : state)
	{
		DoNotOptimize(a);
		double res = static_cast<double>(a);
		DoNotOptimize(res);
	}
}

static void cvt_fp64_uint64(State& state)
{
	double a = RandomDouble(0, UINT64_MAX);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		uint64_t res = a;
		DoNotOptimize(res);
	}
}

static void cvt_uint64_fp64(State& state)
{
	auto a = RandomBinary<uint64_t>();
	for (auto _ : state)
	{
		DoNotOptimize(a);
		double res = static_cast<double>(a);
		DoNotOptimize(res);
	}
}

static void cvt_uint128x4_fp64(State& state)
{
	auto a = RandomBinary<fixed128x4>();
	long4 one = _mm256_set1_epi64x(1);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		auto res = ufixed128_to_double(a.upper, a.lower);
		DoNotOptimize(res);
	}
}

static void cvt_fp64_uint128x4(State& state)
{
	double4 a = _mm256_set_pd(
		RandomDouble(0, Sqr(UINT64_MAX)),
		RandomDouble(0, Sqr(UINT64_MAX)),
		RandomDouble(0, Sqr(UINT64_MAX)),
		RandomDouble(0, Sqr(UINT64_MAX))
		);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		ulong4 upper, lower;
		double_to_ufixed128_full(a, upper, lower);
		DoNotOptimize(upper);
		DoNotOptimize(lower);
	}
}

static void cvt_int128x4_fp64(State& state)
{
	auto a = RandomBinary<fixed128x4>();
	for (auto _ : state)
	{
		DoNotOptimize(a);
		auto res = fixed128_to_double(a.upper, a.lower);
		DoNotOptimize(res);
	}
}

static void cvt_fp64_int128x4(State& state)
{
	double4 a = _mm256_set_pd(
		RandomDouble(Sqr(INT64_MIN), Sqr(INT64_MAX)),
		RandomDouble(Sqr(INT64_MIN), Sqr(INT64_MAX)),
		RandomDouble(Sqr(INT64_MIN), Sqr(INT64_MAX)),
		RandomDouble(Sqr(INT64_MIN), Sqr(INT64_MAX))
		);
	for (auto _ : state)
	{
		DoNotOptimize(a);
		long4 upper, lower;
		double_to_fixed128_full(a, upper, lower);
		DoNotOptimize(upper);
		DoNotOptimize(lower);
	}
}
BENCHMARK(cvt_fp32_uint64);
BENCHMARK(cvt_fp32_int64);
BENCHMARK(cvt_uint64_fp32);
BENCHMARK(cvt_int64_fp32);
BENCHMARK(cvt_fp64_uint64);
BENCHMARK(cvt_fp64_int64);
BENCHMARK(cvt_uint64_fp64);
BENCHMARK(cvt_int64_fp64);
BENCHMARK(cvt_fp64_uint128x4);
BENCHMARK(cvt_fp64_int128x4);
BENCHMARK(cvt_uint128x4_fp64);
BENCHMARK(cvt_int128x4_fp64);
int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	system("pause");
} int main(int, char**);
#else
int main()
{
	//////////////////////////////////////////////
	/*uint64_t count = 0;
	for (double e = 0; e < 127; e += 0.1)
	{
		double a = pow(2.0, e);
		double b = 10000;
		//a = a * pow(2.0, -fixed_frac_bits);
		a = trunc(a);
		fixed128x4 ia = fixed128x4{ a };
		fixed128x4 ib = fixed128x4{ b };
		fixed128x4 iresi = ia;
		double dres = a + b;
		iresi -= ib;
		double4 ires = (double4)(iresi);
		if ((dres - 0.001 > ires.m256d_f64[0]) || (dres + 0.001 < (ires.m256d_f64[0])))
		{
			cout << fixed << dres << endl;
			cout << fixed << ires.m256d_f64[0] << endl;
			cout << bitset<64>(iresi.upper.m256i_u64[0]) << endl;
			cout << bitset<64>(iresi.lower.m256i_u64[0]) << endl << endl;
			count++;
		}
	}
	cout << count;*/
	/////////////////////////////////////////////
	double ad = 1000;
	double bd = -0.001;
	cout << "-----fixed128-----" << '\n';
	auto as = fixed128{ad};
	auto bs = fixed128{bd};
	auto nas = fixed128{-ad};
	auto nbs = fixed128{-bd};
	cout << "a:	" << static_cast<double>(as) << '\n';
	cout << "b:	" << static_cast<double>(bs) << '\n';
	fixed128 adds = as;
	for (int i = 0; i < 1000000; i++) {
		adds += bs;
	}
	cout << "add:	" << static_cast<double>(adds) << '\n';
	fixed128 subs = as;
	for (int i = 0; i < 10000; i++) {
		subs -= bs;
	}
	cout << "sub:	" << static_cast<double>(subs) << '\n';
	cout << "a:	" << static_cast<double>(nas) << '\n';
	cout << "b:	" << static_cast<double>(nbs) << '\n';
	cout << "add:	" << static_cast<double>(nas + nbs) << '\n';
	cout << "sub:	" << static_cast<double>(nas - nbs) << '\n';

	auto a = fixed128x4{ad};
	auto b = fixed128x4{bd};
	auto na = fixed128x4{-ad};
	auto nb = fixed128x4{-bd};
	cout << "-----fixed128x4-----" << '\n';
	cout << "a:	" << static_cast<double>(a[0]) << '\n';
	cout << "b:	" << static_cast<double>(b[0]) << '\n';
	cout << "add:	" << static_cast<double>((a + b)[0]) << '\n';
	cout << "sub:	" << static_cast<double>((a - b)[0]) << '\n';
	cout << "a:	" << static_cast<double>(na[0]) << '\n';
	cout << "b:	" << static_cast<double>(nb[0]) << '\n';
	cout << "add:	" << static_cast<double>((na + nb)[0]) << '\n';
	cout << "sub:	" << static_cast<double>((na - nb)[0]) << '\n';
}
#endif
