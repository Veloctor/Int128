#include "../include/int128.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <ctime>
#include <bitset>
using namespace std;
using namespace benchmark;

#if false
static void cvt_fp64_int128(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	double a = static_cast<double>(rand()) * static_cast<double>(RAND_MAX);
	for (auto _ : state)
	{
		a += 0.114514;
		fixed128 res{ a };
		DoNotOptimize(res);
	}
}
static void cvt_int128_fp64(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	fixed128 a{ rand(), rand()};

	for (auto _ : state)
	{
		a += 1LL;
		auto res = (double)a;
		DoNotOptimize(res);
	}
}

static void cvt_uint128x4_fp64(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	fixed128x4 a{ rand(), rand(), rand(), rand() };
	__m256i one = _mm256_set1_epi64x(1);
	for (auto _ : state)
	{
		a += one;
		auto res = ufixed128_to_double(a.upper, a.lower);
		DoNotOptimize(res);
	}
}

static void cvt_fp64_uint128x4(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	__m256d a = _mm256_set_pd(rand(), rand(), rand(), rand());
	for (auto _ : state)
	{
		__m256i upper, lower;
		double_to_ufixed128_full(a, upper, lower);
		DoNotOptimize(upper);
		DoNotOptimize(lower);
	}
}

static void cvt_int128x4_fp64(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	fixed128x4 a{ rand(), rand(), rand(), rand() };

	for (auto _ : state)
	{
		auto res = fixed128_to_double(a.upper, a.lower);
		DoNotOptimize(res);
	}
}

static void cvt_fp64_int128x4(benchmark::State& state)
{
	srand(static_cast<unsigned>(time(0)));
	__m256d a = _mm256_set_pd(rand(), rand(), rand(), rand());
	for (auto _ : state)
	{
		__m256i upper, lower;
		double_to_fixed128_full(a, upper, lower);
		DoNotOptimize(upper);
		DoNotOptimize(lower);
	}
}
BENCHMARK(cvt_fp64_int128);
BENCHMARK(cvt_int128_fp64);
BENCHMARK(cvt_uint128x4_fp64);
BENCHMARK(cvt_fp64_uint128x4);
BENCHMARK(cvt_int128x4_fp64);
BENCHMARK(cvt_fp64_int128x4);
BENCHMARK_MAIN();
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
		__m256d ires = (__m256d)(iresi);
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
	cout << "-----fixed128-----" << endl;
	auto as = fixed128{ ad };
	auto bs = fixed128{ bd };
	auto nas = fixed128{ -ad };
	auto nbs = fixed128{ -bd };
	cout << "a:	" << (double)(as) << endl;
	cout << "b:	" << (double)(bs) << endl;
	fixed128 adds = as;
	for (int i = 0; i < 1000000; i++)
	{
		adds += bs;
	}
	cout << "add:	" << (double)(adds) << endl;
	fixed128 subs = as;
	for (int i = 0; i < 10000; i++)
	{
		subs -= bs;
	}
	cout << "sub:	" << (double)(subs) << endl;
	cout << "a:	" << (double)(nas) << endl;
	cout << "b:	" << (double)(nbs) << endl;
	cout << "add:	" << (double)(nas + nbs) << endl;
	cout << "sub:	" << (double)(nas - nbs) << endl;

	auto a = fixed128x4{ ad };
	auto b = fixed128x4{ bd };
	auto na = fixed128x4{ -ad };
	auto nb = fixed128x4{ -bd };
	cout << "-----fixed128x4-----" << endl;
	cout << "a:	" << (double)(a) << endl;
	cout << "b:	" << (double)(b) << endl;
	cout << "add:	" << (double)(a + b) << endl;
	cout << "sub:	" << (double)(a - b) << endl;
	cout << "a:	" << (double)(na) << endl;
	cout << "b:	" << (double)(nb) << endl;
	cout << "add:	" << (double)(na + nb) << endl;
	cout << "sub:	" << (double)(na - nb) << endl;
}
#endif