#include <iomanip>
#include <iostream>
#include <numeric>
#include "Maxtrix.h"
#include "dbg.h"

#include <windows.h>

using namespace std;
using namespace TimerSys;

static long num_steps = 1000000;



double CalculatePi()
{		
	double x;
	double pi;
	double sum=0.0;
	
	double step = 1.0/(double)num_steps;
	for (long i=1; i<num_steps; i++)
	{
		x = (i+0.5)*step;
		sum += 4.0/(1+x*x);
	}
	pi = step*sum;

	cout << "CalculatePi:" << pi << endl;
			
	return pi;
}

double CalculatePiMp()
{
	int tid;
	double x;
	double pi;
	double sum[NUM_THREADS]={0.0};

	double step = 1.0/(double)num_steps;

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(x, tid)
	{
		tid = omp_get_thread_num();
		for (long i=tid; i<num_steps; i+=NUM_THREADS)
		{
			x = (i+0.5)*step;
			sum[tid] += 4.0/(1.0+x*x);
		}
	}

	double total = 0.0;
	for (int i=0; i<NUM_THREADS; i++)
	{
		total += sum[i];
	}

	pi = step*total;

	cout << "CalculatePiMp:" << pi << endl;

	return pi;
}

double CalculatePiMpFor()
{
	double x;
	int tid;

	double pi;
	double sum[NUM_THREADS]={0.0};
	vector<vector<long>> idx(NUM_THREADS);

	double step = 1.0/(double)num_steps;

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(static,4) private(tid, x)
	for (long i=0; i<num_steps; i++)
	{
		tid = omp_get_thread_num();
		x = (i+0.5)*step;
		sum[tid] += 4.0/(1.0+x*x);
		idx[tid].push_back(i);
	}
	

	double total = 0.0;
	for (int i=0; i<NUM_THREADS; i++)
	{
		total += sum[i];
	}
	pi = step*total;

	cout << "CalculatePiMpFor:" << pi << endl;

	return pi;
}

double CalculatePiMpForWithReduction()
{
	double x;
	int tid;

	double pi;
	double sum=0.0;
	vector<vector<long>> idx(NUM_THREADS);

	double step = 1.0/(double)num_steps;

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(tid, x)
	{
		tid = omp_get_thread_num();
#pragma omp for schedule(static,4) reduction(+:sum)
		for (long i=0; i<num_steps; i++)
		{
			x = (i+0.5)*step;
			sum += 4.0/(1.0+x*x);
			idx[tid].push_back(i);
		}
	}

	pi = step*sum;

	cout << "CalculatePiMpForWithReduction:" << pi << endl;

	return pi;
}

double CalculatePiMpCritical()
{
	int tid;

	double pi=0.0;
	vector<vector<long>> idx(NUM_THREADS);

	double step = 1.0/(double)num_steps;

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(tid)
	{
		double x;
		double sum = 0.0;
		tid = omp_get_thread_num();
		for (long i=tid; i<num_steps; i+=NUM_THREADS)
		{
			x = (i+0.5)*step;
			sum += 4.0/(1.0+x*x);
			idx[tid].push_back(i);
		}

//#pragma omp critical
#pragma omp atomic
		pi += sum*step;
	}

		
	cout << "CalculatePiMpCritical:" << pi << endl;

	return pi;
}

void MonteCarlo()
{
	long long max=10000;
	long long count=0;
	double x,y,z,bulk;
	time_t t;

	srand((unsigned) time(&t));
	for(long long i=0; i<max; i++)
	{
		for(long long j=0; j<max; j++)
		{ 
			x = rand(); x = x/32767;
			y = rand(); y = y/32767;
			z = rand(); z = z/32767;
			if( (x*x + y*y + z*z) <= 1 )
				count++;
		}
	}		
	bulk = 8*(double(count)/max);

	cout << "MonteCarlo Sphere bulk is :" << bulk << endl;
}

void MonteCarloMp()
{
	long long max=10000;
	long long count=0;
	double x,y,z,bulk;
	time_t t;
	srand((unsigned) time(&t));

	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(x,y,z)
	{
#pragma omp for reduction(+:count)
		for(long long i=0; i<max; i++)
		{
			for(long long j=0; j<max; j++)
			{ 
				x = rand(); x = x/32767;
				y = rand(); y = y/32767;
				z = rand(); z = z/32767;
				if( (x*x + y*y + z*z) <= 1 )
					count++;
			}
		}
	}
		
	bulk = 8*(double(count)/max);

	cout << "MonteCarloMp Sphere bulk is :" << bulk << endl;
}


int main()
{
#ifdef _DEBUG
#ifdef _MEMCHKE
	DetectMemoryLeaks(true);
#endif
#endif

	system("color 0A");
	
	cout << setiosflags(ios::fixed);
	cout << setprecision(10);


	//short result[8] = {0};
	//int A[] = { 0xEF01, 0xEE02, 0xEA03, 0xFF04, 0xEC05, 0xEB06, 0xEF07, 0xEF08 };
	//int B[] = { 0xEF01, 0xEE02, 0xEA03, 0xFF04, 0xEC05, 0xEB06, 0xEF07, 0xEF08 };
	//__m128i X, Y, Z8, Z16, Z32, Z64, Zsi8, Zsi16, Zsu8, Zsu16;
	//X = _mm_loadu_si128((__m128i*)A);
	//Y = _mm_loadu_si128((__m128i*)B);
	//Z8 = _mm_add_epi8(X, Y);
	//Z16 = _mm_add_epi16(X, Y);
	//Z32 = _mm_add_epi32(X, Y);
	//Z64 = _mm_add_epi64(X, Y);

	//Zsi8 = _mm_adds_epi8(X, Y);
	//Zsi16 = _mm_adds_epi16(X, Y);

	//Zsu8 = _mm_adds_epu8(X, Y);
	//Zsu16 = _mm_adds_epu16(X, Y);

	//__m128i Z = _mm_madd_epi16(X, Y);// int result[0] = A[0]*B[0]+A[1]*B[1];...A/B:short
	//__m128i Z = _mm_mul_epu32(X, Y);// long long result[0] = A[0]*B[0];result[1] = A[2]*B[2];  A/B:unsigned int
	//__m128i Z = _mm_mulhi_epi16(X, Y);// int tmp = A[0]*B[0]; short result[0] = tmp[31:16];...A/B:short
	//__m128i Z = _mm_mulhi_epu16(X, Y);// int tmp = A[0]*B[0]; unsigned short result[0] = tmp[31:16];...A/B:unsigned short
	//__m128i Z = _mm_mullo_epi16(X, Y);// int tmp = A[0]*B[0]; short result[0] = tmp[15:0];...A/B:short
	//__m128i Z = _mm_sad_epu8(X, Y);// uchar tmp[0...7] = abs(A[0...7]-B[0...7]);dst[15:0]=sum(tmp[0]...tmp[7]);dst[63:16]=0;dst[79:64]=sum(tmp[8]..tmp[15]);dst[127:80]=0;A/B:unsigned char
	//__m128i Z = _mm_sub_epi8(X, Y);// char result[0] = A[0]-B[0];...A/B:char
	//__m128i Z = _mm_sub_epi16(X, Y);// short result[0] = A[0]-B[0];...A/B:short
	//__m128i Z = _mm_sub_epi32(X, Y);// int result[0] = A[0]-B[0];...A/B:int
	//__m128i Z = _mm_sub_epi64(X, Y);// long long result[0] = A[0]-B[0];...A/B:long long
	//__m128i Z = _mm_subs_epi16(X, Y);// ±¥ºÍ¼õ·¨
	//_mm_storeu_si128((__m128i*)result, Z);

// 	int a = 100;
// 	int b = 102;
// 	__m64 C = _mm_cvtsi32_si64(a);
// 	__m64 D = _mm_cvtsi32_si64(b);
// 	__m64 Z = _mm_mul_su32(C, D);
// 	int ret = _mm_cvtsi64_si32(Z);
	
// 	MonteCarlo();
// 	MonteCarloMp();

	Maxtrix<double> matrix;
	auto arrA = matrix.GenerateMaxtrix(100,100);
	auto arrB = matrix.GenerateMaxtrix(100,100);
	
// 	auto res = matrix.MultiplyV2(arrA, arrB, false);
// 	auto res_sse = matrix.MultiplyV2(arrA, arrB, true);
// 	auto sum_res = matrix.AccumulateMaxtrix(res);
// 	auto sum_res_sse = matrix.AccumulateMaxtrix(res_sse);
// 	assert(sum_res == sum_res);


	auto res = matrix.Multiply(arrA, arrB, false);
	auto res_sse = matrix.Multiply(arrA, arrB, true);
	auto res_mp = matrix.MultiplyMp(arrA, arrB, false);
	auto res_mp_sse = matrix.MultiplyMp(arrA, arrB, true);

	auto sum_res = matrix.AccumulateMaxtrix(res);
 	auto sum_res_sse = matrix.AccumulateMaxtrix(res_sse);
 	auto sum_res_mp = matrix.AccumulateMaxtrix(res_mp);
	auto sum_res_mp_sse = matrix.AccumulateMaxtrix(res_mp_sse);

	assert(sum_res == sum_res_sse);
	assert(sum_res_sse == sum_res_mp);
	assert(sum_res_mp == sum_res_mp_sse);
		
	cout << setprecision(6);
	
	system("pause");

	return 0;
}
