#pragma once
#include <vector>
#include <time.h>
#include <intrin.h>
#include <assert.h>
#include <omp.h>

#include "TimeClock.h"

#define NUM_THREADS 8

template <class T>
class Maxtrix
{
public:
	Maxtrix() {};
	~Maxtrix() {};

	vector<vector<T>> GenerateMaxtrix(long long nrow, long long ncol)
	{
		vector<vector<T>> rst;
		time_t t;
		srand((unsigned)time(&t));
		for (long long i = 0; i < nrow; i++)
		{
			vector<T> row;
			for (long long j = 0; j < ncol; j++)
			{
				row.push_back(10 * rand() / RAND_MAX);	// 0~100
			}
			rst.push_back(row);
		}

		return rst;
	}

	T AccumulateMaxtrix(const vector<vector<T>>& res)
	{
		T ret = 0;

		for (auto var : res)
		{
			ret += accumulate(var.begin(), var.end(), 0);
		}

		return ret;
	}

	vector<vector<double>> Multiply(const vector<vector<double>>& arrA, const vector<vector<double>>& arrB, bool sse)
	{
		string func_name(__FUNCTION__);
		if (sse)
			func_name += "_sse";
		TimeClock tc(func_name);

		int rowA = arrA.size();
		int colA = arrA[0].size();
		int rowB = arrB.size();
		int colB = arrB[0].size();
		vector<vector<double>> res;
		if (colA != rowB)
		{
			return res;
		}
		else
		{
			res.resize(rowA);
			for (int i = 0; i < rowA; ++i)
			{
				res[i].resize(colB);
			}

			for (int i = 0; i < rowA; ++i)
			{
				for (int j = 0; j < colB; ++j)
				{
					if (sse)
					{
						double result[2];
						__m128d X, Y, Z=_mm_setzero_pd();
						int nsize = 2 * (colA / 2);
						for (int k = 0; k < nsize; k += 2)
						{
							X = _mm_set_pd(*(arrA[i].data() + k), *(arrA[i].data() + k + 1));
							Y = _mm_set_pd(*(arrB[k].data() + j), *(arrB[k + 1].data() + j));
							X = _mm_mul_pd(X, Y);
							Z = _mm_add_pd(X, Z);
						}
						_mm_storeu_pd(result, Z);
						res[i][j] += result[0] + result[1];
						for (int n = nsize; n < colA; ++n)
						{
							res[i][j] += arrA[i][n] * arrB[n][j];
						}
					}
					else
					{
						for (int k = 0; k < colA; ++k)
						{
							res[i][j] += arrA[i][k] * arrB[k][j];
						}
					}
				}
			}
		}
		return res;
	}

	vector<vector<double>> MultiplyMp(const vector<vector<double>>& arrA, const vector<vector<double>>& arrB, bool sse)
	{
		string func_name(__FUNCTION__);
		if (sse)
			func_name += "_sse";
		TimeClock tc(func_name);

		int rowA = arrA.size();
		int colA = arrA[0].size();
		int rowB = arrB.size();
		int colB = arrB[0].size();
		vector<vector<double>> res;
		if (colA != rowB)
		{
			return res;
		}
		else
		{
			res.resize(rowA);
			for (int i = 0; i < rowA; ++i)
			{
				res[i].resize(colB);
			}

			omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < rowA; ++i)
			{
				for (int j = 0; j < colB; ++j)
				{
					if (sse)
					{
						double result[2];
						__m128d X, Y, Z = _mm_setzero_pd();
						int nsize = 2 * (colA / 2);
						for (int k = 0; k < nsize; k += 2)
						{
							X = _mm_set_pd(*(arrA[i].data() + k), *(arrA[i].data() + k + 1));
							Y = _mm_set_pd(*(arrB[k].data() + j), *(arrB[k + 1].data() + j));
							X = _mm_mul_pd(X, Y);
							Z = _mm_add_pd(X, Z);
						}
						_mm_storeu_pd(result, Z);
						res[i][j] += result[0] + result[1];
						for (int n = nsize; n < colA; ++n)
						{
							res[i][j] += arrA[i][n] * arrB[n][j];
						}
					}
					else
					{
						res[i][j] = 0.0;
						for (int k = 0; k < colA; ++k)
						{
							res[i][j] += arrA[i][k] * arrB[k][j];
						}
					}
				}
			}
		}
		return res;
	}

	vector<vector<T>> MultiplyV2(const vector<vector<T>>& arrA, const vector<vector<T>>& arrB, bool sse)
	{
		string func_name(__FUNCTION__);
		if (sse)
			func_name += "_sse";
		TimeClock tc(func_name);

		int rowA = arrA.size();
		int colA = arrA[0].size();
		int rowB = arrB.size();
		int colB = arrB[0].size();
		vector<vector<T>> res;
		if (colA != rowB)
		{
			return res;
		}
		else
		{
			res.resize(rowA);
			for (int i = 0; i < rowA; ++i)
			{
				res[i].resize(colB);
			}

			for (int i = 0; i < rowA - 1; ++i)
			{
				for (int j = 0; j < colB - 1; ++j)
				{
					if (sse)
					{
						if (typeid(arrA[0][0]) == typeid(int))
						{
							T result[4];
							__m128i X, Y;
							X = _mm_set_epi32(arrA[i][j], arrA[i][j + 1], arrA[i + 1][j], arrA[i + 1][j + 1]);
							Y = _mm_set_epi32(arrB[i][j], arrB[i][j + 1], arrB[i + 1][j], arrB[i + 1][j + 1]);
							X = _mm_mullo_epi32(X, Y);			// X*Y  非饱和乘法
							_mm_storeu_si128((__m128i*)result, X);

							res[i][j] = result[0] + result[1] + result[2] + result[3];
						}
						else if (typeid(arrA[0][0]) == typeid(float))
						{
							T result[4];
							__m128 X, Y;
							X = _mm_set_ps(arrA[i][j], arrA[i][j + 1], arrA[i + 1][j], arrA[i + 1][j + 1]);
							Y = _mm_set_ps(arrB[i][j], arrB[i][j + 1], arrB[i + 1][j], arrB[i + 1][j + 1]);
							X = _mm_mul_ps(X, Y);
							_mm_storeu_ps((float*)result, X);

							res[i][j] = result[0] + result[1] + result[2] + result[3];
						}
						else if (typeid(arrA[0][0]) == typeid(double))
						{
							__m128d X, Y;
							__m128d Z = _mm_setzero_pd();

							X = _mm_set_pd(arrA[i][j], arrA[i][j + 1]);
							Y = _mm_set_pd(arrB[i][j], arrB[i][j + 1]);								
							X = _mm_mul_pd(X, Y);
							Z = _mm_add_pd(X, Z);

							X = _mm_set_pd(arrA[i + 1][j], arrA[i + 1][j + 1]);
							Y = _mm_set_pd(arrB[i + 1][j], arrB[i + 1][j + 1]);
							X = _mm_mul_pd(X, Y);
							Z = _mm_add_pd(X, Z);
							
							T result[2];
							_mm_storeu_pd((double*)result, Z);
							res[i][j] = result[0] + result[1];
						}
						else
						{
							assert(false);
						}
					}
					else
					{
						T A[] = { arrA[i][j] ,arrA[i][j + 1], arrA[i + 1][j], arrA[i + 1][j + 1] };
						T B[] = { arrB[i][j] ,arrB[i][j + 1], arrB[i + 1][j], arrB[i + 1][j + 1] };
						for (int k = 0; k < sizeof(A) / sizeof(A[0]); ++k)
						{
							res[i][j] += A[k] * B[k];		// 非饱和操作
						}
					}
				}
			}
		}
		return res;
	}

private:

};

