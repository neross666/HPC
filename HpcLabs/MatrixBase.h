#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "profiler/Profiler.h"
#include "kernel.cuh"
#include "freshman.h"

using namespace std;

template<class T>
class MatrixBase
{
public:
	MatrixBase() : m_rows(0), m_cols(0), m_pitch(0), m_pData(nullptr)
	{
	}

	virtual ~MatrixBase()
	{
		if (m_pData != nullptr)
		{
			_aligned_free(m_pData);
			m_pData = nullptr;
		}
	}

	bool IsEmpty()
	{
		return m_pData == nullptr;
	}

	void Create(size_t rows, size_t cols)
	{
		if (m_pData != nullptr)
		{
			_aligned_free(m_pData);
			m_pData = nullptr;
		}
		m_rows = rows;
		m_cols = cols;
		m_pitch = ((8 * sizeof(T)*cols + 31) >> 5) << 2;		// 4字节对齐后一行数据所占字节数
		m_pData = (T*)_aligned_malloc(m_pitch*rows, 4);		// 起始地址也要4字节对齐
		memset(m_pData, 0, m_pitch*rows);
		assert((unsigned long long)m_pData % 4 == 0);
	}

	void InitData()
	{
		time_t t;
		srand((unsigned)time(&t));
		for (size_t i = 0; i < m_rows; i++)
		{
			size_t offset = i * m_pitch;
			T* ptr = (T*)((char*)m_pData + offset);
			for (size_t j = 0; j < m_cols; j++)
			{
				//ptr[j] = T(rand() & 0xff);
				ptr[j] = T(49);
			}
		}
	}

	T* Col(size_t i)
	{
		return (T*)((char*)m_pData) + i;
	}

	T* Row(size_t i)
	{
		return (T*)((char*)m_pData + i * m_pitch);
	}

	void Print()
	{
		cout << "Matrix<" << m_rows << "," << m_cols << ">\n";
		for (size_t i = 0; i < m_rows; i++)
		{
			size_t offset = i * m_pitch;
			T* ptr = (T*)((char*)m_pData + offset);
			for (size_t j = 0; j < m_cols; j++)
			{
				cout << ptr[j] << " ";
			}
			printf("\n");
		}
	}

	bool operator==(const MatrixBase& a)
	{
		if (a.m_rows == m_rows && a.m_cols == m_cols)
		{
			double epsilon = 1.0E-8;
			for (size_t i = 0; i < m_rows; i++)
			{
				size_t offset = i * m_pitch;
				T* ptr = (T*)((char*)m_pData + offset);
				T* ptr_a = (T*)((char*)a.m_pData + offset);
				for (size_t j = 0; j < m_cols; j++)
				{
					if (abs(ptr_a[j] - ptr[j]) > epsilon)
					{
						cout << "check result failed. row[" << i << "],col[" << j << "]" << endl;
						cout << ptr_a[j] << "!=" << ptr[j] << endl;
						return false;
					}
				}
			}
			cout << "check result success.\n";
			return true;
		}
		cout << "check result failed. rows or cols not equal.\n";
		return false;
	}

	MatrixBase* GpuTraverse()
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		dst->Create(this->m_rows, this->m_cols);

		T* psrc = this->m_pData;
		T* pdst = dst->m_pData;
		T* pdst_d = nullptr;
		T* psrc_d = nullptr;
		size_t pitch = 0;
		CHECK(cudaMallocPitch(&psrc_d, &pitch, m_cols * sizeof(T), m_rows));
		CHECK(cudaMallocPitch(&pdst_d, &pitch, dst->m_cols * sizeof(T), dst->m_rows));
		CHECK(cudaMemcpy2D(psrc_d, pitch, psrc, m_pitch, m_cols * sizeof(T), m_rows, cudaMemcpyHostToDevice));
		CHECK(cudaDeviceSynchronize());

// 		cudaSharedMemConfig sConfig;
// 		CHECK(cudaDeviceGetSharedMemConfig(&sConfig));
// 		sConfig = cudaSharedMemBankSizeEightByte;
// 		CHECK(cudaDeviceSetSharedMemConfig(sConfig));
// 		CHECK(cudaDeviceGetSharedMemConfig(&sConfig));
// 		CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

		CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		{
			dim3 block(TILE_WIDTH, TILE_WIDTH);
			dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
			TIMING("TraverseKernelRow")
				TraverseKernelRow << <grid, block >> > (psrc_d, pdst_d, pitch, m_rows, m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
		{
			dim3 block(TILE_WIDTH, TILE_WIDTH);
			dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
			TIMING("TraverseKernelCol")
				TraverseKernelCol << <grid, block >> > (psrc_d, pdst_d, pitch, m_rows, m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
// 		{
// 			dim3 block(TILE_WIDTH, TILE_WIDTH);
// 			dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
// 			TIMING("TraverseKernelSMEM")
// 				TraverseKernelSMEM << <grid, block >> > (psrc_d, pdst_d, pitch, m_rows, m_cols);
// 			CHECK(cudaGetLastError());
// 			CHECK(cudaDeviceSynchronize());
// 		}
// 		{
// 			dim3 block(TILE_HEIGHT, TILE_WIDTH);
// 			dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
// 			TIMING("TraverseKernelSMEMRect")
// 				TraverseKernelSMEMRect << <grid, block >> > (psrc_d, pdst_d, pitch, m_rows, m_cols);
// 			CHECK(cudaGetLastError());
// 			CHECK(cudaDeviceSynchronize());
// 		}

		CHECK(cudaMemcpy2D(pdst, dst->m_pitch, pdst_d, pitch, dst->m_cols * sizeof(T), dst->m_rows, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());

		cudaFree(psrc_d);
		cudaFree(pdst_d);
		return dst;
	}

	MatrixBase* CpuAdd(const MatrixBase& a)
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		if (a.m_rows == m_rows &&
			a.m_cols == m_cols)
		{
			dst->Create(m_rows, m_cols);
			T* pself = m_pData;
			T* pa = a.m_pData;
			T* pc = dst->m_pData;
			{
				TIMING("CpuAdd");
				for (size_t i = 0; i < m_rows; i++)
				{
					size_t offset = i * m_pitch;
					T* ptr = (T*)((char*)pself + offset);
					T* ptr_a = (T*)((char*)pa + offset);
					T* ptr_c = (T*)((char*)pc + offset);
					for (size_t j = 0; j < m_cols; j++)
					{
						ptr_c[j] = ptr_a[j] + ptr[j];
					}
				}
			}
			{
				TIMING("CpuAdd UnRolling");
				for (size_t i = 0; i < m_rows; i++)
				{
					size_t offset = i * m_pitch;
					T* ptr = (T*)((char*)pself + offset);
					T* ptr_a = (T*)((char*)pa + offset);
					T* ptr_c = (T*)((char*)pc + offset);
					for (size_t j = 0; j < m_cols / 4; j += 4)	// 这里m_cols必须被4整除
					{
						ptr_c[j] = ptr_a[j] + ptr[j];
						ptr_c[j + 1] = ptr_a[j + 1] + ptr[j + 1];
						ptr_c[j + 2] = ptr_a[j + 2] + ptr[j + 2];
						ptr_c[j + 3] = ptr_a[j + 3] + ptr[j + 3];
					}
				}
			}

		}
		return dst;
	}

	MatrixBase* GpuAdd(const MatrixBase& a)
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		if (a.m_rows == m_rows &&
			a.m_cols == m_cols)
		{
			initDevice(0);

			dst->Create(m_rows, m_cols);
			T* ps = m_pData;
			T* pa = a.m_pData;
			T* pc = dst->m_pData;
			size_t nsize = m_rows * m_cols;

			T* pa_d = nullptr;
			T* ps_d = nullptr;
			T* pc_d = nullptr;
			size_t pitch = 0;
			CHECK(cudaMallocPitch(&ps_d, &pitch, m_cols * sizeof(T), m_rows));
			CHECK(cudaMallocPitch(&pa_d, &pitch, m_cols * sizeof(T), m_rows));
			CHECK(cudaMallocPitch(&pc_d, &pitch, m_cols * sizeof(T), m_rows));
			CHECK(cudaMemcpy2D(ps_d, pitch, ps, m_pitch, m_cols * sizeof(T), m_rows, cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy2D(pa_d, pitch, pa, m_pitch, m_cols * sizeof(T), m_rows, cudaMemcpyHostToDevice));
			CHECK(cudaDeviceSynchronize());


			{
				dim3 block(1024);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("Warmup")
					WarmupAdd << <grid, block >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(1024);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(32,32) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32, 32);
				dim3 grid((m_cols - 1) / block.x + 1, (m_rows - 1) / block.y + 1);
				TIMING("AddKernelV2 << <block(32,32) >> >")
					AddKernelV2 << <grid, block >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(1024);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("Add4UnRollingKernelV1 << <block(1024) >> >")
					Add4UnRollingKernelV1 << <grid.x / 4, block.x >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(1024);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("Add4UnRollingKernelV2 << <block(1024) >> >")
					Add4UnRollingKernelV2 << <grid.x / 4, block.x >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32, 32);
				dim3 grid((m_cols / 4 - 1) / block.x + 1, (m_rows - 1) / block.y + 1);
				TIMING("Add4UnRollingKernelV3 << <block(32,32) >> >")
					Add4UnRollingKernelV3 << <grid, block >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32, 32);
				dim3 grid((m_cols - 1) / block.x + 1, (m_rows / 4 - 1) / block.y + 1);
				TIMING("Add4UnRollingKernelV4 << <block(32,32) >> >")
					Add4UnRollingKernelV4 << <grid, block >> > (pa_d, ps_d, pc_d, pitch, m_rows, m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}


			/*
#pragma region Grid 1D; Block 1D
			{
				dim3 block(16);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(16) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(32) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(64);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(64) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(128);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(128) >> >")
				AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(256);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(256) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(512);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("AddKernel << <block(512) >> >")
					AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
#pragma endregion Grid 1D; Block 1D
			*/

			/*
#pragma region Grid 2D; Block 2D
			{
				dim3 block(32, 32);
				dim3 grid((m_rows - 1) / block.x + 1, (m_cols - 1) / block.y + 1);
				TIMING("AddKernel << <block(32,32) >> >")
				AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32, 16);
				dim3 grid((m_rows - 1) / block.x + 1, (m_cols - 1) / block.y + 1);
				TIMING("AddKernel << <block(32,16) >> >")
				AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(16, 32);
				dim3 grid((m_rows - 1) / block.x + 1, (m_cols - 1) / block.y + 1);
				TIMING("AddKernel << <block(16,32) >> >")
				AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(16, 16);
				dim3 grid((m_rows - 1) / block.x + 1, (m_cols - 1) / block.y + 1);
				TIMING("AddKernel << <block(16,16) >> >")
				AddKernel << <grid, block >> > (pa_d, ps_d, pc_d, nsize);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
#pragma endregion Grid 2D; Block 2D
			*/


			CHECK(cudaMemcpy2D(pc, m_pitch, pc_d, pitch, m_cols * sizeof(T), m_rows, cudaMemcpyDeviceToHost));
			CHECK(cudaDeviceSynchronize());

			cudaFree(pa_d);
			cudaFree(ps_d);
			cudaFree(pc_d);
			cudaDeviceReset();
		}
		return dst;
	}

	MatrixBase* CpuMulti(const MatrixBase& b)
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		if (this->m_cols == b.m_rows)
		{
			dst->Create(this->m_rows, b.m_cols);
			T* pself = m_pData;
			T* pb = b.m_pData;
			T* pdst = dst->m_pData;
			{
				TIMING("CppuMulti");
				for (size_t i = 0; i < this->m_rows; i++)
				{
					size_t offset_self = i * this->m_pitch;
					size_t offset_dst = i * dst->m_pitch;
					T* ptr_self = (T*)((char*)pself + offset_self);
					T* ptr_dst = (T*)((char*)pdst + offset_dst);
					for (size_t j = 0; j < b.m_cols; j++)
					{
						T sum = T(0);
						for (size_t k = 0; k < this->m_cols; k++)
						{
							size_t offset_b = k * b.m_pitch;
							T* ptr_b = (T*)((char*)pb + offset_b);
							sum += ptr_self[k] * ptr_b[j];
						}
						ptr_dst[j] = sum;
					}
				}
			}
		}
		return dst;
	}

	MatrixBase* GpuMulti(const MatrixBase& b)
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		if (this->m_cols == b.m_rows)
		{
			initDevice(0);

			dst->Create(this->m_rows, b.m_cols);
			T* pself = m_pData;
			T* pb = b.m_pData;
			T* pdst = dst->m_pData;
			size_t nsize = dst->m_rows * dst->m_cols;

			T* pb_d = nullptr;
			T* pself_d = nullptr;
			T* pdst_d = nullptr;
			size_t pitch_self = 0;
			size_t pitch_b = 0;
			size_t pitch_dst = 0;
			CHECK(cudaMallocPitch(&pself_d, &pitch_self, this->m_cols * sizeof(T), this->m_rows));
			CHECK(cudaMallocPitch(&pb_d, &pitch_b, b.m_cols * sizeof(T), b.m_rows));
			CHECK(cudaMallocPitch(&pdst_d, &pitch_dst, dst->m_cols * sizeof(T), dst->m_rows));
			CHECK(cudaMemcpy2D(pself_d, pitch_self, pself, this->m_pitch, this->m_cols * sizeof(T), this->m_rows, cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy2D(pb_d, pitch_b, pb, b.m_pitch, b.m_cols * sizeof(T), b.m_rows, cudaMemcpyHostToDevice));
			CHECK(cudaDeviceSynchronize());

			{
				dim3 block(1024);
				dim3 grid((nsize - 1) / block.x + 1);
				TIMING("WarmupMulti")
					WarmupMulti << <grid, block >> > (pself_d, pb_d, pdst_d,
						pitch_self, pitch_b, pitch_dst, this->m_rows, this->m_cols, b.m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(32, 16);
				dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
				TIMING("MultiKernel")
					MultiKernel << <grid, block >> > (pself_d, pb_d, pdst_d,
						pitch_self, pitch_b, pitch_dst, this->m_rows, this->m_cols, b.m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
			{
				dim3 block(TILE_WIDTH, TILE_WIDTH);
				dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
				TIMING("MultiKernelTile")
					MultiKernelTile << <grid, block>> > (pself_d, pb_d, pdst_d,
						pitch_self, pitch_b, pitch_dst, this->m_rows, this->m_cols, b.m_cols);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}

			CHECK(cudaMemcpy2D(pdst, dst->m_pitch, pdst_d, pitch_dst, dst->m_cols * sizeof(T), dst->m_rows, cudaMemcpyDeviceToHost));
			CHECK(cudaDeviceSynchronize());

			cudaFree(pb_d);
			cudaFree(pself_d);
			cudaFree(pdst_d);
			cudaDeviceReset();
		}
		return dst;
	}

	MatrixBase* CpuTransposition()
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		dst->Create(this->m_cols, this->m_rows);
		T* ptr_s = m_pData;
		T* ptr_d = dst->m_pData;
		{
			TIMING("CpuTransposition");
			for (size_t i = 0; i < dst->m_rows; i++)
			{
				size_t offset_d = i * dst->m_pitch;
				T* ptr_dr = (T*)((char*)ptr_d + offset_d);
				for (size_t j = 0; j < dst->m_cols; j++)
				{
					size_t offset_s = j * this->m_pitch;
					T* ptr_sr = (T*)((char*)ptr_s + offset_s);
					ptr_dr[j] = ptr_sr[i];
				}
			}
		}

		return dst;
	}

	MatrixBase* GpuTransposition()
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		dst->Create(this->m_cols, this->m_rows);

		T* psrc = this->m_pData;
		T* pdst = dst->m_pData;
		T* pdst_d = nullptr;
		T* psrc_d = nullptr;
		size_t pitch_s = 0;
		size_t pitch_d = 0;
 		CHECK(cudaMallocPitch(&psrc_d, &pitch_s, m_cols * sizeof(T), m_rows));
		CHECK(cudaMallocPitch(&pdst_d, &pitch_d, dst->m_cols * sizeof(T), dst->m_rows));
		CHECK(cudaMemcpy2D(psrc_d, pitch_s, psrc, m_pitch, m_cols*sizeof(T), m_rows, cudaMemcpyHostToDevice));
 		CHECK(cudaDeviceSynchronize());

		{
			CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
			dim3 block(TILE_WIDTH, TILE_WIDTH);
			dim3 grid((dst->m_cols - 1) / block.x + 1, (dst->m_rows - 1) / block.y + 1);
			TIMING("TranspositionKernel")
				TranspositionKernel << <grid, block >> > (psrc_d, pdst_d, pitch_s, pitch_d, dst->m_rows, dst->m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
		{			
			CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
			dim3 block(TILE_HEIGHT, TILE_WIDTH);
			dim3 grid((m_cols - 1) / block.x + 1, (m_rows - 1) / block.y + 1);
			TIMING("TranspositionKernelSMEM")
				TranspositionKernelSMEM << <grid, block >> > (psrc_d, pdst_d, pitch_s, pitch_d, m_rows, m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}

		CHECK(cudaMemcpy2D(pdst, dst->m_pitch, pdst_d, pitch_d, dst->m_cols * sizeof(T), dst->m_rows, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());

		cudaFree(psrc_d);
		cudaFree(pdst_d);
		return dst;
	}

	MatrixBase* CpuSmooth()
	{
		float ceof[7][7];
		for (int i=0; i<7; i++)
		{
			for (int j=0; j<7; j++)
			{
				ceof[i][j] = 0.5f;
			}
		}


		MatrixBase<T>* dst = new MatrixBase<T>;
		dst->Create(this->m_rows, this->m_cols);
		T* ptr_s = m_pData;
		T* ptr_d = dst->m_pData;
		{
			TIMING("CpuSmooth");
			for (size_t r = 0; r < dst->m_rows; r++)
			{
				size_t offset_s = r * this->m_pitch;
				size_t offset_d = r * dst->m_pitch;
				T* ptr_dr = (T*)((char*)ptr_d + offset_d);
				for (size_t c = 0; c < dst->m_cols; c++)
				{
					float tmp = 0.0f;
					for (int i = -3; i <= 3; i++)
					{
						size_t rr = r;
						if (r + i < 0 || r + i >= m_rows)
							rr = r - i;
						size_t offset_s = rr * this->m_pitch;
						T* ptr_sr = (T*)((char*)ptr_s + offset_s);
						for (int j = -3; j <= 3; j++)
						{
							size_t cc = c;
							if (c + j < 0 || c + j >= m_cols)
								cc = c - j;
							tmp += ceof[i + 3][j + 3] * ptr_sr[cc];
						}
					}
					ptr_dr[c] = (T)(tmp/49);
				}
			}
		}

		return dst;
	}

	MatrixBase* GpuSmooth()
	{
		MatrixBase<T>* dst = new MatrixBase<T>;
		dst->Create(this->m_rows, this->m_cols);

		T* psrc = this->m_pData;
		T* pdst = dst->m_pData;
		T* pdst_d = nullptr;
		T* psrc_d = nullptr;
		size_t pitch = 0;
		CHECK(cudaMallocPitch(&psrc_d, &pitch, m_cols * sizeof(T), m_rows));
		CHECK(cudaMallocPitch(&pdst_d, &pitch, m_cols * sizeof(T), m_rows));
		CHECK(cudaMemcpy2D(psrc_d, pitch, psrc, m_pitch, m_cols * sizeof(T), m_rows, cudaMemcpyHostToDevice));
		CHECK(cudaDeviceSynchronize());

		float ceof[49];
		for (int i = 0; i < 49; i++)
			ceof[i] = 0.5f;

		float* pcoef_d = nullptr;
		CHECK(cudaMalloc(&pcoef_d, 49 * sizeof(float)));
		CHECK(cudaMemcpy(pcoef_d, ceof, 49 * sizeof(float), cudaMemcpyHostToDevice));

		CHECK(cudaMemcpyToSymbol(coef, ceof, 49 * sizeof(float), 0, cudaMemcpyHostToDevice));

		{
			dim3 block(TILE_WIDTH, TILE_WIDTH);
			dim3 grid((m_cols - 1) / block.x + 1, (m_rows - 1) / block.y + 1);
			TIMING("SmoothKernelCMEM")
				SmoothKernelCMEM << <grid, block >> > (psrc_d, pdst_d, pitch, m_rows, m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
		{
			dim3 block(TILE_WIDTH, TILE_WIDTH);
			dim3 grid((m_cols - 1) / block.x + 1, (m_rows - 1) / block.y + 1);
			TIMING("SmoothKernel")
				SmoothKernel << <grid, block >> > (psrc_d, pdst_d, pcoef_d, pitch, m_rows, m_cols);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}

		CHECK(cudaMemcpy2D(pdst, m_pitch, pdst_d, pitch, m_cols * sizeof(T), m_rows, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());

		cudaFree(psrc_d);
		cudaFree(pdst_d);
		return dst;
	}

protected:
	size_t m_rows;
	size_t m_cols;
	size_t m_pitch;

	T* m_pData;
};