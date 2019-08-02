#pragma once
#include <cuda_runtime.h>


#pragma region �ӷ�����
// GPUԤ�ȣ�ʹ����ͨ���㷽ʽ
__global__ void WarmupAdd(int* src1, int* src2, int*dst, size_t pitch, size_t rows, size_t cols);

// ʹ����ͨ���㷽ʽ��ÿ���������ҽ���һ���̴߳������߳��������Ϊ���Ե�
template<class T>
__global__ void AddKernel(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// grid 1D; block 1D;
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// grid 2D; block 1D;
	//unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
	//unsigned int tid = bid * blockDim.x + threadIdx.x;

	// grid 2D; block 2D;
	//unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
	//unsigned int tid = bid * (blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;

	// ��ȡ���̺߳Ŷ�Ӧ�������ڶ�ά�����е���������
	unsigned int idx_r = tid / cols;
	unsigned int idx_c = tid % cols;
	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// ʹ����ͨ���㷽ʽ��ÿ���������ҽ���һ���̴߳������߳��������Ϊ��ά��
template<class T>
__global__ void AddKernelV2(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// �߳�y����������Ӧ�����з����������߳�x����������Ӧ�����з�������
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// ʹ��4��ѭ��չ�����㷽ʽһ��ͬһ�е�����4�����ݱ�һ���̴߳������߳��������Ϊ���Ե�
template<class T>
__global__ void Add4UnRollingKernelV1(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// 	{
	// 		// grid 1D; block 1D; ����ʹ��һά�����洢��ʽ������������֮�䲻���ڼ�϶
	// 		unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// 		unsigned int idx = 4 * tid;
	// 		if (idx + 3 < size)
	// 		{
	// 			dst[idx] = src1[idx] + src2[idx];
	// 			dst[idx + 1] = src1[idx + 1] + src2[idx + 1];
	// 			dst[idx + 2] = src1[idx + 2] + src2[idx + 2];
	// 			dst[idx + 3] = src1[idx + 3] + src2[idx + 3];
	// 		}
	// 	}

		// ����ʹ��һά�������洢��ʽ������������֮����ڼ�϶
		// ��ȡ���̺߳Ŷ�Ӧ�������ڶ�ά�����е���������
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = tid / (cols / 4);
	unsigned int idx_c = 4 * (tid % (cols / 4));		// ����Ϊ4�ı���	

	if (idx_r < rows && (idx_c + 3) < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
		ptr_d[idx_c + 1] = ptr_s1[idx_c + 1] + ptr_s2[idx_c + 1];
		ptr_d[idx_c + 2] = ptr_s1[idx_c + 2] + ptr_s2[idx_c + 2];
		ptr_d[idx_c + 3] = ptr_s1[idx_c + 3] + ptr_s2[idx_c + 3];
	}
}

// ʹ��4��ѭ��չ�����㷽ʽ����ͬһ�е�����4�����ݱ�һ���̴߳������߳��������Ϊ���Ե�
template<class T>
__global__ void Add4UnRollingKernelV2(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// grid 1D; block 1D; ����ʹ��һά�����洢��ʽ������������֮�䲻���ڼ�϶
//	unsigned int tid = threadIdx.x + 4*blockIdx.x*blockDim.x;
// 	if (tid + 3 * blockDim.x < size)
// 	{
// 		dst[tid] = src1[tid] + src2[tid];
// 		dst[tid + blockDim.x] = src1[tid + blockDim.x] + src2[tid + blockDim.x];
// 		dst[tid + 2 * blockDim.x] = src1[tid + 2 * blockDim.x] + src2[tid + 2 * blockDim.x];
// 		dst[tid + 3 * blockDim.x] = src1[tid + 3 * blockDim.x] + src2[tid + 3 * blockDim.x];
// 	}

	// ����ʹ��һά�������洢��ʽ������������֮����ڼ�϶
	// ��ȡ���̺߳Ŷ�Ӧ�������ڶ�ά�����е���������
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = 4 * (tid / cols);
	unsigned int idx_c = tid % cols;
	if (idx_r + 3 < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 1) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 2) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 3) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// ʹ��4��ѭ��չ�����㷽ʽ����ͬһ�е�����4�����ݱ�һ���̴߳������߳��������Ϊ��ά��
template<class T>
__global__ void Add4UnRollingKernelV3(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// �߳�y����������Ӧ�����з����������߳�x����������Ӧ4��������ʼ����
	unsigned int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idx_r = tid_y;
	unsigned int idx_c = tid_x << 2;	// ����Ϊ4�ı���

	if (idx_r < rows && (idx_c + 3) < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
		ptr_d[idx_c + 1] = ptr_s1[idx_c + 1] + ptr_s2[idx_c + 1];
		ptr_d[idx_c + 2] = ptr_s1[idx_c + 2] + ptr_s2[idx_c + 2];
		ptr_d[idx_c + 3] = ptr_s1[idx_c + 3] + ptr_s2[idx_c + 3];
	}
}

// ʹ��4��ѭ��չ�����㷽ʽ�ģ�ͬһ�е�����4�����ݱ�һ���̴߳������߳��������Ϊ��ά��
template<class T>
__global__ void Add4UnRollingKernelV4(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// �߳�y����������Ӧ�����з����������߳�x����������Ӧ4��������ʼ����
	unsigned int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idx_r = tid_y << 2;		// ����Ϊ4�ı���
	unsigned int idx_c = tid_x;
	if (idx_r + 3 < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 1) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 2) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 3) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}
#pragma endregion �ӷ�����


#pragma region �˷�����
__global__ void WarmupMulti(int* src1, int* src2, int*dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S);

template<class T>
__global__ void MultiKernel(T* src1, T* src2, T*dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx_r < M && idx_c < S)
	{
		size_t offset_s1 = idx_r * pitch_src1;
		size_t offset_dst = idx_r * pitch_dst;
		T* ptr_s1 = (T*)((char*)src1 + offset_s1);
		T* ptr_d = (T*)((char*)dst + offset_dst);

		ptr_d[idx_c] = 0;
		for (size_t i = 0; i < N; i++)
		{
			size_t offset_s2 = i * pitch_src2;
			T* ptr_s2 = (T*)((char*)src2 + offset_s2);
			ptr_d[idx_c] += ptr_s1[i] * ptr_s2[idx_c];
		}
	}
}
#pragma endregion �˷�����


// bank_no = (addr/4)%32
// nvprof  --metrics shared_load_transactions_per_request
// nvprof  --metrics shared_store_transactions_per_request
template<class T>
__global__ void TraverseKernelSMEM(T* dst, size_t pitch, size_t rows, size_t cols)
{
	__shared__ T tile[32][32];
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
		unsigned int tid = bid * (blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
		tile[threadIdx.y][threadIdx.x] = tid;				// ������д
		__syncthreads();
		
		size_t offset = idx_r * pitch;
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = tile[threadIdx.y][threadIdx.x];		// ������д
	}
}

template<class T>
__global__ void TraverseKernelSMEMRect(T* dst, size_t pitch, size_t rows, size_t cols)
{
	__shared__ T tile[16][32+2];					// ��䣬����bank��ͻ
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
		unsigned int tid = bid * (blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x;

		tile[threadIdx.y][threadIdx.x] = tid;		// ������д
		__syncthreads();

		unsigned int irow = idx / blockDim.y;
		unsigned int icol = idx % blockDim.y;

		size_t offset = idx_r * pitch;
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = tile[icol][irow];			// �������
	}
}
