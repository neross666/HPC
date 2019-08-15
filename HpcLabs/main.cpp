#include <stdio.h>
#include "freshman.h"
#include "MatrixBase.h"


int main(int argc, char** argv)
{
	// ��ʼ������
	size_t M = 1 << 11;
	size_t N = 1 << 10;
	size_t S = 1 << 11;
	MatrixBase<int> mat_a, mat_b;
	mat_a.Create(M, N);
	mat_b.Create(N, S);

	mat_a.InitData();
	mat_b.InitData();

	//mat_a.Print();
	//mat_b.Print();


// #pragma region ����
// 	MatrixBase<int>* traverse_gpu = mat_a.GpuTraverse();
// 	*traverse_gpu == mat_a;
// 	delete traverse_gpu;
// #pragma endregion ����

// #pragma region �ӷ�
// 	// �ֱ�ʹ��CPU��GPU������˷�����
// 	MatrixBase<int>* add_cpu = nullptr;
// 	MatrixBase<int>* add_gpu = nullptr;
// 	add_cpu = mat_a.CpuAdd(mat_b);
// 	add_gpu = mat_a.GpuAdd(mat_b);
// 
// 	// У����
// 	*add_cpu == *add_gpu;
// 
// 	// ���پ���
// 	delete add_cpu;
// 	delete add_gpu;
// #pragma endregion �ӷ�

// #pragma region �˷�
// 	// �ֱ�ʹ��CPU��GPU������˷�����
// 	MatrixBase<int>* multi_cpu = nullptr;
// 	MatrixBase<int>* multi_gpu = nullptr;
// 	multi_cpu = mat_a.CpuMulti(mat_b);
// 	multi_gpu = mat_a.GpuMulti(mat_b);
// 
// 	
// 	// У����
// 	*multi_cpu == *multi_gpu;
// 
// 	// ���پ���
// 	delete multi_cpu;
// 	delete multi_gpu;
// #pragma endregion �˷�

// #pragma region ת��
// 	MatrixBase<int>* tran_cpu = mat_a.CpuTransposition();
// 	MatrixBase<int>* tran_gpu = mat_a.GpuTransposition();
// 	
// 	*tran_gpu == *tran_cpu;
//	delete tran_cpu;
// 	delete tran_gpu;
// #pragma endregion ת��

// #pragma region ƽ��
// 	MatrixBase<int>* smooth_cpu = mat_a.CpuSmooth();
// 	MatrixBase<int>* smooth_gpu = mat_a.GpuSmooth();
// 
// 	//smooth_cpu->Print();
// 	//smooth_gpu->Print();
// 
// 	*smooth_cpu == *smooth_gpu;
// 
// 	delete smooth_cpu;
// 	delete smooth_gpu;
// #pragma endregion ƽ��


#pragma region �ں�
 	MatrixBase<int>* blend_cpu = mat_a.CpuBlend(mat_b);
 	MatrixBase<int>* blend_gpu = mat_a.GpuBlend(mat_b);
 
 	//smooth_cpu->Print();
 	//smooth_gpu->Print();
 
 	*blend_cpu == *blend_gpu;
 
 	delete blend_cpu;
 	delete blend_gpu;
 #pragma endregion �ں�


		// �ȶ�ʱ��
	TimerSys::Profiler::Instance()->print();

	getchar();
	return 0;
}
