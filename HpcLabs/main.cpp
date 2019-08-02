#include <stdio.h>
#include "freshman.h"
#include "MatrixBase.h"


int main(int argc, char** argv)
{
	// ��ʼ������
	size_t M = 1 << 5;
	size_t N = 1 << 5;
	size_t S = 1 << 6;
	MatrixBase<int> mat_a, mat_b;
	mat_a.Create(M, N);
	mat_b.Create(N, S);

	mat_a.InitData();
	mat_b.InitData();


#pragma region ����
	MatrixBase<int>* traverse_gpu = mat_a.GpuTraverse();
	traverse_gpu->Print();

	delete traverse_gpu;
#pragma endregion ����



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
	// 	// У����
	// 	*multi_cpu == *multi_gpu;
	// 
	// 	// ���پ���
	// 	delete multi_cpu;
	// 	delete multi_gpu;
	// #pragma endregion �˷�


		// �ȶ�ʱ��
	TimerSys::Profiler::Instance()->print();

	getchar();
	return 0;
}
