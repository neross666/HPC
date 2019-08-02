#include <stdio.h>
#include "freshman.h"
#include "MatrixBase.h"


int main(int argc, char** argv)
{
	// 初始化矩阵
	size_t M = 1 << 5;
	size_t N = 1 << 5;
	size_t S = 1 << 6;
	MatrixBase<int> mat_a, mat_b;
	mat_a.Create(M, N);
	mat_b.Create(N, S);

	mat_a.InitData();
	mat_b.InitData();


#pragma region 遍历
	MatrixBase<int>* traverse_gpu = mat_a.GpuTraverse();
	traverse_gpu->Print();

	delete traverse_gpu;
#pragma endregion 遍历



	// #pragma region 加法
	// 	// 分别使用CPU和GPU做矩阵乘法运算
	// 	MatrixBase<int>* add_cpu = nullptr;
	// 	MatrixBase<int>* add_gpu = nullptr;
	// 	add_cpu = mat_a.CpuAdd(mat_b);
	// 	add_gpu = mat_a.GpuAdd(mat_b);
	// 
	// 	// 校验结果
	// 	*add_cpu == *add_gpu;
	// 
	// 	// 销毁矩阵
	// 	delete add_cpu;
	// 	delete add_gpu;
	// #pragma endregion 加法

	// #pragma region 乘法
	// 	// 分别使用CPU和GPU做矩阵乘法运算
	// 	MatrixBase<int>* multi_cpu = nullptr;
	// 	MatrixBase<int>* multi_gpu = nullptr;
	// 	multi_cpu = mat_a.CpuMulti(mat_b);
	// 	multi_gpu = mat_a.GpuMulti(mat_b);
	// 	
	// 	// 校验结果
	// 	*multi_cpu == *multi_gpu;
	// 
	// 	// 销毁矩阵
	// 	delete multi_cpu;
	// 	delete multi_gpu;
	// #pragma endregion 乘法


		// 比对时间
	TimerSys::Profiler::Instance()->print();

	getchar();
	return 0;
}
