#include "CpuID.h" 
#include <mmintrin.h>		// MMX  
#include <xmmintrin.h>		// SSE(include mmintrin.h)  
#include <emmintrin.h>		// SSE2(include xmmintrin.h)  
#include <pmmintrin.h>		// SSE3(include emmintrin.h)  
#include <tmmintrin.h>		// SSSE3(include pmmintrin.h)  
#include <smmintrin.h>		// SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>		// SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>		// AES(include nmmintrin.h)  
#include <immintrin.h>		// AVX(include wmmintrin.h)  
#include <intrin.h>			// (include immintrin.h)


void CpuID::Executecpuid(DWORD veax)
{ 
	// 因为嵌入式的汇编代码不能识别类成员变量,所以定义四个临时变量作为过渡 
	DWORD deax;
	DWORD debx;
	DWORD decx;
	DWORD dedx;
	__asm {
		mov eax, veax;		// 将输入参数移入eax
		cpuid;				// 执行cpuid
		mov deax, eax;		// 以下四行代码把寄存器中的变量存入临时变量
		mov debx, ebx
		mov decx, ecx
		mov dedx, edx
	}
	m_eax = deax;			// 把临时变量中的内容放入类成员变量 
	m_ebx = debx;
	m_ecx = decx;
	m_edx = dedx;
}

string CpuID::GetVID()
{
	char cVID[13]; // 字符串，用来存储制造商信息
	memset(cVID, 0, 13); // 把数组清0 
	Executecpuid(0); // 执行cpuid指令，使用输入参数 eax = 0 
	memcpy(cVID, &m_ebx, 4); // 复制前四个字符到数组 
	memcpy(cVID + 4, &m_edx, 4); // 复制中间四个字符到数组 
	memcpy(cVID + 8, &m_ecx, 4); // 复制最后四个字符到数组 
	return string(cVID); // 以string的形式返回 
}

string CpuID::GetBrand()
{
	const DWORD BRANDID = 0x80000002; // 从0x80000002开始，到0x80000004结束 
	char cBrand[49]; // 用来存储商标字符串，48个字符 
	memset(cBrand, 0, 49); // 初始化为0 
	for (DWORD i = 0; i < 3; i++) // 依次执行3个指令 
	{
		Executecpuid(BRANDID + i);
		memcpy(cBrand + i * 16, &m_eax, 16); // 每次执行结束后，保存四个寄存器里的asc码到数组 
	} // 由于在内存中，m_eax, m_ebx, m_ecx, m_edx是连续排列 // 所以可以直接以内存copy的方式进行保存 
	return string(cBrand); // 以string的形式返回 
}

bool CpuID::IsHyperThreading() // 判断是否支持hyper-threading 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1 
	bool res = m_edx & (1 << 28);
	if (res) 
		printf("support HyperThreading\n");
	else 
		printf("don't support HyperThreading\n");
	return res; // 返回edx的bit 28 
}

bool CpuID::IsEST() // 判断是否支持speed step 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1 
	return m_ecx & (1 << 7); // 返回ecx的bit 7 
} 

bool CpuID::IsMMX()
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1
	bool res = m_edx & (1 << 23);
	if (res) 
		printf("support MMX/n");
	else 
		printf("don't support MMX/n");
	return res; // 返回edx的bit 23 
} 

bool CpuID::IsSSE() // 判断是否支持SSE 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1
	bool res = m_edx & (1 << 25);
	if (res) 
		printf("support SSE\n");
	else 
		printf("don't support SSE\n");
	return res;
}

bool CpuID::IsSSE2() // 判断是否支持SSE2 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1 
	bool res = m_edx & (1 << 26);
	if (res) 
		printf("support SSE2\n");
	else 
		printf("don't support SSE2\n");
	return res;
}

bool CpuID::IsSSE3() // 判断是否支持SSE3 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1
	bool res = m_ecx;
	if (res) 
		printf("support SSE3\n");
	else 
		printf("don't support SSE3\n");
	return res;
}

bool CpuID::IsSSE4_1() // 判断是否支持SSE4.1 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1
	bool res = m_ecx & (1 << 19);
	if (res) 
		printf("support SSE4.1\n");
	else 
		printf("don't support SSE4.1\n");
	return res;
}

bool CpuID::IsSSE4_2() // 判断是否支持SSE4.2 
{
	Executecpuid(1); // 执行cpuid指令，使用输入参数 eax = 1 
	bool res = m_ecx & (1<<20); 
	if (res) 
		printf("support SSE4.2\n");
	else 
		printf("don't support SSE4.2\n");
	return res;
}