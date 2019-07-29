#pragma once
#include <string>

using namespace std;
typedef unsigned long DWORD;

class CpuID
{
public:
	string GetVID();
	string GetBrand();
	bool IsHyperThreading();
	bool IsEST();
	bool IsMMX();
	bool IsSSE();
	bool IsSSE2();
	bool IsSSE3();
	bool IsSSE4_1();
	bool IsSSE4_2();
private:
	void Executecpuid(DWORD eax); // 用来实现cpuid 
	DWORD m_eax; // 存储返回的eax 
	DWORD m_ebx; // 存储返回的ebx 
	DWORD m_ecx; // 存储返回的ecx 
	DWORD m_edx; // 存储返回的edx 
};