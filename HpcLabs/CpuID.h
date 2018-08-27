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
	void Executecpuid(DWORD eax); // ����ʵ��cpuid 
	DWORD m_eax; // �洢���ص�eax 
	DWORD m_ebx; // �洢���ص�ebx 
	DWORD m_ecx; // �洢���ص�ecx 
	DWORD m_edx; // �洢���ص�edx 
};